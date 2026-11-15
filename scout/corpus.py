"""Local corpus with BM25 retrieval.

Why a local corpus instead of a web search API: a research agent is only
trustworthy if its evidence is reproducible. A corpus checked into the repo means
the same question yields the same passages, the citation checks can be asserted in
tests, and a reviewer can see exactly what the agent had to work with.

BM25 rather than embeddings: it needs no model download, runs in milliseconds, and
its failure mode is visible (you can read the term matches). For a corpus of a few
dozen documents that is the right trade — dense retrieval earns its complexity at
a scale this project does not have.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

_WORD = re.compile(r"[a-z0-9_]+")
_CJK = re.compile(r"[\u4e00-\u9fff]+")

# BM25 parameters. k1 controls term-frequency saturation, b controls length
# normalisation. These are the standard defaults from the literature.
K1 = 1.4
B = 0.72
CHUNK_CHARS = 420
CHUNK_OVERLAP = 80


def tokenize(text: str) -> list[str]:
    """English words plus CJK bigrams.

    Character bigrams are a crude but effective substitute for Chinese word
    segmentation: they need no dictionary, and for retrieval the loss is small
    because matching is still local and rare enough to be discriminative.
    """
    lowered = text.lower()
    tokens = [m.group() for m in _WORD.finditer(lowered)]
    for match in _CJK.finditer(lowered):
        segment = match.group()
        if len(segment) == 1:
            tokens.append(segment)
        else:
            tokens.extend(segment[i : i + 2] for i in range(len(segment) - 1))
    return tokens


@dataclass
class Document:
    id: str
    title: str
    text: str
    source: str = ""

    @classmethod
    def from_markdown(cls, path: Path) -> "Document":
        raw = path.read_text(encoding="utf-8")
        lines = raw.splitlines()
        title = path.stem
        body_lines = []
        for line in lines:
            if line.startswith("# ") and title == path.stem:
                title = line[2:].strip()
                continue
            body_lines.append(line)
        return cls(id=path.stem, title=title, text="\n".join(body_lines).strip(), source=path.name)


@dataclass
class Passage:
    """A retrievable chunk. `doc_id` and `chunk` together identify it for citation."""

    doc_id: str
    title: str
    text: str
    chunk: int
    score: float = 0.0

    @property
    def ref(self) -> str:
        return f"{self.doc_id}#{self.chunk}"


@dataclass
class Corpus:
    documents: list[Document] = field(default_factory=list)
    _chunks: list[Passage] = field(default_factory=list, repr=False)
    _df: Counter = field(default_factory=Counter, repr=False)
    _avg_len: float = 0.0
    _chunk_tokens: list[list[str]] = field(default_factory=list, repr=False)

    @classmethod
    def load(cls, directory: str | Path) -> "Corpus":
        path = Path(directory)
        docs = [Document.from_markdown(p) for p in sorted(path.glob("*.md"))]
        corpus = cls(documents=docs)
        corpus._build_index()
        return corpus

    def _build_index(self) -> None:
        self._chunks = []
        for doc in self.documents:
            for i, text in enumerate(_split(doc.text, CHUNK_CHARS, CHUNK_OVERLAP)):
                self._chunks.append(Passage(doc_id=doc.id, title=doc.title, text=text, chunk=i))

        self._chunk_tokens = [tokenize(c.text) for c in self._chunks]
        self._df = Counter()
        for tokens in self._chunk_tokens:
            for term in set(tokens):
                self._df[term] += 1
        self._avg_len = (
            sum(len(t) for t in self._chunk_tokens) / len(self._chunk_tokens)
            if self._chunk_tokens
            else 0.0
        )

    def search(self, query: str, k: int = 5) -> list[Passage]:
        query_terms = tokenize(query)
        if not query_terms or not self._chunks:
            return []

        total = len(self._chunks)
        scores: list[float] = []
        for tokens in self._chunk_tokens:
            counts = Counter(tokens)
            length = len(tokens) or 1
            score = 0.0
            for term in query_terms:
                tf = counts.get(term, 0)
                if not tf:
                    continue
                idf = math.log(1 + (total - self._df[term] + 0.5) / (self._df[term] + 0.5))
                denom = tf + K1 * (1 - B + B * length / (self._avg_len or 1))
                score += idf * tf * (K1 + 1) / denom
            scores.append(score)

        ranked = sorted(range(total), key=lambda i: scores[i], reverse=True)[:k]
        results = []
        for idx in ranked:
            if scores[idx] <= 0:
                continue
            passage = self._chunks[idx]
            results.append(
                Passage(passage.doc_id, passage.title, passage.text, passage.chunk, scores[idx])
            )
        return results

    def get(self, ref: str) -> Passage | None:
        """Resolve a citation like 'doc-id#2' back to the exact passage."""
        if "#" not in ref:
            return None
        doc_id, _, chunk = ref.partition("#")
        try:
            index = int(chunk)
        except ValueError:
            return None
        for passage in self._chunks:
            if passage.doc_id == doc_id and passage.chunk == index:
                return passage
        return None


def _split(text: str, size: int, overlap: int) -> list[str]:
    """Paragraph-aware chunking with overlap, so a citation never starts mid-thought."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= size:
            current = f"{current}\n\n{para}".strip()
            continue
        if current:
            chunks.append(current)
        # A single oversized paragraph still has to be split.
        if len(para) > size:
            start = 0
            while start < len(para):
                chunks.append(para[start : start + size])
                start += size - overlap
            current = ""
        else:
            current = para
    if current:
        chunks.append(current)
    return chunks
