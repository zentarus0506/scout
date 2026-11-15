"""Research orchestration.

Pipeline: decompose -> retrieve -> extract claims with citations -> verify every
claim -> synthesize a report that carries its own audit trail.

The ordering matters. Verification runs on the claims *before* they are written
into prose, so a fabricated sentence never reaches the report looking confident.
Each finding keeps its verdict attached, which means the report can show its own
weaknesses instead of hiding them behind a citation marker.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .corpus import Corpus, Passage
from .llm import LLM
from .verify import Check, Report, verify_claim

SYSTEM_PROMPT = (
    "你是一个严谨的技术研究助手。你只能使用给定的资料回答问题，"
    "不允许引入资料之外的事实。每个结论都必须标注来源编号。"
)

DECOMPOSE_PROMPT = """把下面的研究问题拆成 {n} 个可以独立检索的子问题。
要求：子问题要具体、覆盖不同角度（例如机制、数据、取舍）。
只输出 JSON 数组，不要其他文字。

问题：{question}

输出格式：["子问题1", "子问题2", "子问题3"]"""

CLAIM_PROMPT = """基于下面的资料，针对研究问题提取 {n} 条关键结论。
硬性要求：
1. 每条结论必须标注它来自哪一段资料（用 ref 字段，如 "cost-structure#0"）。
2. 只允许使用资料中出现的事实和数字，不得推测。
3. 结论要具体，避免"很重要""需要关注"这类空话。

研究问题：{question}

资料：
{material}

只输出 JSON 数组：
[{{"text": "结论", "ref": "doc-id#chunk"}}]"""

SYNTH_PROMPT = """把下面的已核查结论组织成一份简短报告（300 字以内）。
要求：每条结论后用方括号标注来源，例如 [cost-structure#0]。
不要添加结论之外的内容。

问题：{question}

结论：
{findings}"""


@dataclass
class Finding:
    claim: str
    ref: str
    check: Check

    @property
    def marker(self) -> str:
        return {
            "verified": "OK",
            "weak": "WEAK",
            "unverified": "UNVERIFIED",
            "contradicted": "CONTRADICTED",
            "missing_source": "NO-SOURCE",
        }.get(self.check.verdict, "?")


@dataclass
class ResearchResult:
    question: str
    sub_questions: list[str]
    evidence: list[Passage]
    findings: list[Finding]
    report: str
    verification: Report
    llm_used: bool
    tokens_estimated: int = 0

    @property
    def citations(self) -> list[str]:
        return sorted({f.ref for f in self.findings})

    def as_dict(self) -> dict:
        return {
            "question": self.question,
            "sub_questions": self.sub_questions,
            "llm_used": self.llm_used,
            "evidence": [
                {"ref": p.ref, "title": p.title, "score": round(p.score, 3),
                 "preview": p.text[:140]}
                for p in self.evidence
            ],
            "findings": [
                {**f.check.to_dict(), "claim": f.claim, "marker": f.marker}
                for f in self.findings
            ],
            "verification": {
                "total": self.verification.total,
                "verified": self.verification.verified,
                "contradicted": len(self.verification.contradicted),
                "rate": round(self.verification.rate, 3),
                "by_verdict": self.verification.by_verdict(),
            },
            "report": self.report,
        }


_MARKUP_PREFIX = re.compile(r"^[\s>*#\-–—•]+")


def _clean_sentence(text: str) -> str:
    """Strip markdown furniture so an extracted claim reads like a sentence.

    Chunking splits on Chinese full stops, which happily cuts through `**bold**`
    and leaves a stray `**` at the head of the next line.
    """
    return _MARKUP_PREFIX.sub("", text).replace("**", "").strip()


def _estimate_tokens(text: str) -> int:
    ascii_chars = sum(1 for ch in text if ord(ch) < 128)
    return int(ascii_chars / 4 + (len(text) - ascii_chars) / 1.6) + 1


class Scout:
    def __init__(
        self,
        corpus: Corpus,
        llm: LLM | None = None,
        *,
        max_sub_questions: int = 3,
        passages_per_query: int = 4,
        max_claims: int = 6,
    ) -> None:
        self.corpus = corpus
        self.llm = llm if llm is not None else LLM.from_env()
        self.max_sub_questions = max_sub_questions
        self.passages_per_query = passages_per_query
        self.max_claims = max_claims

    @classmethod
    def offline(cls, corpus: Corpus, **kwargs) -> "Scout":
        """Build a Scout that cannot reach the network.

        Passing llm=None means "discover credentials", which is right for the CLI
        and wrong for tests: on a machine that happens to have an API key, every
        test would silently start making real, slow, billable calls.
        """
        return cls(corpus, LLM(), **kwargs)

    # ------------------------------------------------------------ decompose

    def decompose(self, question: str) -> list[str]:
        if self.llm.available:
            raw = self.llm.complete_json(
                DECOMPOSE_PROMPT.format(n=self.max_sub_questions, question=question),
                system=SYSTEM_PROMPT, max_tokens=400,
            )
            if isinstance(raw, list):
                cleaned = [str(x).strip() for x in raw if str(x).strip()]
                if cleaned:
                    return cleaned[: self.max_sub_questions]
        return self._decompose_rules(question)

    def _decompose_rules(self, question: str) -> list[str]:
        """Deterministic fallback.

        Splitting on clause punctuation recovers most multi-part questions; when
        the question is atomic we generate the three angles that most technical
        questions decompose into anyway.
        """
        parts = [p.strip(" ？?。.") for p in re.split(r"[，,；;]|(?:以及|并且|还有)", question)]
        parts = [p for p in parts if len(p) >= 4]
        if len(parts) >= 2:
            return parts[: self.max_sub_questions]

        core = question.strip(" ？?。.")
        return [
            f"{core} 的具体机制或数据",
            f"{core} 的代价与限制",
            f"{core} 的实践建议",
        ][: self.max_sub_questions]

    # -------------------------------------------------------------- retrieve

    def gather(self, sub_questions: list[str]) -> list[Passage]:
        seen: dict[str, Passage] = {}
        for sub in sub_questions:
            for passage in self.corpus.search(sub, k=self.passages_per_query):
                if passage.ref not in seen or passage.score > seen[passage.ref].score:
                    seen[passage.ref] = passage
        return sorted(seen.values(), key=lambda p: p.score, reverse=True)[: self.passages_per_query * 2]

    # ---------------------------------------------------------------- claims

    def extract_claims(self, question: str, passages: list[Passage]) -> tuple[list[dict], bool]:
        """Returns (claims, used_llm)."""
        if self.llm.available and passages:
            material = "\n\n".join(
                f"[{p.ref}] {p.title}\n{p.text}" for p in passages
            )
            raw = self.llm.complete_json(
                CLAIM_PROMPT.format(n=self.max_claims, question=question, material=material),
                system=SYSTEM_PROMPT, max_tokens=1200,
            )
            if isinstance(raw, list):
                claims = []
                for item in raw:
                    if isinstance(item, dict) and item.get("text"):
                        claims.append({
                            "text": str(item["text"]).strip(),
                            "ref": str(item.get("ref", "")).strip(),
                        })
                if claims:
                    return claims[: self.max_claims], True
        return self._claims_rules(passages), False

    @staticmethod
    def _claims_rules(passages: list[Passage]) -> list[dict]:
        """Take the highest-scoring sentence from each passage.

        These claims are extracted verbatim, so they should pass verification.
        That is the point: the offline mode proves the *pipeline* works, and the
        interesting failures show up once a model starts paraphrasing.
        """
        claims: list[dict] = []
        for passage in passages[:3]:
            sentences = [
                _clean_sentence(s)
                for s in re.split(r"[。；\n]", passage.text)
            ]
            sentences = [s for s in sentences if len(s) > 20]
            if not sentences:
                continue
            # Prefer a sentence carrying a number: those are the checkable ones.
            with_numbers = [s for s in sentences if re.search(r"\d", s)]
            chosen = (with_numbers or sentences)[0]
            claims.append({"text": chosen, "ref": passage.ref})
        return claims

    # ------------------------------------------------------------ synthesize

    def synthesize(self, question: str, findings: list[Finding]) -> str:
        if self.llm.available and findings:
            body = "\n".join(f"- {f.claim} [{f.ref}]" for f in findings)
            text = self.llm.complete(
                SYNTH_PROMPT.format(question=question, findings=body),
                system=SYSTEM_PROMPT, max_tokens=700,
            )
            if text:
                return text
        return self._report_template(question, findings)

    @staticmethod
    def _report_template(question: str, findings: list[Finding]) -> str:
        lines = [f"# {question}", "", "## 核查过的结论", ""]
        for finding in findings:
            flag = "" if finding.check.ok else f"  （核查: {finding.check.verdict}）"
            lines.append(f"- {finding.claim} [{finding.ref}]{flag}")
        if not findings:
            lines.append("- 没有检索到可用证据，无法给出结论。")
        return "\n".join(lines)

    # --------------------------------------------------------------- research

    def research(self, question: str, *, inject_errors: bool = False) -> ResearchResult:
        sub_questions = self.decompose(question)
        evidence = self.gather(sub_questions)
        claims, used_llm = self.extract_claims(question, evidence)

        if inject_errors:
            claims = self._corrupt(claims)

        findings: list[Finding] = []
        for claim in claims:
            check = verify_claim(claim["text"], claim["ref"], self.corpus)
            findings.append(Finding(claim=claim["text"], ref=claim["ref"], check=check))

        report = self.synthesize(question, findings)
        verification = Report([f.check for f in findings])

        corpus_text = " ".join(p.text for p in evidence)
        tokens = _estimate_tokens(question) + _estimate_tokens(corpus_text) + _estimate_tokens(report)

        return ResearchResult(
            question=question,
            sub_questions=sub_questions,
            evidence=evidence,
            findings=findings,
            report=report,
            verification=verification,
            llm_used=used_llm,
            tokens_estimated=tokens,
        )

    @staticmethod
    def _corrupt(claims: list[dict]) -> list[dict]:
        """Deliberately damage claims, to prove the verifier earns its keep.

        Used by the demo and by tests: a verifier that never fails is
        indistinguishable from one that never runs.
        """
        corrupted = []
        for index, claim in enumerate(claims):
            text = claim["text"]
            if index % 2 == 0 and re.search(r"\d", text):
                text = re.sub(r"\d+(?:\.\d+)?", lambda m: str(int(float(m.group()) * 3) + 7), text, count=1)
            elif index % 2 == 1:
                text = text + " 这一结论在所有场景下都成立。"
            corrupted.append({"text": text, "ref": claim["ref"]})
        return corrupted
