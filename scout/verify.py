"""Citation verification.

A research agent's most damaging failure is not "I don't know" — it is a
confident sentence with a plausible-looking citation that does not actually
support it. This module makes that failure detectable without asking a model to
grade itself.

Three signals do most of the work:

1. **Number fidelity.** If the claim says "37% faster" and the cited passage never
   mentions 37, the claim is fabricated or misattributed. Models invent numbers
   far more readily than they invent entities, so this check catches the worst
   cases cheaply.
2. **Term overlap.** Weighted coverage of the claim's content words by the passage,
   with stopwords removed so shared grammar does not create false support.
3. **Negation mismatch.** "does not improve" versus "improves" flips meaning while
   keeping every noun, which overlap alone cannot see.

A claim that cites a passage which does not exist is reported as `missing_source`,
separately from `contradicted`: one is a broken pointer, the other is a broken
argument.

A fourth outcome matters just as much: `unverified`. The checker compares terms,
so a correct paraphrase can drift below the threshold. Reporting that as
"unsupported" would punish the model for phrasing things its own way and would
train users to ignore the checker. Low confidence and active contradiction are
kept apart everywhere in this module.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .corpus import Corpus, Passage, tokenize

# Words that appear everywhere and therefore carry no support signal.
STOPWORDS = {
    "的", "是", "在", "和", "了", "有", "与", "及", "等", "把", "被", "对", "从", "为",
    "以", "而", "或", "但", "就", "都", "也", "很", "这", "那", "其", "之", "个", "上",
    "中", "下", "会", "能", "可", "将", "让", "使", "更", "最", "并", "且", "因", "所",
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "of", "to", "in", "on",
    "for", "and", "or", "but", "with", "as", "at", "by", "it", "its", "this", "that",
    "these", "those", "we", "you", "they", "can", "will", "may", "should", "would",
}

_NUMBER = re.compile(r"\d+(?:\.\d+)?\s*%?")
_NEGATION = re.compile(r"(不|没|无法|不能|并非|not\b|no\b|never\b|cannot\b|doesn't\b|don't\b)")
# Absolute quantifiers. Single-character words like 都 are excluded on purpose:
# they are too common in ordinary prose to be evidence of overreach.
_ABSOLUTE = re.compile(r"(所有|任何|一切|必然|一定|绝对|从不|永远|总是|毫无例外|无一例外|\b(always|never|all|every|none)\b)")

VERIFIED = "verified"
WEAK = "weak"
UNVERIFIED = "unverified"
CONTRADICTED = "contradicted"
MISSING = "missing_source"

VERDICT_LABELS = {
    VERIFIED: "已证实",
    WEAK: "证据较弱",
    UNVERIFIED: "未找到足够证据",
    CONTRADICTED: "与引文矛盾",
    MISSING: "引文不存在",
}


@dataclass
class Check:
    claim: str
    ref: str
    verdict: str
    overlap: float
    numbers_in_claim: list[str] = field(default_factory=list)
    numbers_missing: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.verdict == VERIFIED

    @property
    def contradicted(self) -> bool:
        return self.verdict in (CONTRADICTED, MISSING)

    def to_dict(self) -> dict:
        return {
            "claim": self.claim,
            "ref": self.ref,
            "verdict": self.verdict,
            "label": VERDICT_LABELS.get(self.verdict, self.verdict),
            "overlap": round(self.overlap, 3),
            "numbers_missing": self.numbers_missing,
            "notes": self.notes,
        }


def content_terms(text: str) -> set[str]:
    return {t for t in tokenize(text) if t not in STOPWORDS and len(t) > 1}


def _numbers(text: str) -> list[str]:
    return [m.group().replace(" ", "") for m in _NUMBER.finditer(text)]


def verify_claim(claim: str, ref: str, corpus: Corpus) -> Check:
    passage: Passage | None = corpus.get(ref)
    if passage is None:
        return Check(claim, ref, MISSING, 0.0, _numbers(claim), [], ["引用的段落不存在"])

    claim_terms = content_terms(claim)
    if not claim_terms:
        return Check(claim, ref, UNVERIFIED, 0.0, [], [], ["结论没有可核查的实词"])

    passage_terms = content_terms(passage.text)
    overlap = len(claim_terms & passage_terms) / len(claim_terms)

    claim_numbers = _numbers(claim)
    passage_numbers = set(_numbers(passage.text))
    missing = [n for n in claim_numbers if n not in passage_numbers]

    notes: list[str] = []
    if missing:
        notes.append(f"引文中未出现这些数字: {', '.join(missing)}")

    # One-directional on purpose: a claim is usually a *slice* of the passage, so
    # the passage containing a negation the claim lacks is normal. The suspicious
    # case is the reverse — the claim asserts a negation its source never makes.
    negation_flip = bool(_NEGATION.search(claim)) and not _NEGATION.search(passage.text)
    if negation_flip:
        notes.append("结论使用了否定表述，但引文中没有对应的否定")

    # Hedging collapse is the third common failure: a source says "in our tests"
    # and the summary says "always". The nouns and numbers still match, so only a
    # quantifier check can see it.
    absolute_overreach = bool(_ABSOLUTE.search(claim)) and not _ABSOLUTE.search(passage.text)
    if absolute_overreach:
        notes.append("结论使用了绝对化表述，引文中没有同等强度的说法")

    # Distinguish "the source contradicts this" from "the wording drifted too far
    # for me to judge". Collapsing both into one failure mode punishes correct
    # paraphrases, which is exactly how a checker loses its users' trust.
    if missing or negation_flip or absolute_overreach:
        verdict = CONTRADICTED
    elif overlap >= 0.55:
        verdict = VERIFIED
    elif overlap >= 0.3:
        verdict = WEAK
    else:
        verdict = UNVERIFIED

    return Check(claim, ref, verdict, overlap, claim_numbers, missing, notes)


@dataclass
class Report:
    checks: list[Check]

    @property
    def total(self) -> int:
        return len(self.checks)

    @property
    def verified(self) -> int:
        return sum(1 for c in self.checks if c.verdict == VERIFIED)

    @property
    def rate(self) -> float:
        return self.verified / self.total if self.total else 0.0

    @property
    def contradicted(self) -> list[Check]:
        """Claims the evidence actively argues against, as opposed to merely weak."""
        return [c for c in self.checks if c.verdict in (CONTRADICTED, MISSING)]

    def by_verdict(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for check in self.checks:
            counts[check.verdict] = counts.get(check.verdict, 0) + 1
        return counts

    def summary(self) -> str:
        counts = self.by_verdict()
        order = [VERIFIED, WEAK, UNVERIFIED, CONTRADICTED, MISSING]
        parts = [f"{VERDICT_LABELS[k]} {counts[k]}" for k in order if k in counts]
        return f"引用核查 {self.verified}/{self.total} 通过（{self.rate:.0%}）| " + "，".join(parts)


def verify_all(claims: list[dict], corpus: Corpus) -> Report:
    """`claims` is a list of {"text": ..., "ref": ...} produced by the agent."""
    return Report([verify_claim(c["text"], c.get("ref", ""), corpus) for c in claims])
