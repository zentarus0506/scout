"""Retrieval and citation verification.

These tests are the project's actual claim to value: each one pins down a failure
mode that a research agent has to catch, and each would have passed silently if
verification were delegated to the same model that wrote the sentence.
"""

from __future__ import annotations

import re

import pytest

from scout import Corpus, Document, Scout, verify_claim
from scout.cli import DEFAULT_CORPUS


@pytest.fixture(scope="module")
def corpus() -> Corpus:
    return Corpus.load(DEFAULT_CORPUS)


def test_corpus_loads_every_document(corpus):
    assert len(corpus.documents) >= 8
    assert all(doc.text for doc in corpus.documents)


def test_search_finds_the_right_document(corpus):
    hits = corpus.search("前缀缓存 首 token 延迟", k=3)
    assert hits, "no hits for a query built from corpus vocabulary"
    assert hits[0].doc_id == "caching-strategies"


def test_search_is_deterministic(corpus):
    first = [p.ref for p in corpus.search("模型路由 成本", k=5)]
    second = [p.ref for p in corpus.search("模型路由 成本", k=5)]
    assert first == second


def test_get_resolves_a_reference(corpus):
    passage = corpus.search("批处理 吞吐", k=1)[0]
    assert corpus.get(passage.ref) is not None
    assert corpus.get("no-such-doc#0") is None
    assert corpus.get("batching-tradeoffs#999") is None


def test_verbatim_claim_is_verified(corpus):
    passage = corpus.search("前缀缓存 输入成本", k=1)[0]
    sentence = re.split(r"[。；\n]", passage.text)[0][:80]
    check = verify_claim(sentence, passage.ref, corpus)
    assert check.verdict == "verified", check.notes


def test_fabricated_number_is_caught(corpus):
    """The most common hallucination: a real sentence with an invented figure."""
    passage = corpus.search("前缀缓存 输入成本", k=1)[0]
    claim = "前缀缓存可以把这部分输入的计费降到原价的 3%，节省 97% 的成本"
    check = verify_claim(claim, passage.ref, corpus)
    assert check.verdict == "contradicted"
    assert check.numbers_missing, "a fabricated figure must be reported"


def test_negation_flip_is_caught():
    """Same nouns, opposite meaning. Overlap alone cannot see this.

    Built on a minimal corpus rather than the shipped one: the rule under test is
    about negation, and a real passage's incidental wording should not be able to
    mask or manufacture the result.
    """
    corpus = Corpus(documents=[Document(id="demo", title="演示", text="连续批处理提升整体吞吐。")])
    corpus._build_index()

    check = verify_claim("连续批处理不会提升整体吞吐", "demo#0", corpus)
    assert check.verdict != "verified"
    assert any("否定" in note for note in check.notes)


def test_semantic_cache_claim_is_not_verified(corpus):
    """A reversed claim about a risky technique must not pass, whatever the reason."""
    passage = corpus.search("语义缓存 误命中", k=1)[0]
    claim = "我们在生产环境启用了语义缓存，因为它不会带来误命中"
    check = verify_claim(claim, passage.ref, corpus)
    assert check.verdict != "verified"


def test_overgeneralisation_is_caught(corpus):
    """A source says 'in our tests'; the summary says 'always'."""
    passage = corpus.search("升级预算 严重错误率", k=1)[0]
    base = re.split(r"[。；\n]", passage.text)[0][:60]
    check = verify_claim(base + "，并且该结论在所有场景下都一定成立", passage.ref, corpus)
    assert check.verdict != "verified"
    assert any("绝对化" in note for note in check.notes)


def test_missing_source_is_distinct_from_contradicted(corpus):
    """A broken pointer and a broken argument are different bugs."""
    check = verify_claim("某个结论", "ghost-doc#7", corpus)
    assert check.verdict == "missing_source"


def test_research_pipeline_runs_offline(corpus):
    scout = Scout.offline(corpus)
    assert not scout.llm.available
    result = scout.research("如何降低大模型推理成本？")
    assert result.sub_questions
    assert result.evidence
    assert result.findings
    assert "引用核查" in result.verification.summary()
    # Verbatim extraction should pass its own verification.
    assert result.verification.rate >= 0.5


def test_injected_errors_lower_the_pass_rate(corpus):
    """If corruption did not lower the rate, the verifier would be decorative."""
    scout = Scout.offline(corpus)
    clean = scout.research("如何降低大模型推理成本？")
    dirty = scout.research("如何降低大模型推理成本？", inject_errors=True)
    assert dirty.verification.rate < clean.verification.rate


def test_report_carries_citations(corpus):
    scout = Scout.offline(corpus)
    result = scout.research("批处理的延迟代价是什么？")
    assert "[" in result.report and "]" in result.report
    for finding in result.findings:
        assert finding.ref in result.report


def test_result_serialises(corpus):
    scout = Scout.offline(corpus)
    payload = scout.research("缓存有什么风险？").as_dict()
    for key in ("question", "evidence", "findings", "verification", "report"):
        assert key in payload
    assert payload["verification"]["total"] == len(payload["findings"])
