"""Does trimming the claim-extraction context keep quality?

Lens showed that claim extraction takes ~71% of a research run's cost, because it
feeds every retrieved passage into one prompt. This measures whether sending less
costs anything in return.

Measured per configuration, on the same question, with Lens recording the tokens:

* tokens and cost — from the trace, not from estimates
* citation pass rate — whether verification still accepts the claims
* claim count — whether fewer passages simply produce fewer findings

Run:
    python examples/bench_cost.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lens import Lens  # noqa: E402
from scout.corpus import Corpus  # noqa: E402
from scout.llm import LLM  # noqa: E402
from scout.research import Scout  # noqa: E402

QUESTION = "缓存对延迟和成本分别有什么影响？"

CONFIGS = [
    ("baseline (8 段全传)", {"claim_context": 8, "claim_passage_chars": 0}),
    ("top-4", {"claim_context": 4, "claim_passage_chars": 0}),
    ("top-4 + 600 字", {"claim_context": 4, "claim_passage_chars": 600}),
]


def main() -> int:
    corpus = Corpus.load(Path(__file__).resolve().parent.parent / "data" / "corpus")
    lens = Lens(project="scout-bench")

    if not LLM.from_env().available:
        print("需要可用的模型 key 才能测成本；先用 --offline 之外的方式配置 .env")
        return 1

    rows = []
    for name, kwargs in CONFIGS:
        llm = LLM.from_env()
        llm.tracer = lens
        scout = Scout(corpus, llm, **kwargs)

        with lens.trace("cost-bench", input=QUESTION, metadata={"config": name}) as run:
            result = scout.research(QUESTION)
            run.output = result.report

        # Cost figures come from the trace, not from the caller's own arithmetic.
        extract_tokens = sum(
            s.prompt_tokens + s.completion_tokens
            for s in run.spans
            if s.kind == "llm"
        )
        rows.append({
            "name": name,
            "tokens": run.total_tokens,
            "cost": run.total_cost_usd,
            "ms": run.duration_ms,
            "claims": len(result.findings),
            "pass_rate": result.verification.rate,
            "spans": len(run.spans),
        })
        print(f"  完成 {name}: {run.total_tokens} tok, ${run.total_cost_usd:.6f}, "
              f"{len(result.findings)} 条结论, 核查 {result.verification.rate:.0%}")
        time.sleep(1)

    print()
    print(f"{'配置':20} {'tokens':>8} {'成本':>11} {'耗时':>8} {'结论数':>6} {'核查通过':>8}")
    print("-" * 68)
    for row in rows:
        print(f"{row['name']:20} {row['tokens']:>8} ${row['cost']:>10.6f} "
              f"{row['ms'] / 1000:>7.1f}s {row['claims']:>6} {row['pass_rate']:>7.0%}")

    baseline = rows[0]
    print()
    for row in rows[1:]:
        saved = 1 - row["cost"] / baseline["cost"] if baseline["cost"] else 0
        delta_rate = row["pass_rate"] - baseline["pass_rate"]
        print(f"{row['name']}: 成本省 {saved:.0%}，核查通过率变化 {delta_rate:+.0%}，"
              f"结论数 {baseline['claims']} → {row['claims']}")

    print()
    print("判断标准：省成本的同时，核查通过率和结论数不应明显下降。")
    print("只降成本不看质量，等于把省下的钱从结论的可信度里扣。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
