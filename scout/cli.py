"""Command line entry point.

Deliberately text-first: the value of this tool is the audit trail, and a table in
a terminal is easier to scrutinise than a rendered page.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .corpus import Corpus
from .llm import LLM
from .research import Scout
from .verify import VERDICT_LABELS

DEFAULT_CORPUS = Path(__file__).resolve().parent.parent / "data" / "corpus"


def _render(result, *, show_evidence: bool = True) -> str:
    out: list[str] = []
    out.append(f"问题：{result.question}")
    out.append(f"模式：{'LLM 增强' if result.llm_used else '离线规则（未配置 key 或调用失败）'}")
    out.append("")

    out.append("子问题：")
    for sub in result.sub_questions:
        out.append(f"  - {sub}")
    out.append("")

    if show_evidence:
        out.append(f"检索到 {len(result.evidence)} 段证据（BM25）：")
        for passage in result.evidence[:6]:
            out.append(f"  [{passage.ref}] {passage.title}  score={passage.score:.2f}")
        out.append("")

    out.append("结论与核查：")
    if not result.findings:
        out.append("  （没有可核查的结论）")
    for finding in result.findings:
        label = VERDICT_LABELS.get(finding.check.verdict, finding.check.verdict)
        mark = "PASS" if finding.check.ok else label
        out.append(f"  [{mark}] {finding.claim}")
        out.append(f"         来源 {finding.ref} · 词项覆盖 {finding.check.overlap:.0%}")
        for note in finding.check.notes:
            out.append(f"         注意：{note}")
    out.append("")

    out.append("--- 报告 ---")
    out.append(result.report)
    out.append("")
    out.append(result.verification.summary())
    return "\n".join(out)


def cmd_ask(args: argparse.Namespace) -> int:
    corpus = Corpus.load(args.corpus)
    llm = LLM() if args.offline else LLM.from_env(args.provider)
    scout = Scout(corpus, llm)

    result = scout.research(args.question, inject_errors=args.inject_errors)
    if args.json:
        print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
    else:
        print(_render(result, show_evidence=not args.quiet))
    return 0 if result.verification.rate >= args.min_rate else 1


def cmd_demo(args: argparse.Namespace) -> int:
    """Run twice: clean, then with corrupted claims.

    The pair is the argument. Anyone can show a passing checker; showing what it
    rejects is what makes the passing numbers mean something.
    """
    corpus = Corpus.load(args.corpus)
    scout = Scout(corpus, LLM())
    question = args.question

    print("=" * 70)
    print("第一次：正常流程（离线规则抽取，结论逐字来自原文）")
    print("=" * 70)
    clean = scout.research(question)
    print(_render(clean, show_evidence=False))

    print()
    print("=" * 70)
    print("第二次：故意篡改数字与过度概括，看核查器能否发现")
    print("=" * 70)
    dirty = scout.research(question, inject_errors=True)
    print(_render(dirty, show_evidence=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scout", description="带引用核查的深度研究 agent")
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS), help="语料目录")
    sub = parser.add_subparsers(dest="command")

    ask = sub.add_parser("ask", help="研究一个问题")
    ask.add_argument("question")
    ask.add_argument("--json", action="store_true", help="输出 JSON")
    ask.add_argument("--quiet", action="store_true", help="不显示检索片段")
    ask.add_argument("--offline", action="store_true", help="强制离线规则模式")
    ask.add_argument("--provider", default=None)
    ask.add_argument("--inject-errors", action="store_true", help="故意篡改结论以验证核查器")
    ask.add_argument("--min-rate", type=float, default=0.0, help="核查通过率低于此值时返回非零")
    ask.set_defaults(func=cmd_ask)

    demo = sub.add_parser("demo", help="对照演示：正常 vs 篡改")
    demo.add_argument("question", nargs="?",
                      default="如何在不明显损害质量的前提下降低大模型推理成本？")
    demo.set_defaults(func=cmd_demo)

    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
