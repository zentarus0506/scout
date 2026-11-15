"""Scout — a research agent whose claims are checked against their sources."""

from .corpus import Corpus, Document, Passage, tokenize
from .llm import LLM
from .research import Finding, ResearchResult, Scout
from .verify import Check, Report, verify_claim

__all__ = [
    "Check",
    "Corpus",
    "Document",
    "Finding",
    "LLM",
    "Passage",
    "Report",
    "ResearchResult",
    "Scout",
    "tokenize",
    "verify_claim",
]
__version__ = "0.1.0"
