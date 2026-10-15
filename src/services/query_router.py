"""
Rule-based query router: decides whether a question needs deep mode.

Standard retrieval (top-k chunks) answers targeted questions well but cannot
answer questions that need the whole document. Those are recognised by their
wording, in Italian and English:

    summary      "riassumi il report", "summarize the paper"
    exhaustive   "elenca tutte le scadenze", "list every clause"
    comparison   "confronta i due contratti", "differences between A and B"
    count        "quante volte compare X", "how many times"

Rules are deterministic and instant (no extra LLM call), so the UI can show
the reason and the user can override it by picking a mode explicitly.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple

MODES = ("auto", "standard", "deep")


@dataclass(frozen=True)
class Route:
    mode: str    # "standard" | "deep"
    reason: str  # short label shown in the UI


# (reason, pattern) — first match wins. Patterns run on the lowercased question.
_RULES: List[Tuple[str, re.Pattern]] = [
    ("summary", re.compile(
        r"\b(riassum\w*|riassunt\w*|sintes[ie]|sintetizz\w*|panoramica|"
        r"summar\w*|overview|tl;?dr|recap)\b"
    )),
    ("count", re.compile(
        r"\b(quante volte|quanti sono|quante sono|how many times|how often|"
        r"count|conta(?:re)?|numero (?:totale|di volte))\b"
    )),
    # "compare" alone is also Italian for "appears" ("dove compare X"), so the
    # English verb only counts when followed by what is being compared.
    ("comparison", re.compile(
        r"\b(confront\w*|comparazion\w*|differenz\w*|differenc\w*|differ|"
        r"comparisons?|comparing|compared|versus|vs\.?|similarit\w*|analogi\w*)\b"
        r"|\bcompare\s+(?:the|these|those|them|it|(?:\w+\s+){1,3}(?:and|with|to)\b)"
    )),
    ("exhaustive list", re.compile(
        r"\b(elenc\w*|lista|list(?:\s+(?:all|every|each|the))?|enumera\w*)\b"
        r"|\b(tutt[ie]|ogni|all|every|each)\s+(?:i\s+|le\s+|gli\s+|the\s+)?"
        r"(?:\w+\s+)?(scadenz\w*|clausol\w*|articol\w*|punt[io]|date|"
        r"nom[ei]|import[io]|cifr\w*|riferiment\w*|obblig\w*|rischi\w*|"
        r"deadlines?|clauses?|articles?|points?|dates?|names?|amounts?|"
        r"figures?|references?|obligations?|risks?|items?|mentions?)\b"
    )),
    ("whole document", re.compile(
        r"\b(intero documento|documento intero|tutto il documento|"
        r"in tutti i documenti|whole document|entire document|"
        r"across (?:all )?(?:the )?documents)\b"
    )),
]


def route_question(question: str) -> Route:
    """Return Route("deep", reason) for whole-document questions, else standard."""
    text = question.lower()
    for reason, pattern in _RULES:
        if pattern.search(text):
            return Route("deep", reason)
    return Route("standard", "targeted question")
