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

When no rule fires, EmbeddingRouter gives a second opinion: it compares the
question embedding (already computed for retrieval) with example questions
per category, which catches paraphrases ("what deadlines does the contract
set?") and other languages. It only switches to deep when the question is
clearly closer to a deep category than to the targeted examples.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

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


# ── Embedding second opinion ──────────────────────────────────────────────────

# Example questions per category. "targeted" is the standard-mode anchor: a
# question must be clearly closer to a deep category than to these.
# Keep the examples keyword-free where possible: the regex rules already cover
# the obvious phrasings, these exist to catch everything else.
EXEMPLARS: Dict[str, List[str]] = {
    "summary": [
        "Di cosa parla questo documento?",
        "Qual è il succo del report?",
        "Quali sono i punti principali del testo?",
        "Spiegami in breve il contenuto del contratto",
        "Cosa dice in generale il documento?",
        "Quali sono le conclusioni dello studio?",
        "What is this document about?",
        "What are the key takeaways?",
        "What are the main points of the paper?",
        "Give me the gist of this report",
        "What does the document say overall?",
        "What are the conclusions of the study?",
    ],
    "list": [
        "Quali scadenze prevede il contratto?",
        "Quali obblighi ha il fornitore?",
        "Quali rischi vengono citati?",
        "Che date sono indicate nel documento?",
        "Quali penali sono previste?",
        "Quali persone vengono nominate?",
        "What deadlines does the contract set?",
        "Which obligations does the supplier have?",
        "What risks are mentioned?",
        "Which dates appear in the document?",
        "What penalties are foreseen?",
        "Who are the people mentioned in the text?",
    ],
    "comparison": [
        "In cosa differiscono i due accordi?",
        "Cosa cambia tra la versione vecchia e quella nuova?",
        "Il documento A e il documento B dicono la stessa cosa?",
        "Quale delle due offerte è più conveniente?",
        "How do the two agreements differ?",
        "What changed between the old and the new version?",
        "Do document A and document B say the same thing?",
        "Which of the two offers is better?",
    ],
    "count": [
        "Con quale frequenza viene citato il fornitore?",
        "In quanti punti si parla di riservatezza?",
        "How frequently is the supplier mentioned?",
        "In how many places is confidentiality discussed?",
    ],
    "targeted": [
        "Cosa dice l'articolo 5?",
        "Chi è il fornitore?",
        "Qual è la data di consegna?",
        "Quanto costa il servizio?",
        "Qual è il periodo di preavviso?",
        "Dove ha sede la società?",
        "Il contratto si rinnova automaticamente?",
        "Quanto dura la garanzia?",
        "Chi firma per conto del cliente?",
        "Qual è la legge applicabile?",
        "Cosa succede se il pagamento è in ritardo?",
        "Qual è l'importo della penale per ritardo?",
        "What does article 5 say?",
        "Who is the supplier?",
        "When is the delivery date?",
        "How much does the service cost?",
        "What is the notice period?",
        "Where is the company based?",
        "Does the contract renew automatically?",
        "How long is the warranty?",
        "Who signs on behalf of the client?",
        "Which law governs the agreement?",
        "What happens if a payment is late?",
        "What is the penalty amount for late delivery?",
    ],
}

STANDARD_CATEGORY = "targeted"


class EmbeddingRouter:
    """
    Nearest-examples classifier over question embeddings.

    Score per category = mean cosine similarity of the top_k closest examples.
    Returns a deep Route only when the best deep category scores at least
    min_similarity AND beats the targeted examples by at least margin.
    Example embeddings are computed lazily on first use and cached.
    """

    def __init__(
        self,
        embed: Callable[[List[str]], List[List[float]]],
        min_similarity: float,
        margin: float,
        top_k: int = 3,
        exemplars: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        self._embed = embed
        self.min_similarity = min_similarity
        self.margin = margin
        self._top_k = top_k
        self._exemplars = exemplars or EXEMPLARS
        self._matrix: Optional[np.ndarray] = None
        self._labels: List[str] = []
        self._lock = threading.Lock()

    def _ensure_embedded(self) -> None:
        with self._lock:
            if self._matrix is not None:
                return
            labels, texts = [], []
            for category, questions in self._exemplars.items():
                labels.extend([category] * len(questions))
                texts.extend(questions)
            self._matrix = _normalise(np.asarray(self._embed(texts), dtype=np.float32))
            self._labels = labels

    def scores(self, question_vector: Sequence[float]) -> Dict[str, float]:
        """Category → mean cosine similarity of its top_k closest examples."""
        self._ensure_embedded()
        query = _normalise(np.asarray(question_vector, dtype=np.float32)[None, :])[0]
        sims = self._matrix @ query
        result: Dict[str, float] = {}
        for category in self._exemplars:
            mask = np.fromiter((l == category for l in self._labels), dtype=bool)
            top = np.sort(sims[mask])[-self._top_k:]
            result[category] = float(top.mean())
        return result

    def classify(self, question_vector: Sequence[float]) -> Optional[Route]:
        """Return Route("deep", "similar to <category>") or None if not confident."""
        scores = self.scores(question_vector)
        standard = scores.pop(STANDARD_CATEGORY)
        category, best = max(scores.items(), key=lambda item: item[1])
        if best >= self.min_similarity and best - standard >= self.margin:
            return Route("deep", f"similar to {category}")
        return None


def _normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1, norms)
