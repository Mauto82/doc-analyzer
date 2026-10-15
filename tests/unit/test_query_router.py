"""Tests for the rule-based standard/deep query router."""
import pytest
from src.services.query_router import route_question


@pytest.mark.parametrize("question,reason", [
    # summary
    ("Riassumi il documento", "summary"),
    ("Dammi una sintesi del report", "summary"),
    ("Dammi una panoramica del progetto", "summary"),
    ("Summarize the report", "summary"),
    ("Give me an overview", "summary"),
    # count
    ("Quante volte compare la parola penale?", "count"),
    ("How many times is GDPR mentioned?", "count"),
    # comparison
    ("Confronta le clausole di recesso", "comparison"),
    ("Quali sono le differenze tra A e B?", "comparison"),
    ("What are the differences between A and B?", "comparison"),
    ("Compare the two contracts", "comparison"),
    ("Compare clause 4 with clause 9", "comparison"),
    # exhaustive list
    ("Elenca tutte le scadenze", "exhaustive list"),
    ("Quali sono tutte le scadenze del contratto?", "exhaustive list"),
    ("Elenca tutte le date di pagamento", "exhaustive list"),
    ("List every deadline", "exhaustive list"),
    ("What are all the obligations of the supplier?", "exhaustive list"),
    # whole document
    ("Quali rischi sono citati in tutto il documento?", "whole document"),
])
def test_routes_to_deep(question, reason):
    route = route_question(question)
    assert route.mode == "deep"
    assert route.reason == reason


@pytest.mark.parametrize("question", [
    "Cosa dice l'articolo 5?",
    "Chi è il fornitore?",
    "Qual è la data di consegna?",
    "Rimanda all'articolo 12?",       # "all'" is not English "all"
    "Quanto costa il servizio?",
    "Il contatore viene resettato?",  # "conta" inside a word
    "What is the account number?",    # "count" inside a word
    "Tutti i dati sono cifrati?",     # yes/no question, not a list
    "Dove compare la parola penale?", # Italian "compare" = appears
    "Compare a pagina 3 il nome?",
    "Who signed the contract?",
])
def test_routes_to_standard(question):
    route = route_question(question)
    assert route.mode == "standard"
    assert route.reason == "targeted question"


# ── EmbeddingRouter ────────────────────────────────────────────────────────────

from src.services.query_router import EmbeddingRouter, Route

# Orthogonal toy vectors per category so similarities are easy to reason about
_AXES = {"summary": [1, 0, 0], "list": [0, 1, 0], "targeted": [0, 0, 1]}
_EXEMPLARS = {cat: [f"{cat} {i}" for i in range(3)] for cat in _AXES}


def _router(**kwargs):
    calls = []

    def embed(texts):
        calls.append(list(texts))
        return [_AXES[t.split()[0]] for t in texts]

    router = EmbeddingRouter(embed, exemplars=_EXEMPLARS,
                             **{"min_similarity": 0.5, "margin": 0.06, **kwargs})
    return router, calls


def test_embedding_router_picks_closest_deep_category():
    router, _ = _router()
    assert router.classify([0.9, 0.1, 0.1]) == Route("deep", "similar to summary")


def test_embedding_router_none_when_closer_to_targeted():
    router, _ = _router()
    assert router.classify([0.1, 0.1, 0.9]) is None


def test_embedding_router_requires_margin():
    router, _ = _router(margin=0.2)
    # summary barely beats targeted: below the margin → no opinion
    assert router.classify([0.55, 0, 0.5]) is None


def test_embedding_router_requires_min_similarity():
    router, _ = _router(min_similarity=0.9)
    # split between summary and list: ~0.70 each, below 0.9
    assert router.classify([0.6, 0.6, 0.1]) is None


def test_embedding_router_embeds_examples_once():
    router, calls = _router()
    router.classify([1, 0, 0])
    router.classify([0, 1, 0])
    assert len(calls) == 1
    assert len(calls[0]) == 9


def test_embedding_router_scores_are_cosine():
    router, _ = _router()
    scores = router.scores([2, 0, 0])  # unnormalised input
    assert scores == pytest.approx({"summary": 1.0, "list": 0.0, "targeted": 0.0})
