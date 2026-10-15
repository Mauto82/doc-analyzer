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
