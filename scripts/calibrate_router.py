"""
Calibrate the auto-mode embedding router for the configured embedding model.

Scores a labelled set of questions that are NOT among the router examples and
that the regex rules leave on standard (the only case the embedding router
decides), then suggests ROUTER_MARGIN: the smallest margin with no standard
question routed to deep, plus a safety gap.

Usage (Ollama running, same .env as the app):
    python scripts/calibrate_router.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ollama  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from src.services.query_router import (  # noqa: E402
    EXEMPLARS, STANDARD_CATEGORY, EmbeddingRouter, route_question,
)

SAFETY_GAP = 0.01

DEEP = [
    "Che cosa tratta il report?", "Quali sono i temi affrontati?", "Che scadenze ci sono?",
    "Quali garanzie offre il fornitore?", "Che clausole di recesso esistono?",
    "Quali sono le responsabilità delle parti?", "In che modo i due contratti si distinguono?",
    "Quali importi compaiono nel documento?", "What topics does the report cover?",
    "What deadlines are there?", "Which guarantees does the vendor provide?",
    "What termination clauses exist?", "What are the responsibilities of each party?",
    "How do the two contracts diverge?", "What amounts appear in the document?",
    "What is the paper about?", "Qu'est-ce que ce document raconte?",
    "Quelles sont les échéances du contrat?",
]
STANDARD = [
    "Qual è il foro competente?", "Chi è il responsabile del trattamento dati?",
    "Quando scade il contratto?", "Quanto vale la cauzione?", "Il fornitore può subappaltare?",
    "Qual è l'IBAN per i pagamenti?", "Cosa prevede l'articolo 12?",
    "Entro quanti giorni va pagata la fattura?", "Who is the data controller?",
    "When does the contract expire?", "How much is the deposit?", "Can the supplier subcontract?",
    "What does clause 12 state?", "Within how many days must the invoice be paid?",
    "Is there a non-compete clause?", "What is the governing jurisdiction?",
    "Qui est le fournisseur?", "Quand le contrat expire-t-il?",
]


def main() -> None:
    load_dotenv()
    model = os.getenv("EMBEDDING_MODEL", "mxbai-embed-large:latest")
    host = f"http://{os.getenv('OLLAMA_HOST', 'localhost')}:{os.getenv('OLLAMA_PORT', '11434')}"
    client = ollama.Client(host=host)

    def embed(texts):
        return client.embed(model=model, input=texts).embeddings

    known = {q for questions in EXEMPLARS.values() for q in questions}
    labelled = [(q, "deep") for q in DEEP] + [(q, "standard") for q in STANDARD]
    labelled = [(q, label) for q, label in labelled
                if q not in known and route_question(q).mode == "standard"]

    router = EmbeddingRouter(embed, min_similarity=0.0, margin=0.0)
    rows = []
    for (question, label), vector in zip(labelled, embed([q for q, _ in labelled])):
        scores = router.scores(vector)
        standard = scores.pop(STANDARD_CATEGORY)
        category, best = max(scores.items(), key=lambda item: item[1])
        rows.append((best - standard, best, label, category, question))

    print(f"Embedding model: {model}\n")
    for diff, best, label, category, question in sorted(rows):
        print(f"{label:8} diff={diff:+.3f} best={best:.3f} {category:10} | {question}")

    worst_standard = max(r[0] for r in rows if r[2] == "standard")
    margin = round(worst_standard + SAFETY_GAP, 2)
    caught = sum(1 for r in rows if r[2] == "deep" and r[0] >= margin)
    total_deep = sum(1 for r in rows if r[2] == "deep")
    min_sim = min((r[1] for r in rows if r[2] == "deep" and r[0] >= margin), default=0.5)
    print(f"\nSuggested ROUTER_MARGIN={margin} "
          f"(deep caught {caught}/{total_deep}, standard routed to deep 0)")
    print(f"Lowest best-similarity among caught deep questions: {min_sim:.3f} "
          "(keep ROUTER_MIN_SIMILARITY below it)")


if __name__ == "__main__":
    main()
