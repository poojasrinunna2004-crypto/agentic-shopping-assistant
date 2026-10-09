import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


FAQ_FILE = Path(__file__).resolve().parent.parent / "data" / "product_faq.csv"
load_dotenv()


def retrieve_faqs(question: str, product_id: str | None = None, limit: int = 3) -> list[dict[str, Any]]:
    entries = pd.read_csv(FAQ_FILE, dtype=str).to_dict(orient="records")
    if product_id:
        entries = [entry for entry in entries if entry["product_id"] in (product_id, "all")]
    if not entries:
        return []

    corpus = [f"{entry['question']} {entry['answer']}" for entry in entries]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(corpus)
    query_vector = vectorizer.transform([question])
    scores = cosine_similarity(query_vector, matrix).flatten()
    ranked = sorted(zip(scores, entries), key=lambda item: float(item[0]), reverse=True)
    return [
        {**entry, "relevance": round(float(score), 3)}
        for score, entry in ranked[:limit]
        if score > 0
    ]


def answer_faq(question: str, product_id: str | None = None) -> dict[str, Any]:
    sources = retrieve_faqs(question, product_id)
    if not sources:
        return {
            "answer": "I could not find relevant information in the product FAQ. Please check the product listing or ask a more specific question.",
            "sources": [],
            "llm_used": False,
        }

    context = "\n".join(
        f"[{source['faq_id']}] {source['question']} {source['answer']}"
        for source in sources
    )
    if os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_openai import ChatOpenAI

            model = ChatOpenAI(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                temperature=0,
            )
            response = model.invoke(
                "Answer the customer's question using only the FAQ context below. "
                "If it does not contain the answer, say so. Keep the answer concise.\n\n"
                f"FAQ context:\n{context}\n\nCustomer question: {question}"
            )
            answer = str(response.content)
            llm_used = True
        except Exception:
            answer = " ".join(source["answer"] for source in sources[:2])
            llm_used = False
    else:
        answer = " ".join(source["answer"] for source in sources[:2])
        llm_used = False

    return {
        "answer": answer,
        "sources": [
            {"faq_id": source["faq_id"], "question": source["question"], "relevance": source["relevance"]}
            for source in sources
        ],
        "llm_used": llm_used,
    }
