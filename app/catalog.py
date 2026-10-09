import re
from pathlib import Path
from typing import Any

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PRODUCTS_FILE = DATA_DIR / "products.csv"
STOP_WORDS = {
    "a", "an", "and", "are", "for", "good", "i", "in", "is", "it", "laptop",
    "buy", "looking", "my", "need", "of", "on", "or", "please", "prefer", "the", "to",
    "under", "want", "with", "years",
}


def load_products() -> list[dict[str, Any]]:
    products = pd.read_csv(PRODUCTS_FILE).to_dict(orient="records")

    for product in products:
        for field in (
            "price", "discount_percent", "ram_gb", "battery_hours", "stock",
            "delivery_days", "warranty_years",
        ):
            product[field] = int(product[field])
        product["effective_price"] = round(
            product["price"] * (100 - product["discount_percent"]) / 100
        )
    return products


def get_product(product_id: str) -> dict[str, Any] | None:
    return next(
        (product for product in load_products() if product["product_id"] == product_id),
        None,
    )


def search_catalog(
    query: str,
    max_price: int | None = None,
    preferred_ram_gb: int | None = None,
    limit: int = 12,
) -> list[dict[str, Any]]:
    query_terms = {
        term for term in re.findall(r"[a-z0-9]+", query.lower())
        if term not in STOP_WORDS and not term.isdigit()
    }
    ranked: list[tuple[float, dict[str, Any]]] = []
    for product in load_products():
        searchable = " ".join(
            str(product[field]) for field in ("name", "brand", "category", "processor", "tags", "description")
        ).lower()
        matching_terms = sum(term in searchable for term in query_terms)
        if query_terms and matching_terms == 0:
            continue
        score = float(matching_terms)
        if max_price is not None and product["effective_price"] <= max_price:
            score += 2
        if preferred_ram_gb is not None:
            if product["ram_gb"] >= preferred_ram_gb:
                score += 2
            else:
                score -= 1
        product_result = {**product, "search_score": round(score, 2)}
        ranked.append((score, product_result))

    ranked.sort(
        key=lambda entry: (
            entry[0],
            entry[1]["stock"] > 0,
            entry[1]["ram_gb"],
            -entry[1]["effective_price"],
        ),
        reverse=True,
    )
    return [product for _, product in ranked[:limit]]
