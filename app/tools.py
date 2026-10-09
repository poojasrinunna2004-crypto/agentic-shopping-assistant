from typing import Any

from langchain_core.tools import tool

from app.catalog import get_product, search_catalog


@tool
def search_products(
    query: str,
    max_price: int | None = None,
    preferred_ram_gb: int | None = None,
) -> list[dict[str, Any]]:
    """Search the sample product catalog and rank matches for a shopping request."""
    return search_catalog(query, max_price, preferred_ram_gb)


@tool
def compare_products(product_ids: list[str]) -> list[dict[str, Any]]:
    """Return comparable price, memory, processor, storage, battery, and warranty data."""
    fields = (
        "product_id", "name", "brand", "price", "effective_price", "currency", "ram_gb",
        "processor", "storage", "battery_hours", "warranty_years", "search_score",
    )
    products = [get_product(product_id) for product_id in product_ids]
    return [{field: product[field] for field in fields if field in product} for product in products if product]


@tool
def check_inventory(product_ids: list[str]) -> list[dict[str, Any]]:
    """Check sample stock counts and delivery estimates for product IDs."""
    products = [get_product(product_id) for product_id in product_ids]
    return [
        {
            "product_id": product["product_id"],
            "in_stock": product["stock"] > 0,
            "stock": product["stock"],
            "delivery_days": product["delivery_days"],
        }
        for product in products if product
    ]


@tool
def validate_budget(product_ids: list[str], budget: int | None = None) -> list[dict[str, Any]]:
    """Check each discounted sample product price against a customer's budget."""
    products = [get_product(product_id) for product_id in product_ids]
    return [
        {
            "product_id": product["product_id"],
            "effective_price": product["effective_price"],
            "within_budget": budget is None or product["effective_price"] <= budget,
            "budget": budget,
        }
        for product in products if product
    ]
