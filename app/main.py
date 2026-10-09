from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from app.catalog import get_product, load_products, search_catalog
from app.faq import answer_faq
from app.workflow import recommend


class RecommendationRequest(BaseModel):
    requirements: str = Field(min_length=3, description="Natural-language shopping requirements")
    budget: int | None = Field(default=None, gt=0, description="Optional budget override in catalog currency")
    preferred_ram_gb: int | None = Field(default=None, gt=0)


class FAQRequest(BaseModel):
    question: str = Field(min_length=3)
    product_id: str | None = None


class CartConfirmation(BaseModel):
    product_id: str
    quantity: int = Field(default=1, ge=1, le=10)
    confirmed: bool = False


cart_items: dict[str, dict[str, Any]] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_products()
    yield


app = FastAPI(
    title="Agentic Shopping Assistant",
    description=(
        "A training API that searches a sample catalog, compares products, checks stock and budget, "
        "retrieves product FAQs, and creates a simulated cart only after customer confirmation."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/products")
def list_products(
    q: str = Query(default="", description="Search by product name, use, or specification"),
    max_price: int | None = Query(default=None, gt=0),
) -> dict[str, Any]:
    products = search_catalog(q) if q else load_products()
    if max_price is not None:
        products = [product for product in products if product["effective_price"] <= max_price]
    return {"count": len(products), "products": products}


@app.post("/recommendations")
def create_recommendation(request: RecommendationRequest) -> dict[str, Any]:
    return recommend(request.requirements, request.budget, request.preferred_ram_gb)


@app.post("/faqs")
def ask_faq(request: FAQRequest) -> dict[str, Any]:
    if request.product_id and get_product(request.product_id) is None:
        raise HTTPException(status_code=404, detail="Product not found in the sample catalog.")
    return answer_faq(request.question, request.product_id)


@app.post("/cart/confirm")
def confirm_cart_item(request: CartConfirmation) -> dict[str, Any]:
    if not request.confirmed:
        raise HTTPException(
            status_code=409,
            detail="Customer confirmation is required before adding an item to the cart.",
        )
    product = get_product(request.product_id)
    if product is None:
        raise HTTPException(status_code=404, detail="Product not found in the sample catalog.")
    existing_quantity = cart_items.get(request.product_id, {}).get("quantity", 0)
    new_quantity = existing_quantity + request.quantity
    if product["stock"] < new_quantity:
        raise HTTPException(
            status_code=409,
            detail=f"Only {product['stock']} unit(s) are available in sample inventory.",
        )
    cart_items[request.product_id] = {
        "product_id": product["product_id"],
        "name": product["name"],
        "quantity": new_quantity,
        "unit_price": product["effective_price"],
        "currency": product["currency"],
        "line_total": new_quantity * product["effective_price"],
    }
    return {"message": "Item added to the simulated cart.", **get_cart()}


@app.get("/cart")
def get_cart() -> dict[str, Any]:
    items = list(cart_items.values())
    total = sum(item["line_total"] for item in items)
    return {"items": items, "total": total, "currency": "INR"}


@app.delete("/cart")
def clear_cart() -> dict[str, str]:
    cart_items.clear()
    return {"message": "Simulated cart cleared."}