import os
import re
from typing import Any, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from app.tools import check_inventory, compare_products, search_products, validate_budget


load_dotenv()


class ShoppingState(TypedDict, total=False):
    request_text: str
    budget: int | None
    preferred_ram_gb: int | None
    parsed_requirements: dict[str, Any]
    products: list[dict[str, Any]]
    comparison: list[dict[str, Any]]
    inventory: dict[str, dict[str, Any]]
    budget_checks: dict[str, dict[str, Any]]
    recommendations: list[dict[str, Any]]
    status: str
    message: str


class RequirementExtraction(BaseModel):
    budget: int | None = Field(default=None, description="Explicit customer budget in the catalog currency")
    preferred_ram_gb: int | None = Field(default=None, description="Requested minimum RAM in gigabytes")
    needs: list[str] = Field(default_factory=list, description="Product uses and preferences stated by the customer")
    search_query: str = Field(description="Concise search phrase using only the requested product and needs")


def parse_requirements(text: str, budget_override: int | None) -> dict[str, Any]:
    budget = budget_override
    if budget is None:
        budget_match = re.search(
            r"(?:under|below|less than|within|budget(?: of)?|rs\.?)[^\d]{0,12}([\d,]+)",
            text,
            re.IGNORECASE,
        )
        if budget_match:
            budget = int(budget_match.group(1).replace(",", ""))

    ram_match = re.search(r"(\d+)\s*gb\s*(?:ram|memory)?", text, re.IGNORECASE)
    preferred_ram = int(ram_match.group(1)) if ram_match else None
    return {
        "budget": budget,
        "preferred_ram_gb": preferred_ram,
        "needs": [
            need for keyword, need in (
                ("python", "Python programming"),
                ("machine learning", "machine learning basics"),
                ("class", "online classes"),
                ("battery", "long battery life"),
            )
            if keyword in text.lower()
        ],
    }


def extract_requirements_with_llm(text: str) -> dict[str, Any] | None:
    if not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0,
            timeout=15,
        ).with_structured_output(RequirementExtraction)
        extraction = model.invoke(
            "Extract only requirements explicitly stated by the customer. Do not infer a budget, "
            "RAM preference, or product category. Return a concise catalog search phrase. "
            f"Customer request: {text}"
        )
        return extraction.model_dump()
    except Exception:
        return None


def parse_node(state: ShoppingState) -> dict[str, Any]:
    request_text = state["request_text"]
    parsed = parse_requirements(request_text, state.get("budget"))
    extracted = extract_requirements_with_llm(request_text)
    if extracted:
        if state.get("budget") is None and extracted.get("budget") is not None:
            parsed["budget"] = extracted["budget"]
        if parsed["preferred_ram_gb"] is None and extracted.get("preferred_ram_gb") is not None:
            parsed["preferred_ram_gb"] = extracted["preferred_ram_gb"]
        if extracted.get("needs"):
            parsed["needs"] = extracted["needs"]
        parsed["search_query"] = extracted["search_query"]
    if state.get("preferred_ram_gb") is not None:
        parsed["preferred_ram_gb"] = state["preferred_ram_gb"]
    return {"parsed_requirements": parsed}


def search_node(state: ShoppingState) -> dict[str, Any]:
    parsed = state["parsed_requirements"]
    products = search_products.invoke(
        {
            "query": parsed.get("search_query", state["request_text"]),
            "max_price": parsed["budget"],
            "preferred_ram_gb": parsed["preferred_ram_gb"],
        }
    )
    return {"products": products}


def comparison_node(state: ShoppingState) -> dict[str, Any]:
    product_ids = [product["product_id"] for product in state["products"]]
    return {"comparison": compare_products.invoke({"product_ids": product_ids})}


def inventory_node(state: ShoppingState) -> dict[str, Any]:
    product_ids = [product["product_id"] for product in state["products"]]
    inventory = check_inventory.invoke({"product_ids": product_ids})
    return {"inventory": {item["product_id"]: item for item in inventory}}


def budget_node(state: ShoppingState) -> dict[str, Any]:
    product_ids = [product["product_id"] for product in state["products"]]
    budget = state["parsed_requirements"]["budget"]
    checks = validate_budget.invoke({"product_ids": product_ids, "budget": budget})
    return {"budget_checks": {item["product_id"]: item for item in checks}}


def recommendation_node(state: ShoppingState) -> dict[str, Any]:
    inventory = state["inventory"]
    budget_checks = state["budget_checks"]
    preferred_ram_gb = state["parsed_requirements"]["preferred_ram_gb"]
    products_by_id = {product["product_id"]: product for product in state["products"]}
    eligible = [
        product for product in state["comparison"]
        if inventory[product["product_id"]]["in_stock"]
        and budget_checks[product["product_id"]]["within_budget"]
        and (
            preferred_ram_gb is None
            or product["ram_gb"] >= preferred_ram_gb
        )
    ]
    eligible.sort(
        key=lambda product: (
            product["ram_gb"],
            product["battery_hours"],
            products_by_id[product["product_id"]]["search_score"],
            -product["effective_price"],
        ),
        reverse=True,
    )

    recommendations = []
    for product in eligible[:3]:
        stock = inventory[product["product_id"]]
        recommendations.append({
            **product,
            "budget_status": "within budget",
            "inventory_status": "in stock",
            "stock": stock["stock"],
            "delivery_days": stock["delivery_days"],
            "explanation": (
                f"{product['ram_gb']} GB RAM, {product['processor']}, and "
                f"{product['battery_hours']} hours listed battery life; "
                f"the discounted price is {product['currency']} {product['effective_price']:,}."
            ),
        })

    if recommendations:
        status = "matches_found"
        message = "Ranked matches are based on the catalog, stated preferences, stock, and budget."
    elif not state["products"]:
        status = "no_products_found"
        message = "No products matched this request in the sample catalog. Try broader requirements."
    else:
        if not any(check["within_budget"] for check in budget_checks.values()):
            status = "no_available_match"
            message = "Matching products were found, but none fit the stated budget after discounts."
        elif not any(item["in_stock"] for item in inventory.values()):
            status = "no_available_match"
            message = "Matching products were found, but all are currently out of sample stock."
        elif preferred_ram_gb is not None and any(
            inventory[product_id]["in_stock"] and budget_checks[product_id]["within_budget"]
            for product_id in inventory
        ):
            status = "no_preference_match"
            message = (
                "In-stock products fit the budget but do not meet the requested RAM preference. "
                "See the comparison for lower-RAM alternatives."
            )
        else:
            status = "no_available_match"
            message = "No product is both within budget and in stock. Review the comparison for alternatives."

    comparison = []
    for product in state["comparison"]:
        stock = inventory[product["product_id"]]
        budget_check = budget_checks[product["product_id"]]
        comparison.append({
            **product,
            "budget_status": "within budget" if budget_check["within_budget"] else "over budget",
            "inventory_status": "in stock" if stock["in_stock"] else "out of stock",
            "stock": stock["stock"],
            "delivery_days": stock["delivery_days"],
        })

    return {
        "comparison": comparison,
        "recommendations": recommendations,
        "status": status,
        "message": message,
    }


def build_workflow():
    graph = StateGraph(ShoppingState)
    graph.add_node("requirements_agent", parse_node)
    graph.add_node("product_search_agent", search_node)
    graph.add_node("comparison_agent", comparison_node)
    graph.add_node("inventory_agent", inventory_node)
    graph.add_node("budget_agent", budget_node)
    graph.add_node("recommendation_agent", recommendation_node)
    graph.add_edge(START, "requirements_agent")
    graph.add_edge("requirements_agent", "product_search_agent")
    graph.add_edge("product_search_agent", "comparison_agent")
    graph.add_edge("comparison_agent", "inventory_agent")
    graph.add_edge("inventory_agent", "budget_agent")
    graph.add_edge("budget_agent", "recommendation_agent")
    graph.add_edge("recommendation_agent", END)
    return graph.compile()


shopping_workflow = build_workflow()


def recommend(
    request_text: str,
    budget: int | None = None,
    preferred_ram_gb: int | None = None,
) -> dict[str, Any]:
    result = shopping_workflow.invoke({
        "request_text": request_text,
        "budget": budget,
        "preferred_ram_gb": preferred_ram_gb,
    })
    return {
        "status": result["status"],
        "message": result["message"],
        "parsed_requirements": result["parsed_requirements"],
        "comparison": result["comparison"],
        "recommendations": result["recommendations"],
        "llm_enabled": bool(os.getenv("OPENAI_API_KEY")),
    }
