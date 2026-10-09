import unittest

from fastapi.testclient import TestClient

from app.main import app, cart_items


class ShoppingAssistantApiTests(unittest.TestCase):
    def setUp(self) -> None:
        cart_items.clear()
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.client.close()

    def test_root_redirects_to_api_docs(self) -> None:
        response = self.client.get("/", follow_redirects=False)

        self.assertEqual(response.status_code, 307)
        self.assertEqual(response.headers["location"], "/docs")

    def test_recommendations_check_budget_stock_and_requirements(self) -> None:
        response = self.client.post(
            "/recommendations",
            json={
                "requirements": (
                    "I need a laptop under 60,000 for Python programming, machine learning basics, "
                    "and online classes. I prefer 16 GB RAM and good battery life."
                )
            },
        )

        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["status"], "matches_found")
        self.assertEqual(result["parsed_requirements"]["budget"], 60000)
        self.assertTrue(result["recommendations"])
        self.assertTrue(all(item["effective_price"] <= 60000 for item in result["recommendations"]))
        self.assertTrue(all(item["stock"] > 0 for item in result["recommendations"]))
        self.assertTrue(all(item["ram_gb"] >= 16 for item in result["recommendations"]))
        self.assertTrue(any(item["inventory_status"] == "out of stock" for item in result["comparison"]))

    def test_cart_requires_confirmation_and_checks_stock(self) -> None:
        unconfirmed = self.client.post(
            "/cart/confirm",
            json={"product_id": "LT-100", "quantity": 1, "confirmed": False},
        )
        self.assertEqual(unconfirmed.status_code, 409)
        self.assertEqual(self.client.get("/cart").json()["items"], [])

        confirmed = self.client.post(
            "/cart/confirm",
            json={"product_id": "LT-100", "quantity": 2, "confirmed": True},
        )
        self.assertEqual(confirmed.status_code, 200)
        self.assertEqual(confirmed.json()["items"][0]["quantity"], 2)

        unavailable = self.client.post(
            "/cart/confirm",
            json={"product_id": "LT-102", "quantity": 1, "confirmed": True},
        )
        self.assertEqual(unavailable.status_code, 409)

    def test_unmatched_request_returns_no_products(self) -> None:
        response = self.client.post(
            "/recommendations",
            json={"requirements": "I need a telescope for astronomy and stargazing."},
        )

        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["status"], "no_products_found")
        self.assertEqual(result["comparison"], [])
        self.assertEqual(result["recommendations"], [])

    def test_product_search_filters_by_effective_price(self) -> None:
        response = self.client.get("/products", params={"q": "python", "max_price": 60000})

        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertTrue(result["products"])
        self.assertTrue(all(product["effective_price"] <= 60000 for product in result["products"]))

    def test_faq_retrieves_grounded_sources(self) -> None:
        response = self.client.post(
            "/faqs",
            json={"question": "How long does the ZenBook Lite 14 battery last?", "product_id": "LT-100"},
        )

        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertIn("17 hours", result["answer"])
        self.assertEqual(result["sources"][0]["faq_id"], "FAQ-004")


if __name__ == "__main__":
    unittest.main()
