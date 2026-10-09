# Agentic Shopping Assistant

A FastAPI training project for an agentic e-commerce shopping workflow. The sample catalog is illustrative and is not connected to retailers or live inventory.

## Features

- Natural-language shopping requirements, with budget and RAM extracted from the request.
- LangGraph workflow with separate requirement, search, comparison, inventory, budget, and recommendation agents.
- LangChain tools backed by a CSV product catalog.
- Optional OpenAI structured extraction for natural-language requirements, with deterministic parsing as a no-key fallback.
- Ranked product comparisons with discounted price, stock, delivery estimate, and recommendation explanations.
- FAQ retrieval using TF-IDF over the FAQ CSV. When an OpenAI key is configured, ChatOpenAI generates a concise answer grounded in retrieved FAQ entries; otherwise matching FAQ answers are returned directly.
- Simulated cart with explicit customer confirmation, stock validation, and quantity limits.
- No API key is needed for recommendations, catalog search, FAQ retrieval, or cart simulation.

## Run

From the project directory in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for the interactive API. To enable OpenAI-backed FAQ answer generation, set `OPENAI_API_KEY` in `.env`. The rest of the application works without it.

## Deploy on Render

1. Push this project to a GitHub repository. Confirm `.env` is not committed; it is excluded by `.gitignore`.
2. In Render, choose **New > Blueprint**, connect the GitHub repository, and apply the settings from `render.yaml`.
3. Wait for the build and deploy to finish. Render will provide a public service URL.
4. Open `<your-service-url>/health`; expect `{"status":"ok"}`. Then open `<your-service-url>/docs` to try the API.
5. To enable OpenAI calls, add `OPENAI_API_KEY` under the service's environment variables in Render, then redeploy. Do not put the key in `render.yaml` or commit it to GitHub.

This deployment is for demos and training: the catalog is sample data, and the simulated cart is shared in process memory and resets when the service restarts. Do not use it for real orders or private customer data.

## Example

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/recommendations `
  -ContentType 'application/json' `
  -Body '{"requirements":"I need a laptop under 60,000 for Python programming, machine learning basics, and online classes. I prefer 16 GB RAM and good battery life."}'
```

Add a product only after the customer confirms the choice:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/cart/confirm `
  -ContentType 'application/json' `
  -Body '{"product_id":"LT-100","quantity":1,"confirmed":true}'
```

## Endpoints

- `GET /health`
- `GET /products?q=python&max_price=60000`
- `POST /recommendations`
- `POST /faqs`
- `POST /cart/confirm`
- `GET /cart`
- `DELETE /cart`

The scenario's natural-language request says the budget is 60,000; that stated amount is used rather than the conflicting 260,000 in the introductory sentence. Prices and stock are sample INR values. The cart is in memory and resets when the API process restarts.

## Tests

```powershell
python -m unittest discover -s tests -v
```