# app.py

import os
import logging
import nltk
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
from fastapi.middleware.cors import CORSMiddleware

# ── Setup Logging ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

# ── Init ──────────────────────────────────────────────────────
logger.info("🚀 Booting app.py")
nltk.download("punkt", quiet=True)

# ── Import modul helper ────────────────────────────────────────
try:
    from carbon_calculator import calculate_total_carbon_from_items
    from main import parse_ingredients, recommend_recipes, UNIT_FACTORS
    from logic import process_user_input  # Jika kamu punya pipeline lengkap
    logger.info("✅ Helpers loaded")
except Exception:
    logger.exception("❌ Failed to load helpers")
    raise

# ── FastAPI setup ──────────────────────────────────────────────
app = FastAPI(title="Carbon + Recipe API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ganti dengan URL frontend-mu jika ingin membatasi
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request Model ─────────────────────────────────────────────
class IngredientInput(BaseModel):
    text: str

# ── Endpoints ─────────────────────────────────────────────────
@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/carbon")
def carbon(input: IngredientInput) -> Dict:
    items = parse_ingredients(input.text)
    total = calculate_total_carbon_from_items(items, UNIT_FACTORS)
    return {"items": items, "total_carbon": total}

@app.post("/recipes")
def recipes(input: IngredientInput) -> Dict:
    items = parse_ingredients(input.text)
    recs = recommend_recipes(input.text, top_n=5).to_dict(orient="records")
    return {"items": items, "recipes": recs}

@app.post("/full")
def full_pipeline(input: IngredientInput) -> Dict:
    return process_user_input(input.text)
