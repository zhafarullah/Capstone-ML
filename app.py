# app.py

import os
import logging
import nltk
import gdown
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict

# ── Setup Logging ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

# ── Download asset jika perlu ──────────────────────────────────
logger.info("🚀 Booting app.py")
nltk.download("punkt", quiet=True)

DATA_DIR = os.getcwd()
CSV_NAME = "nama_file.csv"
CSV_PATH = os.path.join(DATA_DIR, CSV_NAME)

if not os.path.exists(CSV_PATH):
    logger.info("📥 Downloading %s …", CSV_NAME)
    gdown.download(
        "https://drive.google.com/uc?id=1qYTi8dJlMGNstWSFO7voOqfJyhzjGSPq",
        CSV_PATH,
        quiet=False
    )
else:
    logger.info("✅ %s already exists", CSV_NAME)

# ── Import modul helper ────────────────────────────────────────
try:
    from carbon_calculator import calculate_total_carbon_from_items
    from main import parse_ingredients, recommend_recipes, UNIT_FACTORS
    logger.info("✅ Helpers loaded")
except Exception:
    logger.exception("❌ Failed to load helpers")
    raise

# ── FastAPI setup ──────────────────────────────────────────────
app = FastAPI(title="Carbon + Recipe API")

class IngredientInput(BaseModel):
    text: str

@app.get("/")
def health_check():
    """
    Simple health check.
    """
    return {"status": "ok"}

@app.post("/carbon")
def carbon(input: IngredientInput) -> Dict:
    """
    Hitung total jejak karbon dari teks bahan yang diberikan.
    """
    items = parse_ingredients(input.text)
    total = calculate_total_carbon_from_items(items, UNIT_FACTORS)
    return {"items": items, "total_carbon": total}

@app.post("/recipes")
def recipes(input: IngredientInput) -> Dict:
    """
    Rekomendasi resep berdasarkan teks bahan yang diberikan.
    """
    items = parse_ingredients(input.text)
    recs = recommend_recipes(input.text, top_n=5).to_dict(orient="records")
    return {"items": items, "recipes": recs}
