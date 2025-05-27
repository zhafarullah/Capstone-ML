# app.py

import os
import logging
import nltk
import gdown
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict

# ── Setup Logging ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Akses Crowdin dan NLTK ────────────────────────────────────────────────────
logger.info("🚀 Starting app.py")
nltk.download("punkt")  # pastikan punkt ada
logger.info("📂 CWD: %s", os.getcwd())
logger.info("📁 Files in root: %s", os.listdir())

# ── Download nama_file.csv jika belum ada ─────────────────────────────────────
def download_nama_file():
    file_id = "1Y3EO2xEUxNZf02yUUYEf6H6xO3zhpNHY"
    output = os.path.join(os.getcwd(), "nama_file.csv")
    if not os.path.exists(output):
        logger.info("📥 Downloading nama_file.csv from Drive…")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output, quiet=False)
    else:
        logger.info("✅ nama_file.csv already exists")

download_nama_file()

# ── Import Helpers & Models ──────────────────────────────────────────────────
try:
    from carbon_calculator import calculate_total_carbon_from_items
    from main import parse_ingredients, recommend_recipes, UNIT_FACTORS
    logger.info("✅ Helpers loaded successfully")
except Exception as e:
    logger.exception("❌ Error importing helpers: %s", e)
    raise

# ── FastAPI App & Schemas ────────────────────────────────────────────────────
app = FastAPI()

class IngredientInput(BaseModel):
    text: str

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

# ── Fallback: jika dipanggil via python app.py langsung, jalankan Uvicorn ───
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info("▶️ Launching Uvicorn on 0.0.0.0:%d", port)
    uvicorn.run("app:app", host="0.0.0.0", port=port, log_level="info")
