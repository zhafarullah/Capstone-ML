# app.py

import os
import logging
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

logger.info("🚀 Booting app.py")


try:
    from carbon_calculator import calculate_total_carbon_from_items
    from main import parse_ingredients, recommend_recipes, UNIT_FACTORS
    from logic import process_user_input 
    logger.info("✅ Helpers loaded")
except Exception:
    logger.exception("❌ Failed to load helpers")
    raise

app = FastAPI(title="Carbon + Recipe API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class IngredientInput(BaseModel):
    text: str

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
