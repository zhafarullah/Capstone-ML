# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
import nltk; nltk.download("punkt")
import os
import gdown

print("🚀 Start app.py")

def download_nama_file():
    file_id = "1Y3EO2xEUxNZf02yUUYEf6H6xO3zhpNHY"
    output = "nama_file.csv"
    if not os.path.exists(output):
        print("📥 Mengunduh nama_file.csv dari Google Drive...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output, quiet=False)
    else:
        print("✅ File nama_file.csv sudah ada")

download_nama_file()
print("✅ nama_file.csv ready")

from carbon_calculator import calculate_total_carbon_from_items
print("✅ carbon_calculator loaded")

from main import parse_ingredients, recommend_recipes, UNIT_FACTORS
print("✅ main.py loaded")

# === STEP 2: Import fungsi setelah file tersedia ===
from carbon_calculator import calculate_total_carbon_from_items
from main import parse_ingredients, recommend_recipes, UNIT_FACTORS

# === STEP 3: FastAPI routes ===
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
