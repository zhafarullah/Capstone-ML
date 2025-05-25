import pandas as pd
from inference import estimate_carbon_from_item
from parser import parse_ingredient

def calculate_carbon_for_ingredients(user_input):
    parsed = parse_ingredient(user_input)
    if not parsed:
        raise ValueError("Input tidak valid atau tidak ada bahan yang dikenali.")
    
    results = []

    for item in parsed:
        name = item["ingredient"]
        qty_gram = item["quantity"]
        qty_kg = qty_gram / 1000  # konversi ke kg

        est = estimate_carbon_from_item(name)
        carbon = round(qty_kg * est['carbon_footprint'], 4)

        results.append({
            "name": name,
            "quantity": qty_gram,
            "unit": "gram",
            "carbon_item": est['carbon_footprint'],  # per item
            "carbon_score": carbon,  # dikali kuantitas
            "matched_items": est.get("matched_to", ""),
            "estimated_items": est.get("method", "")
        })

    return pd.DataFrame(results)

