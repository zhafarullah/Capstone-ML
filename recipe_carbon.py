import pandas as pd
from inference import estimate_carbon_from_item
from parser import unit_conversion

def calculate_carbon_from_recipe_file(csv_path="clean_recipes.csv", output_path="carbon_recipe.csv"):
    df = pd.read_csv(csv_path)
    results = []

    for _, row in df.iterrows():
        ingredient = row["name"]

        # Lewatkan jika tidak valid
        if pd.isna(ingredient) or not isinstance(ingredient, str):
            continue

        quantity = float(row["quantity"])
        unit = row.get("unit", "gram")

        # Lewatkan jika quantity tidak valid
        if pd.isna(quantity):
            continue

        unit = str(unit).lower()

        # Konversi ke gram jika perlu
        if unit != "gram":
            quantity *= unit_conversion.get(unit, 1)

        qty_kg = quantity / 1000
        est = estimate_carbon_from_item(ingredient)
        carbon = round(qty_kg * est['carbon_footprint'], 4)

        results.append({
        "recipe_id": row["Unnamed: 0"],
        "title": row["Title"],
        "ingredient": ingredient,
        "quantity": round(quantity),
        "unit": "gram",
        "carbon_item": est['carbon_footprint'],  # nilai estimasi
        "carbon_score": carbon,  # nilai dikali kuantitas
        "matched_items": est.get("matched_to", ""),
        "estimated_items": est.get("method", "")
    })


    df_result = pd.DataFrame(results)

    # Hitung total karbon per resep
    total_carbon = df_result.groupby(["recipe_id", "title"])["carbon_score"].sum().reset_index()
    total_carbon = total_carbon.rename(columns={"carbon_score": "total_recipe_carbon"})

    # Gabungkan hasil total dengan detail bahan
    merged = pd.merge(df_result, total_carbon, on=["recipe_id", "title"])

    merged["carbon_item"] = merged["carbon_item"].round(3)
    merged["carbon_score"] = merged["carbon_score"].round(3)
    merged["total_recipe_carbon"] = merged["total_recipe_carbon"].round(3)

    # Simpan ke CSV
    columns_to_save = merged.drop(columns=["matched_items", "estimated_items"])
    columns_to_save.to_csv(output_path, index=False)
    print(f"Hasil perhitungan disimpan ke: {output_path}")

    return merged

if __name__ == "__main__":
    calculate_carbon_from_recipe_file()
