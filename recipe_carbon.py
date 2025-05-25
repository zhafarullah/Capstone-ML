import pandas as pd
from inference import estimate_carbon_from_item
from parser import unit_conversion

def calculate_carbon_from_recipe_file(csv_path="nama_file.csv"):
    df = pd.read_csv(csv_path)
    results = []

    for _, row in df.iterrows():
        ingredient = row["name"]

        # Lewatkan jika tidak valid
        if pd.isna(ingredient) or not isinstance(ingredient, str):
            results.append(None)
            continue

        quantity = float(row["quantity"])
        unit = row.get("unit", "gram")

        # Lewatkan jika quantity tidak valid
        if pd.isna(quantity):
            results.append(None)
            continue

        unit = str(unit).lower()

        # Konversi ke gram jika perlu
        if unit != "gram":
            quantity *= unit_conversion.get(unit, 1)

        qty_kg = quantity / 1000
        est = estimate_carbon_from_item(ingredient)
        carbon = round(qty_kg * est['carbon_footprint'], 4)
        results.append(carbon)

    # Tambahkan kolom perhitungan karbon per bahan
    df["carbon_score"] = results

    # Hitung total karbon per resep berdasarkan kolom "0" dan "Title"
    df["total_recipe_carbon"] = df.groupby(["Unnamed: 0", "Title"])["carbon_score"].transform("sum").round(3)

    # Simpan ke file yang sama
    df.to_csv(csv_path, index=False)
    print(f"Hasil perhitungan disimpan ke: {csv_path}")

    return df

if __name__ == "__main__":
    calculate_carbon_from_recipe_file()
