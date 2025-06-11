from main import parse_ingredients, recommend_recipes, parse_number, df_exploded, UNIT_FACTORS
from carbon_calculator import calculate_total_carbon_from_items
from rapidfuzz import fuzz, process
import re
import nltk
nltk.download('punkt', quiet=True)
def process_user_input(text, top_n=5, fuzzy_threshold=75):
    parsed = parse_ingredients(text)
    total_cf = calculate_total_carbon_from_items(parsed, UNIT_FACTORS)
    input_amounts = {}

    for it in parsed:
        qty = parse_number(it["quantity"])
        unit = (it["unit"] or "").lower()
        if qty is None or unit not in UNIT_FACTORS:
            continue
        input_amounts.setdefault(it["ingredient"], 0.0)
        input_amounts[it["ingredient"]] += qty * UNIT_FACTORS[unit]

    res_df = recommend_recipes(text, top_n=top_n, fuzzy_threshold=fuzzy_threshold)
    if res_df.empty:
        return {
            "parsed_ingredients": parsed,
            "total_carbon": total_cf,
            "recommended_recipes": []
        }

    res_df = res_df.reset_index(drop=True)
    results = []

    for _, row in res_df.iterrows():
        title = row["title_cleaned"]
        title_display = row.get("title_display", title)

        df_sel = df_exploded[df_exploded["title_cleaned"] == title].copy()
        df_sel["factor"] = df_sel["unit"].str.lower().map(UNIT_FACTORS)
        df_sel = df_sel[df_sel["factor"].notna()]
        df_sel["required_base"] = df_sel["quantity"] * df_sel["factor"]
        used_amounts = df_sel.groupby("pure_name")["required_base"].sum().to_dict()

        # Hitung leftovers
        used = []
        for inp_name, avail_base in input_amounts.items():
            result = process.extractOne(inp_name, list(used_amounts), scorer=fuzz.token_set_ratio)
            required = (used_amounts.get(result[0], 0.0) if result and result[1]>=fuzzy_threshold else 0.0)
            used_base = min(avail_base, required)
            if used_base>0:
                orig_unit = next((it["unit"] for it in parsed if it["ingredient"]==inp_name), None)
                used_qty = used_base/UNIT_FACTORS.get(orig_unit.lower(),1)
                used.append(f"{used_qty:.2f} {orig_unit} {inp_name}")
        leftovers = used  # rename or overwrite
        # Susun dict untuk carbon terpakai
        used_dicts = []
        for inp_name, avail_base in input_amounts.items():
            # cari match dan jumlah terpakai (sama logika sisa tetapi min(avail, req))
            result = process.extractOne(inp_name, list(used_amounts), scorer=fuzz.token_set_ratio)
            required = (used_amounts.get(result[0], 0.0) if result and result[1] >= 60 else 0.0)
            used_base = min(avail_base, required)
            if used_base > 0:
                orig_unit = next((it["unit"] for it in parsed if it["ingredient"] == inp_name), None)
                if orig_unit and orig_unit.lower() in UNIT_FACTORS:
                    used_qty = used_base / UNIT_FACTORS[orig_unit.lower()]
                    used_dicts.append({
                        "quantity": str(used_qty),
                        "unit":     orig_unit,
                        "ingredient": inp_name
                    })
        total_used_carbon = calculate_total_carbon_from_items(used_dicts, UNIT_FACTORS)
        # Hitung missing
        df_group = (
            df_sel.groupby("pure_name")
            .agg({
                "required_base": "sum",
                "unit": lambda s: s.iloc[0],
                "factor": lambda s: s.iloc[0]
            })
            .reset_index()
        )

        missing = []
        for _, r in df_group.iterrows():
            name = r["pure_name"]
            req_base = r["required_base"]
            unit = r["unit"]
            factor = r["factor"]

            result = process.extractOne(name, list(input_amounts), scorer=fuzz.token_set_ratio)
            avail = 0.0
            if result is not None:
                match, score, _ = result
                if score >= 60:
                    avail = input_amounts.get(match, 0.0)

            if avail < req_base:
                short_base = req_base - avail
                short_qty = short_base / factor
                missing.append(f"{short_qty:.2f} {unit} {name}")

        miss_dicts = []
        for _, r in df_group.iterrows():
            name   = r["pure_name"]
            req    = r["required_base"]
            factor = r["factor"]
            unit   = r["unit"]
            result = process.extractOne(name, list(input_amounts), scorer=fuzz.token_set_ratio)
            avail  = (input_amounts.get(result[0], 0.0) if result and result[1] >= 60 else 0.0)
            if avail < req:
                short_base = req - avail
                short_qty  = short_base / factor
                miss_dicts.append({
                    "quantity": str(short_qty),
                    "unit":     unit,
                    "ingredient": name
                })
        total_missing_carbon = calculate_total_carbon_from_items(miss_dicts, UNIT_FACTORS)

        # ─── Terakhir, efisiensi ───
        total_recipe_carbon = total_used_carbon + total_missing_carbon
        efficiency = (total_used_carbon / total_recipe_carbon) if total_recipe_carbon > 0 else 0.0

        # Masukkan semua ke hasil
        results.append({
            "title":                   title_display,
            "used":               leftovers,
            "total_used_carbon":       round(total_used_carbon, 3),
            "missing":                 missing,
            "total_missing_carbon":    round(total_missing_carbon, 3),
            "efficiency":              round(efficiency, 3)
        })

    return {
        "parsed_ingredients": parsed,
        "total_carbon": round(total_cf, 3),
        "recommended_recipes": results
    }
