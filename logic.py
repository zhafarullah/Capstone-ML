from main import parse_ingredients, recommend_recipes, parse_number, df_exploded, UNIT_FACTORS
from carbon_calculator import calculate_total_carbon_from_items
from rapidfuzz import fuzz, process
import textwrap
import re

def process_user_input(text, selected_index=0, top_n=5, fuzzy_threshold=75):
    parsed = parse_ingredients(text)
    total_cf = calculate_total_carbon_from_items(parsed, UNIT_FACTORS)

    matched_names = []
    res_df = recommend_recipes(text, top_n=top_n, fuzzy_threshold=fuzzy_threshold)
    if res_df.empty:
        return {
            "parsed_ingredients": parsed,
            "total_carbon": total_cf,
            "matched_names": [],
            "recommended_recipes": [],
            "selected_recipe": None,
            "leftovers": [],
            "missing": []
        }

    recipes = res_df.reset_index(drop=True)
    if selected_index >= len(recipes):
        selected_index = 0
    selected = recipes.iloc[selected_index]
    title = selected["title_cleaned"]

    instr_lines = []
    split_pattern = re.compile(r'(?<!\b\d)(?<!tsp)(?<!tbsp)\.\s+')
    for part in split_pattern.split(selected["instructions_cleaned"].strip()):
        sent = part.strip().rstrip('.')
        if sent:
            instr_lines.append(sent)

    ingredients = selected["cleaned_ingredients"]

    input_amounts = {}
    for it in parsed:
        qty = parse_number(it["quantity"])
        unit = (it["unit"] or "").lower()
        if qty is None or unit not in UNIT_FACTORS:
            continue
        input_amounts.setdefault(it["ingredient"], 0.0)
        input_amounts[it["ingredient"]] += qty * UNIT_FACTORS[unit]

    df_sel = df_exploded[df_exploded["title_cleaned"] == title].copy()
    df_sel["factor"] = df_sel["unit"].str.lower().map(UNIT_FACTORS)
    df_sel = df_sel[df_sel["factor"].notna()]
    df_sel["required_base"] = df_sel["quantity"] * df_sel["factor"]

    used_amounts = df_sel.groupby("pure_name")["required_base"].sum().to_dict()

    leftovers = []
    for inp_name, avail_base in input_amounts.items():
        match, score, _ = process.extractOne(inp_name, list(used_amounts), scorer=fuzz.token_set_ratio)
        required = used_amounts.get(match, 0.0) if score >= 60 else 0.0
        sisa_base = avail_base - required
        if sisa_base > 0:
            orig_unit = next((it["unit"] for it in parsed if it["ingredient"] == inp_name), None)
            if orig_unit and orig_unit.lower() in UNIT_FACTORS:
                sisa_qty = sisa_base / UNIT_FACTORS[orig_unit.lower()]
                leftovers.append(f"{sisa_qty:.2f} {orig_unit} {inp_name}")
            else:
                leftovers.append(f"{sisa_base:.2f} base-unit {inp_name}")

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
        match, score, _ = process.extractOne(name, list(input_amounts), scorer=fuzz.token_set_ratio)
        avail = input_amounts.get(match, 0.0) if score >= 60 else 0.0
        if avail < req_base:
            short_base = req_base - avail
            short_qty = short_base / factor
            missing.append(f"{short_qty:.2f} {unit} {name}")

    return {
        "parsed_ingredients": parsed,
        "total_carbon": round(total_cf, 3),
        "recommended_recipes": recipes["title_cleaned"].tolist(),
        "selected_recipe": {
            "title": title,
            "instructions": instr_lines,
            "ingredients": ingredients
        },
        "leftovers": leftovers,
        "missing": missing
    }

