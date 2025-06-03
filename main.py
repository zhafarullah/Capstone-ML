import json
import numpy as np
import pandas as pd
import re
import ast
from fractions import Fraction
import textwrap
from pprint import pprint
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle
from rapidfuzz import fuzz, process
from carbon_calculator import calculate_total_carbon_from_items
import gdown
import os

#===========Mongo IMPORT===============#
from pymongo import MongoClient

# URI MongoDB Atlas (jangan dibagikan publik)
MONGO_URI = "mongodb+srv://anzzanafa:fWZJzU2FGfWlobHY@cluster0.1xeasvn.mongodb.net/ecorecipes?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(MONGO_URI)
db = client['ecorecipes']
collection = db['recipes']
'''
# ============== AUTO DOWNLOAD CSV BESAR ====================
def download_nama_file():
    file_id = "1qYTi8dJlMGNstWSFO7voOqfJyhzjGSPq"
    output = "nama_file.csv"
    if not os.path.exists(output):
        print("📥 Mengunduh nama_file.csv dari Google Drive...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output, quiet=False)
# ===========================================================

# Pastikan file tersedia sebelum dibaca
download_nama_file()
'''
# — Load machine learning artifacts
model = load_model('best_ner_bilstm.h5')
tok2idx = pickle.load(open('tok2idx.pkl', 'rb'))
le = pickle.load(open('label_encoder.pkl', 'rb'))

# — Constants
MAXLEN = 50

# — Load recipe dataset (ubah path sesuai lokasi file Anda)
df_exploded = pd.DataFrame(list(collection.find({})))

# Konversi kolom jika perlu (misal: cleaned_ingredients dari string ke list)
if 'cleaned_ingredients' in df_exploded.columns:
    df_exploded['cleaned_ingredients'] = df_exploded['cleaned_ingredients'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

df_exploded['pure_name'] = df_exploded['pure_name'].fillna('').astype(str)
PURE_NAMES = df_exploded['pure_name'].unique().tolist()

UNIT_FACTORS = {
    # Mass
    'mg': 0.001, 'milligram': 0.001, 'milligrams': 0.001,
    'g': 1, 'gram': 1, 'grams': 1,
    'kg': 1000, 'kilo': 1000, 'kilogram': 1000, 'kilograms': 1000,
    'oz': 28.3495, 'ounce': 28.3495, 'ounces': 28.3495,
    'lb': 453.592, 'lbs': 453.592, 'pound': 453.592, 'pounds': 453.592,

    # Volume (assume density ≈ water)
    'ml': 1, 'milliliter': 1, 'milliliters': 1, 'millilitre': 1, 'millilitres': 1,
    'l': 1000, 'liter': 1000, 'litre': 1000, 'liters': 1000, 'litres': 1000,
    'dl': 100, 'deciliter': 100, 'deciliters': 100, 'decilitre': 100, 'decilitres': 100,
    'tsp': 4.92892, 'teaspoon': 4.92892, 'teaspoons': 4.92892,
    'tbsp': 14.7868, 'tablespoon': 14.7868, 'tablespoons': 14.7868,
    'cup': 240, 'cups': 240, 'c': 240,
    'pt': 473.176, 'pint': 473.176, 'pints': 473.176,
    'qt': 946.353, 'quart': 946.353, 'quarts': 946.353,
    'gal': 3785.41, 'gallon': 3785.41, 'gallons': 3785.41,
}

# — Helper: konversi string kuantitas → angka
def parse_number(q):
    q = q.strip().lower()
    if '/' in q:
        try:
            return float(Fraction(q))
        except:
            pass
    WORD2NUM = {'half':0.5, 'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8}
    if q in WORD2NUM:
        return WORD2NUM[q]
    try:
        return float(q)
    except:
        return None

# — Fungsi NER: parse ingredients dari teks input
def parse_ingredients(text):
    toks = text.lower().replace(',', '').split()
    seq = [tok2idx.get(w, tok2idx['UNK']) for w in toks]
    pad = pad_sequences([seq], maxlen=MAXLEN, padding='post', value=tok2idx['PAD'])
    preds = model.predict(pad)[0]
    labs = le.inverse_transform(preds.argmax(-1))[:len(toks)]

    items = []
    cur_item = {'quantity': None, 'unit': None, 'ingredient': None}
    ent_tokens, ent_type = [], None

    for w, tag in zip(toks, labs):
        if tag != 'O':
            typ = tag.split('-',1)[1].lower()
            if ent_type != typ:
                if ent_tokens and ent_type:
                    cur_item[ent_type] = " ".join(ent_tokens)
                    ent_tokens = []
                ent_type = typ
                ent_tokens = [w]
            else:
                ent_tokens.append(w)
        else:
            if ent_tokens and ent_type:
                cur_item[ent_type] = " ".join(ent_tokens)
                ent_tokens = []
                ent_type = None
        if all(cur_item.values()):
            items.append(cur_item.copy())
            cur_item = {'quantity': None, 'unit': None, 'ingredient': None}

    if ent_tokens and ent_type:
        cur_item[ent_type] = " ".join(ent_tokens)
    if all(cur_item.values()):
        items.append(cur_item)

    return items

# — Fungsi rekomendasi resep berdasarkan bahan parsed
def recommend_recipes(text, top_n=5, fuzzy_threshold=75):
    items = parse_ingredients(text)
    pprint(items)
    if not items:
        return pd.DataFrame(columns=['title_cleaned', 'instructions_cleaned', 'cleaned_ingredients', 'total_recipe_carbon', 'title_display'])

    title_sets = []
    for it in items:
        qty_num = parse_number(it['quantity'])
        unit    = (it['unit'] or '').lower()
        ing     = (it['ingredient'] or '').lower()

        suggestions = process.extract(
            ing,
            PURE_NAMES,
            scorer=fuzz.token_set_ratio,
            limit=3
        )
        good = [(name, score) for name, score, _ in suggestions if score >= fuzzy_threshold]
        if good:
            print(f">>> Input '{ing}' matched to:")
            for name, score in good:
                print(f"    • {name} (score {score}%)")
        else:
            print(f">>> No good fuzzy match for '{ing}' (top was {suggestions[0][1]}%)")

        mask_name = df_exploded['pure_name'].apply(
            lambda x: fuzz.token_set_ratio(str(x).lower(), ing) >= fuzzy_threshold
        )

        factor_in = UNIT_FACTORS.get(unit)
        factors   = df_exploded['unit'].map(UNIT_FACTORS)
        if factor_in and qty_num is not None:
            avail     = qty_num * factor_in
            mask_conv = (
                mask_name &
                factors.notna() &
                ((df_exploded['quantity'] * factors) <= avail)
            )
        else:
            mask_conv = pd.Series(False, index=df_exploded.index)

        if qty_num is not None:
            mask_fb = (
                mask_name &
                (df_exploded['unit'].str.lower() == unit) &
                (df_exploded['quantity'] <= qty_num)
            )
        else:
            mask_fb = pd.Series(False, index=df_exploded.index)

        mask   = mask_conv | mask_fb
        titles = set(df_exploded.loc[mask, 'title_cleaned'])
        if titles:
            title_sets.append(titles)

    if not title_sets:
        return pd.DataFrame(columns=['title_cleaned', 'instructions_cleaned', 'cleaned_ingredients', 'total_recipe_carbon', 'title_display'])

    common = set.intersection(*title_sets)
    if not common:
        return pd.DataFrame(columns=['title_cleaned', 'instructions_cleaned', 'cleaned_ingredients', 'total_recipe_carbon', 'title_display'])

    df_meta = (
        df_exploded
        .drop_duplicates(subset=['title_cleaned'])
        .loc[:, ['title_cleaned', 'instructions_cleaned', 'cleaned_ingredients', 'total_recipe_carbon']]
    )

    # Tambahkan kolom tampilan dengan info CO2
    df_meta['title_display'] = df_meta.apply(
        lambda row: f"{row['title_cleaned']} ({row['total_recipe_carbon']} CO2eq/kg)"
        if 'total_recipe_carbon' in row and not pd.isnull(row['total_recipe_carbon'])
        else row['title_cleaned'],
        axis=1
    )

    return df_meta[df_meta['title_cleaned'].isin(common)].head(top_n)

# — Main program: input dan cetak hasil rekomendasi
def main():
    # 1) Input & parsing bahan user
    text = input("Masukkan bahan makanan: ")
    input_items = parse_ingredients(text)

    # 3) Hitung total karbon dari bahan yang diinput
    total_cf = calculate_total_carbon_from_items(input_items, UNIT_FACTORS)
    print(f"\nTotal Jejak Karbon dari Bahan: {total_cf} kg CO2")

    # 2) Rekomendasi resep
    res = recommend_recipes(text, top_n=5)
    split_pattern = re.compile(r'(?<!\b\d)(?<!tsp)(?<!tbsp)\.\s+')

    if res.empty:
        print("Maaf, tidak ada resep yang cocok.")
        return

    # 3) Tampilkan daftar singkat resep
    print("\nResep yang cocok ditemukan:")
    indexed_res = res.reset_index(drop=True)
    for i, row in indexed_res.iterrows():
        print(f"{i+1}. {row.get('title_display', row['title_cleaned'])}")

    # 4) User pilih resep
    while True:
        try:
            pilihan = int(input("\nMasukkan nomor resep yang ingin dilihat: "))
            if 1 <= pilihan <= len(indexed_res):
                break
            print("Nomor tidak valid. Coba lagi.")
        except ValueError:
            print("Input harus berupa angka.")
    row = indexed_res.iloc[pilihan-1]
    selected_title = row['title_cleaned']

    # 5) Cetak detail resep
    print(f"\n=== {selected_title} ===\n")
    print("Instructions:")
    instr = row['instructions_cleaned'].strip()
    for part in split_pattern.split(instr):
        sent = part.strip().rstrip('.')
        if not sent:
            continue
        wrapped = textwrap.fill(sent, width=80)
        for j, line in enumerate(wrapped.split('\n')):
            prefix = "- " if j == 0 else "  "
            print(f"  {prefix}{line}")

    print("\nIngredients:")
    for ing in row['cleaned_ingredients']:
        print(f"  - {ing}")

    # 6) Hitung stok user (base unit)
    input_amounts = {}
    for it in input_items:
        qty = parse_number(it['quantity'])
        unit = (it['unit'] or '').lower()
        if qty is None or unit not in UNIT_FACTORS:
            continue
        input_amounts.setdefault(it['ingredient'], 0.0)
        input_amounts[it['ingredient']] += qty * UNIT_FACTORS[unit]

    # 7) Ambil kebutuhan resep dari df_exploded
    df_sel = df_exploded[df_exploded['title_cleaned'] == selected_title].copy()
    df_sel['factor'] = df_sel['unit'].str.lower().map(UNIT_FACTORS)
    df_sel = df_sel[df_sel['factor'].notna()]
    df_sel['required_base'] = df_sel['quantity'] * df_sel['factor']

    # 8) Siapkan used_amounts per pure_name
    used_amounts = df_sel.groupby('pure_name')['required_base'].sum().to_dict()

    # 9) Hitung & tampilkan bahan tersisa
    leftovers = []
    for inp_name, avail_base in input_amounts.items():
        # cari match terbaik di used_amounts
        match, score, _ = process.extractOne(inp_name,
                                             list(used_amounts),
                                             scorer=fuzz.token_set_ratio)
        required = used_amounts[match] if score >= 60 else 0.0
        sisa_base = avail_base - required
        if sisa_base > 0:
            orig_unit = next((it['unit'] for it in input_items if it['ingredient'] == inp_name), None)
            if orig_unit and orig_unit.lower() in UNIT_FACTORS:
                sisa_qty = sisa_base / UNIT_FACTORS[orig_unit.lower()]
                leftovers.append(f"{sisa_qty:.2f} {orig_unit} {inp_name}")
            else:
                leftovers.append(f"{sisa_base:.2f} base-unit {inp_name}")

    if leftovers:
        print("\nBahan kamu masih tersisa:")
        for item in leftovers:
            print(f"  - {item}")

    # 10) Hitung & tampilkan bahan yang kurang
    df_group = (
        df_sel
        .groupby('pure_name')
        .agg({
            'required_base': 'sum',
            'unit':         lambda s: s.iloc[0],
            'factor':       lambda s: s.iloc[0]
        })
        .reset_index()
    )
    missing = []
    for _, r in df_group.iterrows():
        name     = r['pure_name']
        req_base = r['required_base']
        unit     = r['unit']
        factor   = r['factor']

        # cari berapa yang user punya (fuzzy-match)
        match, score, _ = process.extractOne(name,
                                             list(input_amounts),
                                             scorer=fuzz.token_set_ratio)
        avail = input_amounts[match] if score >= 60 else 0.0

        if avail < req_base:
            short_base = req_base - avail
            short_qty  = short_base / factor
            missing.append(f"{short_qty:.2f} {unit} {name}")

    if missing:
        print("\nBahan yang kamu kurang:")
        for item in missing:
            print(f"  - {item}")

    print("\n" + "="*40)


if __name__ == "__main__":
    main()
