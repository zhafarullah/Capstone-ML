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

# — Load machine learning artifacts
model = load_model('best_ner_bilstm.h5')
tok2idx = pickle.load(open('tok2idx.pkl', 'rb'))
le = pickle.load(open('label_encoder.pkl', 'rb'))

# — Constants
MAXLEN = 50

df_exploded = pd.DataFrame(list(collection.find({})))
if 'cleaned_ingredients' in df_exploded.columns:
    df_exploded['cleaned_ingredients'] = df_exploded['cleaned_ingredients'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

df_exploded['pure_name'] = df_exploded['pure_name'].fillna('').astype(str)
PURE_NAMES = df_exploded['pure_name'].unique().tolist()

UNIT_FACTORS = {
    'mg': 0.001, 'milligram': 0.001, 'milligrams': 0.001,
    'g': 1, 'gram': 1, 'grams': 1,
    'kg': 1000, 'kilo': 1000, 'kilogram': 1000, 'kilograms': 1000,
    'oz': 28.3495, 'ounce': 28.3495, 'ounces': 28.3495,
    'lb': 453.592, 'lbs': 453.592, 'pound': 453.592, 'pounds': 453.592,
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

# Helper to convert textual numbers to float

def parse_number(q):
    q = q.strip().lower()
    if '/' in q:
        try:
            return float(Fraction(q))
        except:
            pass
    WORD2NUM = {'half':0.5, 'one':1, 'two':2, 'three':3, 'four':4, 'five':5,
                'six':6, 'seven':7, 'eight':8}
    if q in WORD2NUM:
        return WORD2NUM[q]
    try:
        return float(q)
    except:
        return None

# NER parsing

def parse_ingredients(text):
    toks = text.lower().replace(',', '').split()
    seq = [tok2idx.get(w, tok2idx['UNK']) for w in toks]
    pad = pad_sequences([seq], maxlen=MAXLEN, padding='post', value=tok2idx['PAD'])
    preds = model.predict(pad)[0]
    labs = le.inverse_transform(preds.argmax(-1))[:len(toks)]

    items = []
    cur = {'quantity':None, 'unit':None, 'ingredient':None}
    tokens, typ = [], None
    for w, tag in zip(toks, labs):
        if tag!='O':
            t = tag.split('-',1)[1].lower()
            if typ!=t:
                if tokens and typ:
                    cur[typ] = ' '.join(tokens)
                    tokens=[]
                typ = t
                tokens=[w]
            else:
                tokens.append(w)
        else:
            if tokens and typ:
                cur[typ]=' '.join(tokens)
                tokens=[]; typ=None
        if all(cur.values()):
            items.append(cur.copy())
            cur={'quantity':None,'unit':None,'ingredient':None}
    if tokens and typ:
        cur[typ]=' '.join(tokens)
    if all(cur.values()): items.append(cur)
    return items

# Recipe recommendation (fuzzy + match)

def recommend_recipes(text, top_n=5, fuzzy_threshold=75):
    items = parse_ingredients(text)
    pprint(items)
    if not items:
        return pd.DataFrame(columns=['title_cleaned','instructions_cleaned',
                                     'cleaned_ingredients','total_recipe_carbon','title_display'])
    sets=[]
    for it in items:
        qty = parse_number(it['quantity'])
        unit= (it['unit'] or '').lower()
        ing = (it['ingredient'] or '').lower()
        mask_name = df_exploded['pure_name'].apply(
            lambda x: fuzz.token_set_ratio(x.lower(), ing)>=fuzzy_threshold)
        factors = df_exploded['unit'].map(UNIT_FACTORS)
        factor_in=UNIT_FACTORS.get(unit)
        if factor_in and qty is not None:
            avail = qty*factor_in
            mask = mask_name & factors.notna() & ((df_exploded['quantity']*factors)<=avail)
        else:
            mask = mask_name & (df_exploded['unit'].str.lower()==unit)
        titles=set(df_exploded.loc[mask,'title_cleaned'])
        if titles: sets.append(titles)
    if not sets: return pd.DataFrame()
    common=set.intersection(*sets)
    df_meta = df_exploded.drop_duplicates('title_cleaned')[['title_cleaned',
                                                           'instructions_cleaned',
                                                           'cleaned_ingredients',
                                                           'total_recipe_carbon']]
    df_meta['title_display']=df_meta.apply(
        lambda r: f"{r['title_cleaned']} ({r['total_recipe_carbon']} CO2eq/kg)", axis=1)
    return df_meta[df_meta['title_cleaned'].isin(common)].head(top_n)

# Main flow
def main():
    text = input("Masukkan bahan makanan: ")
    input_items = parse_ingredients(text)

    # Total karbon dari input user
    total_cf = calculate_total_carbon_from_items(input_items, UNIT_FACTORS)
    print(f"\nTotal Jejak Karbon dari Bahan: {total_cf:.2f} kg CO2")

    # Rekomendasi resep
    res = recommend_recipes(text)
    if res.empty:
        print("Maaf, tidak ada resep yang cocok.")
        return
    print("\nResep yang cocok ditemukan:")
    idx = res.reset_index(drop=True)
    for i, r in idx.iterrows(): print(f"{i+1}. {r['title_display']}")

    # Pilih resep
    while True:
        try:
            p=int(input("\nMasukkan nomor resep: "))
            if 1<=p<=len(idx): break
        except: pass
    sel=idx.iloc[p-1]['title_cleaned']

    # Cetak detail
    print(f"\n=== {sel} ===\nInstructions:")
    split_p = re.compile(r'(?<!\b\d)(?<!tsp)(?<!tbsp)\.\s+')
    inst=idx.iloc[p-1]['instructions_cleaned']
    for part in split_p.split(inst):
        s=part.strip().rstrip('.')
        if not s: continue
        wrap=textwrap.fill(s, width=80)
        for j,ln in enumerate(wrap.split('\n')):
            pref = "- " if j==0 else "  "
            print(f"  {pref}{ln}")

    print("\nIngredients:")
    for ing in idx.iloc[p-1]['cleaned_ingredients']:
        print(f"  - {ing}")

    # Stok user ke base unit
    input_amounts = {}
    for it in input_items:
        qty=parse_number(it['quantity'])
        unit=(it['unit'] or '').lower()
        if qty is None or unit not in UNIT_FACTORS: continue
        input_amounts.setdefault(it['ingredient'],0.0)
        input_amounts[it['ingredient']]+=qty*UNIT_FACTORS[unit]

    # Kebutuhan resep
    df_sel=df_exploded[df_exploded['title_cleaned']==sel].copy()
    df_sel['factor']=df_sel['unit'].str.lower().map(UNIT_FACTORS)
    df_sel['required_base']=df_sel['quantity']*df_sel['factor']
    used_amounts = df_sel.groupby('pure_name')['required_base'].sum().to_dict()

    # Bahan terpakai & total karbon
    used_items=[]
    used_dicts=[]
    for name, avail in input_amounts.items():
        match,score,_=process.extractOne(name, list(used_amounts), scorer=fuzz.token_set_ratio)
        req=used_amounts.get(match,0.0) if score>=60 else 0.0
        used_base=min(avail, req)
        if used_base>0:
            # unit asli
            orig_unit=next((it['unit'] for it in input_items if it['ingredient']==name),None)
            if orig_unit and orig_unit.lower() in UNIT_FACTORS:
                used_qty = used_base/UNIT_FACTORS[orig_unit.lower()]
                used_items.append(f"{used_qty:.2f} {orig_unit} {name}")
                used_dicts.append({'quantity':str(used_qty),'unit':orig_unit,'ingredient':name})
            else:
                used_items.append(f"{used_base:.2f} base-unit {name}")
    if used_items:
        print("\nBahan kamu yang terpakai:")
        for it in used_items: print(f"  - {it}")
        total_used = calculate_total_carbon_from_items(used_dicts, UNIT_FACTORS)
        print(f"Total Karbon dari Bahan Terpakai: {total_used:.2f} kg CO₂")

    # Bahan yang kurang & total karbon
    df_group = df_sel.groupby('pure_name').agg({'required_base':'sum','unit':lambda s:s.iloc[0],'factor':lambda s:s.iloc[0]}).reset_index()
    missing=[]
    miss_dicts=[]
    for _, r in df_group.iterrows():
        nm=r['pure_name']; req=r['required_base']; unit=r['unit']; fac=r['factor']
        match,score,_=process.extractOne(nm,list(input_amounts),scorer=fuzz.token_set_ratio)
        avail=input_amounts[match] if score>=60 else 0.0
        if avail<req:
            short_base=req-avail
            short_qty=short_base/fac
            missing.append(f"{short_qty:.2f} {unit} {nm}")
            miss_dicts.append({'quantity':str(short_qty),'unit':unit,'ingredient':nm})
    if missing:
        print("\nBahan yang kamu kurang:")
        for it in missing: print(f"  - {it}")
        total_miss = calculate_total_carbon_from_items(miss_dicts, UNIT_FACTORS)
        print(f"Total Karbon dari Bahan yang Kurang: {total_miss:.2f} kg CO₂")
        
        total_recipe = total_used + total_miss
        if total_recipe > 0:
            efficiency = total_used / total_recipe
        else:
            efficiency = 0.0
        print(f"Efisiensi Pemanfaatan Karbon: {efficiency*100:.1f}%")

        print("\n" + "="*40)

if __name__ == '__main__':
    main()
    ## redeploy
