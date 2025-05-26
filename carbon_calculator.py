# carbon_calculator.py

import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from rapidfuzz import process, fuzz
import joblib
import nltk
from fractions import Fraction

nltk.download("punkt")

# ── Load models & data ───────────────────────────────────────────────────────
df = pd.read_csv("all_carbon.csv")
w2v_model = Word2Vec.load("my_fooditem_word2vec.model")
scaler = joblib.load("my_scaler.pkl")
km = joblib.load("my_kmeans.pkl")
item_list = df["food_item"].dropna().unique().tolist()
default_cf = df["carbon_footprint"].median()

# ── Helper untuk parsing angka dari string ──────────────────────────────────
def parse_number(q):
    if q is None:
        return None
    q = str(q).strip().lower()
    # handle fractions like "1/2"
    if "/" in q:
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

# ── Embedding & estimasi karbon per item ─────────────────────────────────────
def get_sentence_vector(text, model, dim=100):
    tokens = word_tokenize(text.lower())
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(dim)

def estimate_carbon_from_item(food_item, fuzz_thresh=90):
    match = process.extractOne(food_item, item_list, scorer=fuzz.token_set_ratio)
    if match and match[1] >= fuzz_thresh:
        matched_name, score, _ = match
        return df.loc[df["food_item"] == matched_name, "carbon_footprint"].mean()
    emb = get_sentence_vector(food_item, w2v_model).reshape(1, -1)
    num_vec = scaler.transform(pd.DataFrame({"carbon_footprint": [default_cf]}))
    x_new = np.hstack([emb, num_vec])
    cluster = int(km.predict(x_new)[0])
    return df.loc[df["cluster_id"] == cluster, "carbon_footprint"].mean()

# ── Hitung total karbon dari list items ─────────────────────────────────────
def calculate_total_carbon_from_items(items, unit_factors):
    total = 0.0
    for item in items:
        qty_raw = item.get("quantity")
        unit = (item.get("unit") or "").lower()
        name = (item.get("ingredient") or "").lower()

        # konversi qty ke float
        qty = parse_number(qty_raw)
        if qty is None or unit not in unit_factors:
            continue

        # ubah ke kilogram
        qty_kg = qty * unit_factors[unit] / 1000
        cf_per_unit = estimate_carbon_from_item(name)
        total += qty_kg * cf_per_unit

    return round(total, 4)
