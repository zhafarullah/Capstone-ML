# inference_carbon.py

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from rapidfuzz import process, fuzz
import joblib
import nltk

nltk.download('punkt')

# ── Load CSV utama ──────────────────────────────────────────────────────────
df = pd.read_csv("all_carbon.csv")

# ── Load model, scaler, dan KMeans ──────────────────────────────────────────
w2v_model = Word2Vec.load("my_fooditem_word2vec.model")
scaler = joblib.load("my_scaler.pkl")
km = joblib.load("my_kmeans.pkl")

# ── Siapkan list & default carbon ───────────────────────────────────────────
item_list = df['food_item'].dropna().unique().tolist()
default_cf = df['carbon_footprint'].median()

# ── Fungsi ubah teks ke vektor Word2Vec ─────────────────────────────────────
def get_sentence_vector(text, model, dim=100):
    tokens = word_tokenize(text.lower())
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if len(vectors) == 0:
        return np.zeros(dim)
    return np.mean(vectors, axis=0)

# ── Fungsi estimasi karbon ──────────────────────────────────────────────────
def estimate_carbon_from_item(food_item, fuzz_thresh=90):
    match = process.extractOne(food_item, item_list, scorer=fuzz.token_set_ratio)

    if match and match[1] >= fuzz_thresh:
        matched_name, score, _ = match
        cf_val = df.loc[df['food_item'] == matched_name, 'carbon_footprint'].mean()
        return {
            'carbon_footprint': cf_val,
            'method': 'fuzzy_match',
            'matched_to': matched_name,
            'score': score
        }

    # Embedding
    emb_vector = get_sentence_vector(food_item, w2v_model, dim=w2v_model.vector_size).reshape(1, -1)

    # **Perbaikan di sini**:
    num_df = pd.DataFrame({'carbon_footprint': [default_cf]})
    num_vector = scaler.transform(num_df)

    x_new = np.hstack([emb_vector, num_vector])

    cl_id = int(km.predict(x_new)[0])
    cf_val = df.loc[df['cluster_id'] == cl_id, 'carbon_footprint'].mean()
    return {
        'carbon_footprint': cf_val,
        'method': 'cluster_average',
        'cluster_id': cl_id
    }

# ── Contoh Penggunaan ───────────────────────────────────────────────────────
if __name__ == "__main__":
    test_items = ["chicken", "quinoa", "salmon fillet", "dragonfruit"]
    for item in test_items:
        result = estimate_carbon_from_item(item, fuzz_thresh=85)
        print(f"{item} → {result}")
