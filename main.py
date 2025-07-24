import os
import json
import pickle
import gzip
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
import argparse

# === Load model EfficientNet ===
print("🔄 Loading model...")
model = load_model("efficientnet_model.h5")
print("✅ Model loaded.")

# === Load fitur dan metadata yang sudah dioptimasi ===
print("🔄 Loading optimized features and filenames...")
try:
    # Coba load file compressed terlebih dahulu
    with gzip.open("features_cloudinary_compressed.pkl.gz", "rb") as f:
        data = pickle.load(f)
    
    features = data['features']
    filenames = data['filenames']
    pca_model = data['pca_model']
    n_components = data['n_components']
    explained_variance = data['explained_variance_ratio']
    
    print(f"✅ Loaded {len(filenames)} image features (compressed version).")
    print(f"📊 Feature dimensions: {features.shape}")
    print(f"📈 PCA variance retained: {explained_variance*100:.1f}%")
    
except FileNotFoundError:
    # Fallback ke file optimized tanpa kompresi
    try:
        with open("features_cloudinary_optimized.pkl", "rb") as f:
            data = pickle.load(f)
        
        features = data['features']
        filenames = data['filenames']
        pca_model = data['pca_model']
        n_components = data['n_components']
        explained_variance = data['explained_variance_ratio']
        
        print(f"✅ Loaded {len(filenames)} image features (optimized version).")
        print(f"📊 Feature dimensions: {features.shape}")
        print(f"📈 PCA variance retained: {explained_variance*100:.1f}%")
        
    except FileNotFoundError:
        # Fallback ke file original
        print("⚠️ Optimized files not found, using original file...")
        with open("features_cloudinary.pkl", "rb") as f:
            features, filenames = pickle.load(f)
        pca_model = None
        print(f"✅ Loaded {len(filenames)} image features (original version).")

# Load mapping filename ke URL
with open("filename_to_url.json", "r") as f:
    filename_to_url = json.load(f)

# === Fungsi Ekstraksi Fitur dengan Optimasi ===
def extract_feature_from_image(img_path):
    """
    Ekstrak fitur dari gambar dan aplikasikan transformasi yang sama
    seperti saat training (PCA jika tersedia)
    """
    # Ekstraksi fitur biasa
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    feature = model.predict(img_array, verbose=0)
    feature = feature.flatten()
    
    # Aplikasikan PCA jika tersedia
    if pca_model is not None:
        feature = pca_model.transform([feature])[0]
        # Konversi ke float16 untuk konsistensi
        feature = feature.astype(np.float16)
    
    return feature

# === CLI Argument Parser ===
parser = argparse.ArgumentParser(description="Find similar images using EfficientNet + cosine similarity.")
parser.add_argument("image_path", type=str, help="Path to the image file.")
parser.add_argument("--top_n", type=int, default=5, help="Number of top similar images to return.")

args = parser.parse_args()

if not os.path.exists(args.image_path):
    print("❌ Image file not found.")
    exit(1)

# === Eksekusi Pencarian Gambar Mirip ===
print(f"🔍 Extracting features from: {args.image_path}")
query_feature = extract_feature_from_image(args.image_path).reshape(1, -1)

print("🔄 Computing similarities...")
similarities = cosine_similarity(query_feature, features)[0]
top_idx = np.argsort(similarities)[::-1][:args.top_n]

print(f"\n🔗 Top {args.top_n} most similar images:\n")
for i, idx in enumerate(top_idx, 1):
    fname = os.path.splitext(os.path.basename(filenames[idx]))[0]
    similarity = round(float(similarities[idx]), 4)  # Tambah precision untuk float16
    cloud_url = filename_to_url.get(f"{fname}.jpg", "URL_NOT_FOUND")
    
    print(f"{i}. {fname}")
    print(f"   Similarity: {similarity}")
    print(f"   URL: {cloud_url}")
    print()

# === Info tambahan jika menggunakan PCA ===
if pca_model is not None:
    print(f"ℹ️  Using PCA-optimized features ({n_components} components, {explained_variance*100:.1f}% variance retained)")
else:
    print("ℹ️  Using original features (no PCA optimization)")