import os
import json
import pickle
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

# === Load fitur dan metadata ===
print("🔄 Loading features and filenames...")
with open("features_cloudinary.pkl", "rb") as f:
    features, filenames = pickle.load(f)
print(f"✅ Loaded {len(filenames)} image features.")

with open("filename_to_url.json", "r") as f:
    filename_to_url = json.load(f)

# === Fungsi Ekstraksi Fitur ===
def extract_feature_from_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    feature = model.predict(img_array)
    return feature.flatten()

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
similarities = cosine_similarity(query_feature, features)[0]
top_idx = np.argsort(similarities)[::-1][:args.top_n]

print(f"\n🔗 Top {args.top_n} most similar images:\n")
for idx in top_idx:
    fname = os.path.splitext(os.path.basename(filenames[idx]))[0]
    similarity = round(float(similarities[idx]), 3)
    cloud_url = filename_to_url.get(f"{fname}.jpg", "URL_NOT_FOUND")
    print(f"- {fname} (similarity: {similarity})")
    print(f"  URL: {cloud_url}\n")
