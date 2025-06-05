import os
import json
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from tqdm import tqdm

# === Load model EfficientNet ===
model = load_model("efficientnet_model.h5")

# === Load mapping nama file → URL Cloudinary ===
with open("filename_to_url.json", "r") as f:
    filename_to_url = json.load(f)

# === Konfigurasi folder gambar lokal ===
image_folder = "images/"
features = []
filenames = []

# === Loop semua file gambar dan ekstrak fitur ===
for filename in tqdm(sorted(os.listdir(image_folder))):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    filepath = os.path.join(image_folder, filename)

    try:
        # Load dan pre-process gambar
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Prediksi fitur dengan model
        feature = model.predict(img_array).flatten()

        # Tambahkan fitur dan filename (Cloudinary)
        if filename in filename_to_url:
            features.append(feature)
            filenames.append(filename)  # Hanya filename (bukan path)

        else:
            print(f"⚠️ File {filename} tidak ada dalam mapping Cloudinary — dilewati.")

    except Exception as e:
        print(f"❌ Gagal proses {filename}: {e}")

# === Simpan hasil fitur dan urutan filename ===
with open("features_cloudinary.pkl", "wb") as f:
    pickle.dump((np.array(features), filenames), f)

print(f"\n✅ Ekstraksi selesai. Total: {len(filenames)} gambar.")
print("✅ Disimpan sebagai features_cloudinary.pkl")
