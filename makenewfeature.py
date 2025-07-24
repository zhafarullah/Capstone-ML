import os
import json
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.decomposition import PCA
from tqdm import tqdm
import gzip

print("🚀 Memulai ekstraksi dan optimasi fitur...")

# === Load model EfficientNet ===
print("📥 Loading model EfficientNet...")
model = load_model("efficientnet_model.h5")

# === Load mapping nama file → URL Cloudinary ===
print("📥 Loading mapping filename ke URL...")
with open("filename_to_url.json", "r") as f:
    filename_to_url = json.load(f)

# === Konfigurasi folder gambar lokal ===
image_folder = "images/"
features = []
filenames = []

print(f"🔄 Memproses gambar dari folder: {image_folder}")

# === Loop semua file gambar dan ekstrak fitur ===
for filename in tqdm(sorted(os.listdir(image_folder)), desc="Ekstraksi fitur"):
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
        feature = model.predict(img_array, verbose=0).flatten()  # verbose=0 untuk menghilangkan output predict

        # Tambahkan fitur dan filename (Cloudinary)
        if filename in filename_to_url:
            features.append(feature)
            filenames.append(filename)
        else:
            print(f"⚠️ File {filename} tidak ada dalam mapping Cloudinary — dilewati.")

    except Exception as e:
        print(f"❌ Gagal proses {filename}: {e}")

print(f"\n✅ Ekstraksi selesai. Total: {len(filenames)} gambar.")

# === Konversi ke numpy array ===
print("🔄 Konversi ke numpy array...")
features_array = np.array(features)
print(f"📊 Dimensi fitur asli: {features_array.shape}")

# === LANGKAH OPTIMASI DIMULAI ===
print("\n🎯 MEMULAI OPTIMASI UKURAN FILE...")

# === OPTIMASI 1: Simpan file original dulu (untuk perbandingan) ===
print("💾 Menyimpan file original...")
with open("features_cloudinary.pkl", "wb") as f:
    pickle.dump((features_array, filenames), f)

original_size = os.path.getsize("features_cloudinary.pkl") / (1024*1024)  # MB
print(f"📁 Ukuran file original: {original_size:.2f} MB")

# === OPTIMASI 2: Reduksi Dimensi dengan PCA ===
print("🔄 Melakukan reduksi dimensi dengan PCA...")

# Tentukan jumlah komponen PCA berdasarkan jumlah gambar
if len(filenames) > 5000:
    n_components = 256  # Untuk dataset besar
elif len(filenames) > 1000:
    n_components = 512  # Untuk dataset medium
else:
    n_components = min(512, len(filenames) - 1)  # Untuk dataset kecil

print(f"🎯 Menggunakan {n_components} komponen PCA...")

pca = PCA(n_components=n_components)
features_reduced = pca.fit_transform(features_array)

print(f"📊 Dimensi setelah PCA: {features_reduced.shape}")
print(f"📈 Variance yang dipertahankan: {pca.explained_variance_ratio_.sum():.4f} ({pca.explained_variance_ratio_.sum()*100:.1f}%)")

# === OPTIMASI 3: Konversi ke float16 ===
print("🔄 Konversi ke float16...")
features_reduced = features_reduced.astype(np.float16)

# === OPTIMASI 4: Persiapan data untuk disimpan ===
data_to_save = {
    'features': features_reduced,
    'filenames': filenames,
    'pca_model': pca,
    'n_components': n_components,
    'explained_variance_ratio': pca.explained_variance_ratio_.sum(),
    'original_shape': features_array.shape
}

# === OPTIMASI 5: Simpan versi optimized tanpa kompresi ===
print("💾 Menyimpan versi optimized...")
with open("features_cloudinary_optimized.pkl", "wb") as f:
    pickle.dump(data_to_save, f)

optimized_size = os.path.getsize("features_cloudinary_optimized.pkl") / (1024*1024)  # MB

# === OPTIMASI 6: Simpan dengan kompresi GZIP ===
print("💾 Menyimpan versi compressed...")
with gzip.open("features_cloudinary_compressed.pkl.gz", "wb") as f:
    pickle.dump(data_to_save, f)

compressed_size = os.path.getsize("features_cloudinary_compressed.pkl.gz") / (1024*1024)  # MB

# === HASIL OPTIMASI ===
print("\n" + "="*60)
print("🎉 HASIL OPTIMASI")
print("="*60)
print(f"📊 Total gambar diproses: {len(filenames):,}")
print(f"📊 Dimensi asli: {features_array.shape}")
print(f"📊 Dimensi setelah PCA: {features_reduced.shape}")
print(f"📊 Variance dipertahankan: {pca.explained_variance_ratio_.sum()*100:.1f}%")
print()
print("📁 UKURAN FILE:")
print(f"   Original: {original_size:.2f} MB")
print(f"   Optimized: {optimized_size:.2f} MB")
print(f"   Compressed: {compressed_size:.2f} MB")
print()
print("📉 PENGURANGAN UKURAN:")
print(f"   PCA + float16: {((original_size - optimized_size) / original_size * 100):.1f}%")
print(f"   + Kompresi: {((original_size - compressed_size) / original_size * 100):.1f}%")
print()
print("✅ FILE YANG DIHASILKAN:")
print("   📄 features_cloudinary.pkl (original)")
print("   📄 features_cloudinary_optimized.pkl (PCA + float16)")
print("   📄 features_cloudinary_compressed.pkl.gz (FINAL - gunakan ini!)")
print()
print("🚀 Gunakan features_cloudinary_compressed.pkl.gz untuk Railway!")
print("="*60)