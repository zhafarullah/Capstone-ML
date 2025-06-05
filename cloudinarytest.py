import os
import cloudinary
import cloudinary.uploader
import json

# === 1. Konfigurasi Cloudinary ===
cloudinary.config(
    cloud_name="drcz82fa2",
    api_key="747541653778546",
    api_secret="0atiXyH54uYPGqW59d9HX1U5S90",
    secure=True
)

# === 2. Lokasi Folder Gambar ===
image_folder = "images/"  # ganti dengan foldermu
filename_to_url = {}

# === 3. Upload Gambar Satu per Satu ===
for filename in os.listdir(image_folder):
    file_path = os.path.join(image_folder, filename)

    # Pastikan hanya file gambar yang diproses
    if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        continue

    try:
        # Buat nama publik (tanpa ekstensi)
        public_id = os.path.splitext(filename)[0]

        # Upload ke Cloudinary
        result = cloudinary.uploader.upload(
            file_path,
            public_id=public_id,
            overwrite=True,
            resource_type="image"
        )

        # Simpan hasil mapping
        filename_to_url[filename] = result["secure_url"]
        print(f"Uploaded {filename} → {result['secure_url']}")

    except Exception as e:
        print(f"❌ Gagal upload {filename}: {e}")

# === 4. Simpan ke file JSON ===
with open("filename_to_url.json", "w") as f:
    json.dump(filename_to_url, f, indent=2)

print("\n✅ Semua gambar berhasil diupload dan mapping disimpan di filename_to_url.json.")
