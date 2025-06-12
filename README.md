# Struktur File
1. app.py: Aplikasi FastAPI untuk pencarian gambar mirip menggunakan fitur EfficientNet, terhubung ke MongoDB dan Cloudinary.
2. main.py: Skrip command-line untuk mencari gambar mirip menggunakan model EfficientNet dan fitur yang sudah diekstrak.
3. cloudinarytest.py: Skrip untuk mengupload gambar lokal ke Cloudinary dan menyimpan pemetaan nama file ke URL.
4. makenewfeature.py: Skrip untuk mengekstrak fitur gambar dari folder lokal menggunakan EfficientNet dan menyimpannya untuk pencarian kemiripan.
5. efficientnet_model.h5: File base model EfficientNet digunakan untuk ekstraksi fitur gambar.
6. features_cloudinary.pkl: File pickle yang berisi fitur gambar yang sudah diekstrak dan daftar nama file.
7. filename_to_url.json: File JSON yang memetakan nama file gambar ke URL Cloudinary.
8. requirements.txt: Daftar dependensi Python yang dibutuhkan proyek.

# Cara Menjalankan 
1. python -m venv .venv
2. .venv\Scripts\activate
3. pip install -r requirements.txt
4. jalankan "python app.py" atau jika ingin versi lokalnya bisa menjalankan "python main.py path/to/test_image.jpg --top_n 5"

# Matriks Evaluasi (Heatmap)
![Heatmap Cosine Similarity](img/heatmap.png)
Disini Saya menggunakan heatmap cosine similarity karena proyek ini tidak melibatkan klasifikasi gambar, melainkan pengukuran kemiripan antar gambar berdasarkan fitur visual dari model EfficientNet. Karena tidak ada label kategori, saya tidak bisa menggunakan confusion matrix yang membutuhkan label asli dan prediksi. Heatmap ini menampilkan seberapa mirip satu gambar dengan yang lain, dengan nilai mendekati 1 menunjukkan kemiripan tinggi. Dari hasilnya, saya bisa melihat bahwa gambar-gambar dengan jenis atau visual serupa memang memiliki similarity yang cukup tinggi, sehingga evaluasi ini membantu saya menilai kualitas fitur tanpa perlu label.


