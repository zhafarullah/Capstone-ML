# Capstone
## Struktur Folder

├── main.py                     # Entry point program
├── utils/
│   └── carbon_calculator.py   # Modul perhitungan karbon per bahan
├── best_ner_bilstm.h5         # Model NER yang sudah dilatih
├── tok2idx.pkl                # Kamus token ke indeks
├── label_encoder.pkl          # Label encoder untuk NER
├── requirements.txt           # Daftar dependensi
└── README.md                  # Dokumen ini

## Cara Menjalankan
1. Instalasi depedensi
   ```pip install -r requirements.txt```
2. Pastikan MOngoDB terhubung
   ```MONGO_URI = "mongodb+srv://anzzanafa:fWZJzU2FGfWlobHY@cluster0.1xeasvn.mongodb.net/ecorecipes?retryWrites=true&w=majority&appName=Cluster0"```
3. Jalankan program
   ```python main.py```
4. Masukkan bahan makanan secara natural
   ```2 cups of rice, 1 tablespoon of olive oil, 200g chicken breast```
5. Lihat total jejak karbon dan rekomendasi resep
6. Teknologi yang Digunakan
   a. Python 3.10+
   b. TensorFlow / Keras - Model BiLSTM-CRF untuk NER
   c. MongoDB Atlas - Penyimpanan data resep
   d. RapidFuzz - Fuzzy matching bahan makanan
   e. gdown - (opsional) untuk mengunduh dataset besar dari Google Drive
   f. Pickle - Menyimpan kamus dan encoder
7. 
