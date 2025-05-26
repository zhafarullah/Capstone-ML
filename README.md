# Struktur File
1. inference.py: tes model
2. parser.py: arsing input user, kayak satuan unit nya
3. carbon_calculator.py: perhitungan karbon dari input user
4. recipe_carbon.py: perhitungan karbon untuk resep
5. main.py: user input bahan
6. all_carbon.csv: dataset karbon dari setiap food item atau bahan
7. nama_file.csv: dataset resep bersih dari NLP dan sudah ditambahkan 2 kolom baru yaitu carbon_score (jumlah karbon sesuai kuantitas bahan), total_recipe_carbon (jumlah karbon resep dari setiap bahan)
8. link dataset resep yang sudah dihitung karbonnya: https://drive.google.com/drive/u/7/folders/1la7r2U0dBTddq9tm3elTnQvdKXzxfirs

# Cara Menjalankan
1. Run main.py untuk user dapat menginput bahan.
2. Run recipe_carbon.py untuk melakukan perhitungan carbon untuk semua bahan pada resep, tetapi ini sudah dijalankan dan disimpan menjadi csv namma_file.csv
