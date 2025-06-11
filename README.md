# Capstone Deploy

## Cara Menjalankan
1. Aktifkan virtual environment
   ```
   Windows Commad Prompt 
   .env\Scripts\activate.bat
   Bash
   source .env/bin/activate
   ```
2. Jika belum ada virtual environment
```
Windows
py -m venv .env
Linux/MacOs
python3 -m venv .env
```
3. Instalasi depedensi
   ```pip install -r requirements.txt```
4. Jalankan program
   ```python main.py```
- main.py sebagai program utama yang dijalankan untuk menerima input dari user
- setelah menerima input dari user, teks input tersebut akan diparsing dan dibagi menjadi nama bahan, kuantitas, dan unit
- kemudian teks inout tersebut dihitung melalu carbon_calculator.py berdasarkan kuantitas dan unitnya dari dataset carbon
- sedangkan nama bahannya dicari menggunakan fuzzy match dari dataset carbon, jika belum juga ditemukan dicari menggunakan model Word2Vec dan akan dikelompokkan menjadi cluster
- dan bahan yang sama pun dicari dengan NLP untuk mendapatkan rekomendasi resep dari bahan yang diinputkan
