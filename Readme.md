# Inference Code NLP Feature

Branch berikut adalah branch dari model NLP

## cara Menjalankan
1. Clone Repositories ini dengan  
`git clone -b branch-a https://github.com/zhafarullah/Capstone-ML`
2. Install semua requirements.txt
3. Download file csv berikut dan letakkan di root utama  
https://drive.google.com/file/d/1Y3EO2xEUxNZf02yUUYEf6H6xO3zhpNHY/view?usp=sharing
4. Jalankan main.py

## Penjelasan isi repository 
* Folder generating vocab dan ner  
    * data_prep.py adalah kode untuk membuat vocab yang berisi Ingredient, Unit dan Quantity dari dataset output nya adalah vocab_lists.json
    * generate_synthetic.py adalah kode untuk membuat data Named Entity Recognition (NER) buatan (tidak diambil dari data nyata), dengan pola-pola yang sudah ditentukan, outputnya adalah synthetic_ner.json
* Folder Matriks Evaluasi  
    Berisi matriks evaluasi dari model NLP
* best_ner_bilstm.h5 
  adalah model LSTM untuk melakukan parsing terhadap input teks
* tok2idx.pkl  
  Berisi mapping dari setiap token (kata) ke indeks numerik yang digunakan untuk input ke model NLP.
* label_encoder.pkl  
  Berisi encoder yang mengubah label entitas (seperti B-ING, I-UNIT, dsb.) menjadi angka untuk keperluan pelatihan model NER.