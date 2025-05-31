from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
import io
import base64
import os
import re

app = Flask(__name__)

# Load model & features
model = load_model('efficientnet_model.h5')

# Load features and URLs (dari GDrive)
with open('features_gdrive.pkl', 'rb') as f:
    features, urls = pickle.load(f)

# Load filenames asli (dari path lokal lama)
with open('features.pkl', 'rb') as f2:
    _, filenames = pickle.load(f2)

def extract_feature_from_bytes(img_bytes):
    img = image.load_img(io.BytesIO(img_bytes), target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    feature = model.predict(img_array)
    return feature.flatten()

def to_data_url(file_bytes):
    encoded = base64.b64encode(file_bytes).decode('utf-8')
    return f"data:image/jpeg;base64,{encoded}"

def to_thumbnail_url(gdrive_url):
    file_id = re.search(r'id=([^&]+)', gdrive_url)
    if file_id:
        return f"https://drive.google.com/thumbnail?id={file_id.group(1)}"
    return gdrive_url

def prettify_filename(filename):
    name = os.path.basename(filename)
    name = re.sub(r'-\d+', '', name)  # Hapus angka setelah tanda minus
    name = re.sub(r'\.jpg$|\.jpeg$|\.png$', '', name, flags=re.IGNORECASE)  # Hapus ekstensi
    name = name.replace('-', ' ')     # Ganti - jadi spasi
    name = name.strip()
    name = ' '.join(w.capitalize() for w in name.split())
    return name

@app.route('/', methods=['GET', 'POST'])
def index():
    results = []
    query_img_url = None
    error = None
    if request.method == 'POST':
        if 'file' not in request.files or request.files['file'].filename == '':
            error = "No file uploaded."
        else:
            file = request.files['file']
            img_bytes = file.read()
            try:
                feature = extract_feature_from_bytes(img_bytes).reshape(1, -1)
                similarities = cosine_similarity(feature, features)[0]
                top_n = int(request.form.get('top_n', 5))
                top_idx = np.argsort(similarities)[::-1][:top_n]
                results = [
                    {
                        "filename": prettify_filename(filenames[i]),
                        "url": urls[i],
                        "similarity": round(float(similarities[i]), 3),
                        "thumbnail": to_thumbnail_url(urls[i])
                    }
                    for i in top_idx
                ]
                query_img_url = to_data_url(img_bytes)
            except Exception as e:
                error = f"Gagal memproses gambar: {str(e)}"
    return render_template('index.html', results=results, query_img_url=query_img_url, error=error)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
