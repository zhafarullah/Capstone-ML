from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status
from fastapi.responses import JSONResponse
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
import logging

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Carbon + Recipe API (Image Similarity)",
    description="API for finding similar images based on EfficientNet features.",
    version="0.1.0",
    openapi_tags=[
        {"name": "Image Similarity", "description": "Operations for finding similar images."},
        {"name": "Health Check", "description": "API health monitoring."}
    ]
)

# Global variables
model = None
features = None
urls = None
filenames = None

def get_model():
    global model
    if model is None:
        logger.info("Loading efficientnet_model.h5 (lazy)...")
        model = load_model("efficientnet_model.h5")
        logger.info("Model loaded.")
    return model

def get_data():
    global features, urls, filenames
    if features is None or urls is None or filenames is None:
        logger.info("Loading feature files...")
        with open('features_gdrive.pkl', 'rb') as f:
            features, urls = pickle.load(f)
        with open('features.pkl', 'rb') as f2:
            _, filenames = pickle.load(f2)
        logger.info(f"Loaded {len(features)} features.")
    return features, urls, filenames

def extract_feature_from_bytes(img_bytes: bytes):
    try:
        img = image.load_img(io.BytesIO(img_bytes), target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # DUMMY FEATURE VECTOR untuk bypass model.predict()
        logger.info("⚠️ Menggunakan dummy vector untuk uji ringan")
        features, _, _ = get_data()  # gunakan dimensi fitur asli
        feature = np.random.rand(1, features.shape[1])  # dummy vector sesuai dimensi

        return feature.flatten()

    except Exception as e:
        logger.error(f"Error during feature extraction: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Image processing failed: {e}")


def to_thumbnail_url(gdrive_url: str):
    file_id = re.search(r'id=([^&]+)', gdrive_url)
    if file_id:
        return f"https://drive.google.com/thumbnail?id={file_id.group(1)}"
    return gdrive_url

def prettify_filename(filename: str):
    name = os.path.basename(filename)
    name = re.sub(r'-\d+', '', name)
    name = re.sub(r'\.jpg$|\.jpeg$|\.png$', '', name, flags=re.IGNORECASE)
    name = name.replace('-', ' ').strip()
    return ' '.join(w.capitalize() for w in name.split())

@app.get("/health", tags=["Health Check"])
async def health_check():
    try:
        _ = get_model()
        _ = get_data()
        return {"status": "ok", "message": "Model and data loaded successfully."}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail=f"Health check failed: {e}")

@app.post("/", tags=["Image Similarity"])
async def find_similar_images(
    file: UploadFile = File(..., description="Image file to find similarities for."),
    top_n: int = Form(5, ge=1, le=20, description="Number of top similar results to return.")
):
    if file.filename == '':
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No file uploaded.")
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Only image files are supported.")

    try:
        img_bytes = await file.read()
        feature = extract_feature_from_bytes(img_bytes).reshape(1, -1)
        features, urls, filenames = get_data()

        similarities = cosine_similarity(feature, features)[0]
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

        return JSONResponse(content={"results": results})
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error in / endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Internal server error: {e}")
