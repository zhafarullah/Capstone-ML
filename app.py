from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status
from dotenv import load_dotenv
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
import io
import os
import re
import logging
import json
import cloudinary
import cloudinary.uploader
from pymongo import MongoClient

load_dotenv()

# ─── Konfigurasi Cloudinary ─────────────────────────────
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True
)

# ─── Konfigurasi Logging ────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Koneksi MongoDB ─────────────────────────────────────
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION")

client = MongoClient(MONGO_URI)
db = client[MONGO_DB]
collection = db[MONGO_COLLECTION]

# ─── Inisialisasi FastAPI ───────────────────────────────
app = FastAPI(
    title="Carbon + Recipe API (Image Similarity)",
    description="API for finding similar images based on EfficientNet features.",
    version="0.1.0",
    openapi_tags=[
        {"name": "Image Similarity", "description": "Operations for finding similar images."},
        {"name": "Health Check", "description": "API health monitoring."}
    ]
)

# ─── Variabel Global ─────────────────────────────────────
model = None
features = None
filenames = None
filename_to_url = {}

@app.on_event("startup")
async def load_resources():
    global model, features, filenames, filename_to_url
    try:
        logger.info("🔄 Loading EfficientNet model...")
        model = load_model('efficientnet_model.h5')
        logger.info("✅ Model loaded.")

        logger.info("🔄 Loading features from features_cloudinary.pkl...")
        with open('features_cloudinary.pkl', 'rb') as f:
            features, filenames = pickle.load(f)
        logger.info(f"✅ Features loaded: {len(features)}")

        logger.info("🔄 Loading filename_to_url.json...")
        with open("filename_to_url.json", "r") as f:
            filename_to_url = json.load(f)
        logger.info(f"✅ Cloudinary URL mapping loaded: {len(filename_to_url)} items")

    except Exception as e:
        logger.error(f"❌ Startup error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Startup failed: {e}")

def extract_feature_from_bytes(img_bytes: bytes):
    if model is None:
        raise RuntimeError("Model not loaded.")
    try:
        img = image.load_img(io.BytesIO(img_bytes), target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        feature = model.predict(img_array)
        return feature.flatten()
    except Exception as e:
        logger.error(f"❌ Feature extraction error: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Image processing failed: {e}")

import re

def format_instructions_cleaned(text: str) -> list:
    if not text:
        return []

    split_pattern = re.compile(r'(?<!\b[tT]bsp)(?<!\b[tT]sp)(?<!\betc)(?<!\d)\.\s+')
    steps = split_pattern.split(text)
    return [step.strip() + '.' for step in steps if step.strip()]


@app.get("/health", tags=["Health Check"])
async def health_check():
    if model is not None and features is not None:
        return {"status": "ok", "message": "Model and features loaded."}
    raise HTTPException(status_code=503, detail="Resources not loaded.")

@app.post("/", tags=["Image Similarity"])
async def find_similar_images(
    file: UploadFile = File(..., description="Upload image to find similar results."),
    top_n: int = Form(5, ge=1, le=20, description="Number of top similar results to return.")
):
    if file.filename == '':
        raise HTTPException(status_code=400, detail="No file uploaded.")
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Only image files are supported.")
    try:
        img_bytes = await file.read()
        if model is None or features is None or filenames is None:
            raise HTTPException(status_code=503, detail="Resources not available.")

        # Ekstraksi fitur & hitung similarity
        feature = extract_feature_from_bytes(img_bytes).reshape(1, -1)
        similarities = cosine_similarity(feature, features)[0]
        top_idx = np.argsort(similarities)[::-1][:top_n]

        results = []
        for i in top_idx:
            fname = os.path.splitext(os.path.basename(filenames[i]))[0]
            cloud_url = filename_to_url.get(f"{fname}.jpg", "URL_NOT_FOUND")

            # Ambil data dari MongoDB berdasarkan image_name
            mongo_doc = collection.find_one({"image_name": fname})
            if mongo_doc:
                total_carbon = mongo_doc.get("total_recipe_carbon")
                raw_instructions = mongo_doc.get("instructions_cleaned", "")
                instructions = format_instructions_cleaned(raw_instructions)
                ingredients = mongo_doc.get("cleaned_ingredients", [])
            else:
                total_carbon = None
                instructions = ""
                ingredients = []

            results.append({
                "filename": fname,
                "url": cloud_url,
                "similarity": round(float(similarities[i]), 3),
                "total_recipe_carbon": total_carbon,
                "instructions_cleaned": instructions,
                "cleaned_ingredients": ingredients,
                "thumbnail": cloud_url
            })

        return JSONResponse(content={"results": results})
    except Exception as e:
        logger.error(f"❌ Similarity error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Server error: {e}")

if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
