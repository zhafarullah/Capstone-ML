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
import logging # Tambahkan logging untuk debugging

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inisialisasi FastAPI
app = FastAPI(
    title="Carbon + Recipe API (Image Similarity)",
    description="API for finding similar images based on EfficientNet features.",
    version="0.1.0",
    # Anda bisa menambahkan tag/grouping untuk OpenAPI UI
    openapi_tags=[
        {"name": "Image Similarity", "description": "Operations for finding similar images."},
        {"name": "Health Check", "description": "API health monitoring."}
    ]
)

# Variabel global untuk model dan fitur
model = None
features = None
urls = None
filenames = None

@app.on_event("startup")
async def load_resources():
    """Load the Keras model and features once at application startup."""
    global model, features, urls, filenames
    try:
        logger.info("Loading efficientnet_model.h5...")
        model = load_model('efficientnet_model.h5')
        logger.info("Model loaded successfully.")

        logger.info("Loading features_gdrive.pkl...")
        with open('features_gdrive.pkl', 'rb') as f:
            features, urls = pickle.load(f)
        logger.info(f"Features loaded successfully. Total features: {len(features)}")

        logger.info("Loading features.pkl...")
        with open('features.pkl', 'rb') as f2:
            _, filenames = pickle.load(f2)
        logger.info(f"Filenames loaded successfully. Total filenames: {len(filenames)}")

    except FileNotFoundError as e:
        logger.error(f"Error loading resource: {e}. Make sure the files are in the correct directory.")
        # Menghentikan startup jika file penting tidak ditemukan
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                            detail=f"Failed to load required model/features: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during resource loading: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                            detail=f"Failed to load resources: {e}")


def extract_feature_from_bytes(img_bytes: bytes):
    """Extracts features from image bytes using the loaded model."""
    if model is None:
        logger.error("Model not loaded during feature extraction.")
        raise RuntimeError("Model not loaded. Application startup failed or resources are missing.")
    
    try:
        img = image.load_img(io.BytesIO(img_bytes), target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        feature = model.predict(img_array)
        return feature.flatten()
    except Exception as e:
        logger.error(f"Error during feature extraction: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, 
                            detail=f"Image processing failed: {e}")

def to_thumbnail_url(gdrive_url: str):
    """Generates a Google Drive thumbnail URL from a shared URL."""
    file_id = re.search(r'id=([^&]+)', gdrive_url)
    if file_id:
        return f"https://drive.google.com/thumbnail?id={file_id.group(1)}"
    return gdrive_url

def prettify_filename(filename: str):
    """Cleans up and formats a filename for display."""
    name = os.path.basename(filename)
    name = re.sub(r'-\d+', '', name)  # Remove numbers after a dash
    name = re.sub(r'\.jpg$|\.jpeg$|\.png$', '', name, flags=re.IGNORECASE)  # Remove extensions
    name = name.replace('-', ' ')      # Replace dashes with spaces
    name = name.strip()
    name = ' '.join(w.capitalize() for w in name.split())
    return name

@app.get("/health", tags=["Health Check"])
async def health_check():
    """Checks the health of the API."""
    # Anda bisa menambahkan logika pengecekan model atau database di sini
    if model is not None and features is not None:
        return {"status": "ok", "message": "API is healthy and resources are loaded."}
    else:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
                            detail="API resources are not loaded yet or failed to load.")


@app.post("/", tags=["Image Similarity"])
async def find_similar_images(
    file: UploadFile = File(..., description="Image file to find similarities for."),
    top_n: int = Form(5, ge=1, le=20, description="Number of top similar results to return.")
):
    """
    Uploads an image and returns a list of similar images from the dataset.
    """
    if file.filename == '':
        logger.warning("No file uploaded.")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No file uploaded.")
    
    if not file.content_type.startswith("image/"):
        logger.warning(f"Invalid file type uploaded: {file.content_type}")
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Only image files are supported.")

    try:
        img_bytes = await file.read() # Asynchronously read file bytes
        
        # Ensure model and features are loaded
        if model is None or features is None or urls is None or filenames is None:
            logger.error("Application resources (model/features) are not loaded during POST request.")
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
                                detail="Application resources are not fully loaded. Please try again later.")

        feature = extract_feature_from_bytes(img_bytes).reshape(1, -1)
        
        # Calculate cosine similarity
        similarities = cosine_similarity(feature, features)[0]
        
        # Get top_n indices based on similarity
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
        
        logger.info(f"Successfully processed image and found {len(results)} similar items.")
        return JSONResponse(content={"results": results})

    except HTTPException as e:
        # Re-raise HTTPException if already handled by previous layers
        raise e
    except Exception as e:
        logger.error(f"An unexpected error occurred during image similarity processing: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                            detail=f"Internal server error: {e}")

if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get("PORT", 8000)) 
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)