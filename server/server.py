from fastapi import FastAPI, HTTPException, UploadFile, File

from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import numpy as np
import base64
import io
import queue

from ocr.paddle_ocr_worker import OCRWorker
from translation.argos_translate import translate

app = FastAPI()
ocr_worker = OCRWorker(queue.Queue(2))  # Reuse across requests

def pil_to_numpy(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"))

def process_frame(frame: np.ndarray) -> dict:
    """
    Returns OCR results for a given frame in the following format:
    {
        "texts": [(str, (int, int, int, int)), ...],
        "process_time": currtime
    }
    """
    texts = ocr_worker.process_frame(frame)
    return {
        "texts": [
            {
                "text": text,
                "bbox": [int(coord) for coord in bbox]  # convert all to Python ints
            }
            for text, bbox in texts.get("texts", [])
        ],
        "process_time": float(texts["process_time"])  # ensure float, not np.float32
    }

def translate_batch(texts: list[str]) -> list[str]:
    """
    Translates a batch of texts using the translation service.
    """
    translated_texts = []
    for text in texts:
        translated_text = translate(text)
        translated_texts.append(translated_text)
    return translated_texts

class ImageRequest(BaseModel):
    image_base64: str

class TextRequest(BaseModel):
    text: str

@app.post("/analyze/image")
async def analyze_image_base64(payload: ImageRequest):
    try:
        # Decode base64 -> image bytes -> PIL -> NumPy
        image_data = base64.b64decode(payload.image_base64)
        image = Image.open(io.BytesIO(image_data))
        frame = pil_to_numpy(image)

        result = process_frame(frame)
        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process image: {e}")
    
@app.post("/analyze/image/file")
async def analyze_image_file(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        frame = pil_to_numpy(image)

        result = process_frame(frame)
        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image upload failed: {e}")
    
@app.post("/ocr/image")
async def ocr_image(payload: ImageRequest):
    try:
        # Decode base64 -> image bytes -> PIL -> NumPy
        image_data = base64.b64decode(payload.image_base64)
        image = Image.open(io.BytesIO(image_data))
        frame = pil_to_numpy(image)

        result = process_frame(frame)
        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process image: {e}")

@app.post("/translate")
async def translate_text(req: TextRequest):
    print(f"Received text for translation: {req.text}")
    return {"translated": translate(req.text)}

@app.post("/translate/batch")
async def translate_batch_texts(req: list[TextRequest]):
    texts = [item.text for item in req]
    print(f"Received batch for translation: {texts}")
    translated_texts = translate_batch(texts)
    return {"translated": translated_texts}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)