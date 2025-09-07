import os, uvicorn, torch
from typing import List
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.schemas import PredictResponse, PredictBatchItem, ModelsResponseItem, Health
from app.config import ServeConfig
from app.inference import Predictor
from app.data.dicom_utils import dicom_to_png_bytes
from app.security import rate_limiter
from PIL import Image
from io import BytesIO

# App setup
app = FastAPI(title="Pneumonia CXR API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

cfg = ServeConfig()
PREDICTORS = {}
OUT_DIR = "served_outputs"
os.makedirs(OUT_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=OUT_DIR), name="static")

def get_predictor(model_id="default"):
    if model_id not in cfg.model_registry:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

    meta = cfg.model_registry[model_id]
    key = (meta["arch"], meta["path"])

    if key not in PREDICTORS:
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            PREDICTORS[key] = Predictor(meta["arch"], meta["path"], meta["img_size"], device)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

    return PREDICTORS[key]

@app.get("/health", response_model=Health)
def health():
    return {"status": "ok"}

@app.get("/models", response_model=List[ModelsResponseItem])
def models():
    return [
        {
            "id": "default",
            "name": "DenseNet-121",
            "params": "7.98M",
            "input_size": "224",
            "trained_on": "Kaggle CXR",
            "roc_auc": 0.95,
        }
    ]

@app.post("/predict", response_model=PredictResponse)
async def predict(
    request: Request,
    file: UploadFile = File(...),
    model_id: str = "default",
    enable_dicom: bool = False,
):
    rate_limiter(request.client.host)

    try:
        data = await file.read()
        if enable_dicom and file.filename.lower().endswith(".dcm"):
            data = dicom_to_png_bytes(data)

        img = Image.open(BytesIO(data)).convert("RGB")
        predictor = get_predictor(model_id)

        out_name = os.path.splitext(os.path.basename(file.filename))[0] + "_gradcam.png"
        out_path = os.path.join(OUT_DIR, out_name)

        label, conf = predictor.predict_with_cam(img, out_path)
        return {"label": label, "confidence": conf, "gradcam_url": f"/static/{out_name}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

@app.post("/predict-batch", response_model=List[PredictBatchItem])
async def predict_batch(
    request: Request,
    files: List[UploadFile] = File(...),
    model_id: str = "default",
    enable_dicom: bool = False,
):
    rate_limiter(request.client.host)
    predictor = get_predictor(model_id)
    results = []

    for f in files:
        try:
            data = await f.read()
            if enable_dicom and f.filename.lower().endswith(".dcm"):
                data = dicom_to_png_bytes(data)

            img = Image.open(BytesIO(data)).convert("RGB")
            out_name = os.path.splitext(os.path.basename(f.filename))[0] + "_gradcam.png"
            out_path = os.path.join(OUT_DIR, out_name)

            label, conf = predictor.predict_with_cam(img, out_path)
            results.append(
                {"filename": f.filename, "label": label, "confidence": conf,
                 "gradcam_url": f"/static/{out_name}"}
            )
        except Exception as e:
            results.append(
                {"filename": f.filename, "label": "Error", "confidence": 0.0, "gradcam_url": str(e)}
            )

    return results

if __name__ == "__main__":
    uvicorn.run(app, host=cfg.host, port=cfg.port)
