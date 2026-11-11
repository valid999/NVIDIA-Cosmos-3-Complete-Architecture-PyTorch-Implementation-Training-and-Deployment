"""Conservative demonstration API for a *trained toy reasoner*.

Never connect generated commands directly to factory PLCs/robots/actuators.
For actual NVIDIA Cosmos 3 deploy official vLLM/TensorRT-LLM/NIM backends.
"""
import base64
import binascii
import io
import os
import threading
from contextlib import asynccontextmanager
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from .train import load_model
from .sampling import generate_answer

_lock = threading.Lock()
_model = None
_device = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _device
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = os.environ.get("TOY_COSMOS_CHECKPOINT", "artifacts/cosmos3_toy.pt")
    _model = load_model(path, _device, require=True).eval()
    yield
    _model = None


app = FastAPI(title="Cosmos3 learning lab (NOT NVIDIA Cosmos inference)", lifespan=lifespan)


class ReasonRequest(BaseModel):
    question: str = Field(min_length=1, max_length=300)
    image_base64: str = Field(min_length=10, max_length=2_000_000)


@app.get("/health")
def health():
    return {"ready": _model is not None, "model": "independent educational MiniCosmos3"}


@app.post("/reason")
def reason(request: ReasonRequest):
    if _model is None:
        raise HTTPException(503, "Model not loaded")
    try:
        raw = base64.b64decode(request.image_base64, validate=True)
        with Image.open(io.BytesIO(raw)) as opened:
            if opened.width * opened.height > 4_000_000:
                raise HTTPException(413, "Image exceeds 4 megapixels")
            rgb = opened.convert("RGB").resize((_model.config.image_size, _model.config.image_size))
            array = np.asarray(rgb).copy()
    except (ValueError, binascii.Error, UnidentifiedImageError, OSError) as exc:
        raise HTTPException(400, f"Invalid image: {type(exc).__name__}") from exc
    frame = torch.from_numpy(array).permute(2, 0, 1).float().div(255)
    frames = frame.unsqueeze(0).unsqueeze(0).repeat(1, _model.config.video_frames, 1, 1, 1).to(_device)
    with _lock, torch.inference_mode():
        answer = generate_answer(_model, request.question, frames, max_new_tokens=12)
    return {"answer": answer, "warning": "Toy model, not calibrated or validated for industrial use"}
