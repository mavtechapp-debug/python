import io
import json
import re
import gc
import time
from typing import Optional, Dict, Any

import requests
from PIL import Image

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from transformers import AutoProcessor, AutoModelForVision2Seq


# =========================
# CONFIG
# =========================
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

DEFAULT_CONF = 0.35  # só para referência (LMM não usa conf, mas mantemos no payload)
REQUEST_TIMEOUT = 20  # segundos
MAX_IMAGE_BYTES = 12 * 1024 * 1024  # 12MB, evita abuso
MAX_NEW_TOKENS = 650

PROMPT = """
You are a traffic enforcement expert.
Analyze the FULL image (global context) and list ALL traffic violations that are VISIBLE or PLAUSIBLE.

Rules:
- Do not invent facts.
- If you cannot confirm from a single photo, set status="inconclusive".
- For each item, include concrete visual evidence from the image.
- Output ONLY valid JSON. No extra text.

Required JSON format:
{
  "scene_summary": "short description",
  "possible_violations": [
    {
      "type": "STRING_ENUM",
      "status": "likely|possible|inconclusive|none",
      "confidence": 0.0,
      "evidence": ["..."],
      "missing_info": ["..."]
    }
  ],
  "notes": ["..."],
  "disclaimer": "..."
}

Use these types when applicable:
- PHONE_WHILE_DRIVING
- NO_HELMET
- NO_SEATBELT
- RUN_RED_LIGHT
- STOP_SIGN_IGNORED
- SPEEDING
- ILLEGAL_PARKING
- WRONG_WAY
- PEDESTRIAN_CROSSWALK_VIOLATION
- LANE_VIOLATION
- OTHER
""".strip()


# =========================
# LOAD MODEL (lazy)
# =========================
processor: Optional[AutoProcessor] = None
model: Optional[AutoModelForVision2Seq] = None


def ensure_model_loaded():
    global processor, model
    if processor is None:
        processor = AutoProcessor.from_pretrained(MODEL_ID)
    if model is None:
        model = AutoModelForVision2Seq.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
        )


def _extract_json(text: str) -> Dict[str, Any]:
    """
    Extract first JSON object from model output.
    """
    text = text.strip()

    # remove ```json fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError("Could not find JSON object in model output.")
    return json.loads(m.group(0))


def _download_image(url: str) -> Image.Image:
    """
    Download image with size limit and basic content-type check.
    """
    headers = {"User-Agent": "traffic-ai/1.0"}
    r = requests.get(url, stream=True, timeout=REQUEST_TIMEOUT, headers=headers)
    r.raise_for_status()

    ctype = (r.headers.get("Content-Type") or "").lower()
    if "image" not in ctype:
        # allow sometimes missing ctype, but try anyway; if it fails PIL will error
        pass

    data = b""
    for chunk in r.iter_content(chunk_size=256 * 1024):
        if not chunk:
            continue
        data += chunk
        if len(data) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail="Image too large (limit 12MB).")

    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image from URL.")
    return img


def analyze_image_with_qwen(img: Image.Image, max_new_tokens: int = MAX_NEW_TOKENS) -> Dict[str, Any]:
    ensure_model_loaded()

    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": PROMPT}
        ]
    }]

    inputs = processor(
        messages=messages,
        images=img,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens)

    text = processor.decode(out[0], skip_special_tokens=True)

    try:
        return _extract_json(text)
    except Exception as e:
        # fallback: return raw output for debugging
        return {
            "error": f"Model output was not valid JSON: {e}",
            "raw_output": text
        }


def clear_gpu_cache():
    """
    "Apagar" cache: libera VRAM e memória do Python.
    """
    global processor, model

    # Mantemos o processor (leve) e liberamos o modelo (pesado), se você quiser.
    # Se preferir manter o modelo carregado (mais rápido), comente as 2 linhas abaixo.
    model = None
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# =========================
# FASTAPI
# =========================
app = FastAPI(title="Traffic Infraction Analyzer (Qwen3-VL)")


class AnalyzeRequest(BaseModel):
    image_url: HttpUrl
    max_new_tokens: Optional[int] = MAX_NEW_TOKENS


@app.get("/health")
def health():
    return {
        "ok": True,
        "model_loaded": model is not None,
        "cuda": torch.cuda.is_available(),
        "model_id": MODEL_ID
    }


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    started = time.time()
    img = _download_image(str(req.image_url))
    result = analyze_image_with_qwen(img, max_new_tokens=int(req.max_new_tokens or MAX_NEW_TOKENS))
    result["_meta"] = {
        "elapsed_sec": round(time.time() - started, 3),
        "model_id": MODEL_ID,
    }
    return result


@app.post("/clear-cache")
def clear_cache():
    """
    Endpoint para "apagar": limpa cache e libera VRAM.
    """
    clear_gpu_cache()
    return {"ok": True, "message": "Cache cleared and VRAM freed (model unloaded)."}

