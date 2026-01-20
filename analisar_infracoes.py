import io, json, re, time, gc
from typing import Optional, Dict, Any

import requests
from PIL import Image

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl

from transformers import AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
REQUEST_TIMEOUT = 20
MAX_IMAGE_BYTES = 12 * 1024 * 1024
MAX_NEW_TOKENS = 650

PROMPT = """
You are a traffic enforcement expert.
Analyze the FULL image and list ALL traffic violations that are VISIBLE or PLAUSIBLE.

Rules:
- Do not invent facts.
- If not certain from one photo, set status="inconclusive".
- Include concrete visual evidence.
- Output ONLY valid JSON.

Return exactly this JSON:
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
""".strip()

processor: Optional[AutoProcessor] = None
model: Optional[Qwen2_5_VLForConditionalGeneration] = None

def ensure_model_loaded():
    global processor, model
    if processor is None:
        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    if model is None:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

def extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("JSON not found in model output")
    return json.loads(m.group(0))

def download_image(url: str) -> Image.Image:
    headers = {"User-Agent": "traffic-ai/1.0"}
    r = requests.get(url, timeout=REQUEST_TIMEOUT, headers=headers)
    r.raise_for_status()
    data = r.content
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "Image too large (limit 12MB)")
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not decode image")

def analyze_with_qwen(image: Image.Image) -> Dict[str, Any]:
    ensure_model_loaded()

    # Qwen2.5-VL expects messages with explicit image slots
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]

    # Build text prompt
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Process vision inputs
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    # Move tensors to model device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

    # Decode only the newly generated part
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    try:
        return extract_json(output_text)
    except Exception:
        return {"raw_output": output_text}

def clear_gpu_cache(unload_model: bool = True):
    global model
    if unload_model:
        model = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

app = FastAPI(title="Traffic Infraction Analyzer (Qwen2.5-VL)")

class AnalyzeRequest(BaseModel):
    image_url: HttpUrl

@app.get("/health")
def health():
    return {
        "ok": True,
        "cuda": torch.cuda.is_available(),
        "model_loaded": model is not None,
        "model_id": MODEL_ID
    }

@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    started = time.time()
    try:
        img = download_image(str(req.image_url))
        result = analyze_with_qwen(img)
        result["_meta"] = {"elapsed_sec": round(time.time() - started, 3)}
        return result
    except Exception as e:
        # devolve erro explícito pra não ficar cego no 500
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear-cache")
def clear_cache():
    clear_gpu_cache(unload_model=True)
    return {"ok": True, "message": "Cleared cache and unloaded model from GPU"}
