import io, json, re, time, gc
from typing import Optional, Dict, Any

import requests
from PIL import Image

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl

from transformers import AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration

# =========================
# CONFIG
# =========================
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
REQUEST_TIMEOUT = 25
MAX_IMAGE_BYTES = 12 * 1024 * 1024
MAX_NEW_TOKENS = 650

PROMPT = """
Você é um especialista em fiscalização de trânsito no Brasil.

Analise A IMAGEM INTEIRA (contexto completo) e liste TODAS as infrações de trânsito que estejam VISÍVEIS ou POSSÍVEIS de serem inferidas a partir da imagem.

REGRAS OBRIGATÓRIAS:
- Responda SOMENTE em PORTUGUÊS (Brasil).
- Não invente fatos.
- Se a infração não puder ser confirmada com certeza a partir de uma única imagem, marque o status como "inconclusiva".
- Para cada infração, descreva claramente as evidências visuais observadas na imagem.
- Se faltar informação (ex: movimento, tempo, sinalização fora do enquadramento), informe em "informacoes_faltantes".
- Retorne APENAS um JSON válido, sem texto fora do JSON.

FORMATO EXATO DO JSON:
{
  "resumo_cena": "descrição curta da cena",
  "infracoes_possiveis": [
    {
      "tipo": "ENUM_STRING",
      "status": "provavel|possivel|inconclusiva|nenhuma",
      "confianca": 0.0,
      "evidencias": ["..."],
      "informacoes_faltantes": ["..."]
    }
  ],
  "observacoes": ["..."],
  "aviso_legal": "string"
}

UTILIZE ESTES TIPOS (quando aplicável):
- ESTACIONAMENTO_IRREGULAR
- USO_CELULAR_AO_VOLANTE
- SEM_CAPACETE
- SEM_CINTO_DE_SEGURANCA
- AVANCO_SINAL_VERMELHO
- DESRESPEITO_PARE
- EXCESSO_VELOCIDADE
- CONTRAMAO
- FAIXA_PEDESTRE
- FAIXA_OU_PISTA_IRREGULAR
- OUTRA
""".strip()

# =========================
# MODEL (lazy load)
# =========================
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
            trust_remote_code=True,
        )


def extract_json_anywhere(text: str) -> Dict[str, Any]:
    """
    Extrai o PRIMEIRO objeto JSON válido do texto, mesmo se vier com:
    - logs "system/user/assistant"
    - bloco ```json ... ```
    """
    if not text:
        raise ValueError("Saída do modelo vazia.")

    # remove fences ```json
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE)

    start = text.find("{")
    if start == -1:
        raise ValueError("Não encontrei '{' para iniciar um JSON na saída do modelo.")

    depth = 0
    end = None
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if end is None:
        raise ValueError("Não encontrei o final do JSON (chaves desbalanceadas).")

    candidate = text[start:end].strip()
    return json.loads(candidate)


def download_image(url: str) -> Image.Image:
    headers = {"User-Agent": "traffic-ai/1.0"}
    r = requests.get(url, timeout=REQUEST_TIMEOUT, headers=headers)
    r.raise_for_status()

    data = r.content
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Imagem muito grande (limite 12MB).")

    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Não foi possível decodificar a imagem.")


def analyze_with_qwen(image: Image.Image) -> Dict[str, Any]:
    """
    Qwen2.5-VL: monta o prompt com apply_chat_template e passa images=[PIL].
    """
    ensure_model_loaded()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("apply_chat_template retornou texto inválido/vazio.")

    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    try:
        return extract_json_anywhere(output_text)
    except Exception:
        # Se por algum motivo não vier JSON, devolve bruto pra debug
        return {"erro": "Modelo não retornou JSON válido", "raw_output": output_text}


def clear_gpu_cache(unload_model: bool = True):
    """
    "Apagar": limpa VRAM e (opcionalmente) descarrega o modelo.
    """
    global model
    if unload_model:
        model = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# =========================
# API
# =========================
app = FastAPI(title="Analisador de Infracoes (Qwen2.5-VL)")

class AnalyzeRequest(BaseModel):
    image_url: HttpUrl


@app.get("/health")
def health():
    return {
        "ok": True,
        "cuda": torch.cuda.is_available(),
        "modelo_carregado": model is not None,
        "model_id": MODEL_ID,
    }


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    started = time.time()
    try:
        img = download_image(str(req.image_url))
        result = analyze_with_qwen(img)
        result["_meta"] = {
            "tempo_segundos": round(time.time() - started, 3),
            "model_id": MODEL_ID,
        }
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clear-cache")
def clear_cache():
    clear_gpu_cache(unload_model=True)
    return {"ok": True, "mensagem": "Cache limpo e modelo descarregado da GPU (VRAM liberada)."}
