import base64
import io
import os

import runpod
from PIL import Image

# Lazy-load globals
_MODEL = None
_PROCESSOR = None

MODEL_ID = os.environ.get("MODEL_ID", "allenai/olmOCR-2-7B-1025-FP8")

def load_model():
    import transformers
    print("Transformers version:", transformers.__version__)
    global _MODEL, _PROCESSOR
    if _MODEL is not None:
        return

    import torch
    from transformers import AutoProcessor, AutoModelForVision2Seq

    _PROCESSOR = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    _MODEL = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    _MODEL.eval()

def decode_image(b64: str) -> Image.Image:
    # Accept raw base64 or data URLs like: data:image/jpeg;base64,....
    if "," in b64 and b64.strip().startswith("data:"):
        b64 = b64.split(",", 1)[1]
    img_bytes = base64.b64decode(b64)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")

def handler(event):
    load_model()

    inp = event.get("input", {}) or {}
    image_b64 = inp.get("image_base64")
    if not image_b64:
        return {"error": "Missing input.image_base64 (base64-encoded JPEG/PNG)"}

    prompt = inp.get("prompt") or "Transcribe all text in this image. Return only the text."

    image = decode_image(image_b64)

    # NOTE: Exact pre/post-processing can vary per model.
    # This is a generic Vision2Seq pattern and may need adjustment for olmOCR.
    inputs = _PROCESSOR(images=image, text=prompt, return_tensors="pt")
    inputs = {k: v.to(_MODEL.device) for k, v in inputs.items()}

    generated_ids = _MODEL.generate(
        **inputs,
        max_new_tokens=int(inp.get("max_new_tokens", 512)),
        temperature=float(inp.get("temperature", 0.0)),
    )

    text = _PROCESSOR.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return {"text": text}

runpod.serverless.start({"handler": handler})
