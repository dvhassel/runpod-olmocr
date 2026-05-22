FROM nvidia/cuda:12.1.0-base-ubuntu22.04

RUN apt-get update -y \
 && apt-get install -y python3-pip git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /

COPY requirements.txt /requirements.txt

# Install dependencies (Transformers from source via requirements.txt)
RUN python3 -m pip install --upgrade pip \
 && python3 -m pip install --no-cache-dir -r /requirements.txt

# Sanity check: ensure this Transformers build recognizes the model architecture
RUN python3 -c "from transformers import AutoConfig; AutoConfig.from_pretrained('allenai/olmOCR-2-7B-1025-FP8', trust_remote_code=True); print('Config OK')"

COPY handler.py /handler.py

ENV MODEL_ID=allenai/olmOCR-2-7B-1025-FP8

CMD ["python3", "-u", "/handler.py"]
