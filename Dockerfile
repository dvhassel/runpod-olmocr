FROM nvidia/cuda:12.1.0-base-ubuntu22.04

RUN apt-get update -y && apt-get install -y python3-pip git && rm -rf /var/lib/apt/lists/*

WORKDIR /
COPY requirements.txt /requirements.txt


# Install deps (no cache) and fail the build if transformers is wrong
RUN python3 -m pip install --upgrade pip \
 && python3 -m pip install --no-cache-dir -r /requirements.txt \
 && python3 -c "import transformers; print('Transformers:', transformers.__version__)" \
 && python3 -c "from transformers import AutoModelForVision2Seq; print('AutoModelForVision2Seq OK')"

COPY handler.py /handler.py
ENV MODEL_ID=allenai/olmOCR-2-7B-1025-FP8

CMD ["python3", "-u", "/handler.py"]
