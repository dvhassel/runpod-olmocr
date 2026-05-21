FROM nvidia/cuda:12.1.0-base-ubuntu22.04

RUN apt-get update -y && apt-get install -y python3-pip git && rm -rf /var/lib/apt/lists/*

WORKDIR /
COPY requirements.txt /requirements.txt
RUN python3 -m pip install --upgrade pip && python3 -m pip install -r /requirements.txt

COPY handler.py /handler.py
ENV MODEL_ID=allenai/olmOCR-2-7B-1025-FP8

CMD ["python3", "-u", "/handler.py"]