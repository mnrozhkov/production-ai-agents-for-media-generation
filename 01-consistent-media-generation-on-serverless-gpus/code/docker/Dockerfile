# Lesson 03/04 image worker: CUDA runtime base pinned by tag and digest; weights, secrets and outputs stay outside.
# Re-pin the digest on purpose (e.g. for security fixes).
FROM pytorch/pytorch:2.13.0-cuda13.0-cudnn9-runtime@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_BREAK_SYSTEM_PACKAGES=1 HF_HOME=/cache/huggingface
RUN pip install diffusers==0.39.0 transformers==5.13.1 accelerate==1.14.0
WORKDIR /app
COPY common.py generate.py generation.json /app/
RUN mkdir -p /cache /outputs
CMD ["python", "/app/generate.py", "--n", "1", "--out", "/outputs/run"]
