# syntax=docker/dockerfile:1
FROM nvidia/cuda:13.2.1-cudnn-runtime-ubuntu24.04@sha256:1c0c68dbf3258d32b446a02cb4be05c8478b65d320d7856ad33c4bbdf898ca86

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
    VIRTUAL_ENV=/opt/venv \
    WANDB_DISABLED=true

ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN apt-get update \
    && apt-get install --no-install-recommends --yes python3-pip python3.12 python3.12-venv \
    && python3.12 -m venv "$VIRTUAL_ENV" \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m pip install --upgrade pip==25.1.1 \
    && python3.12 -m pip install --index-url https://download.pytorch.org/whl/cu132 torch==2.12.1

COPY requirements.txt /tmp/requirements.txt
RUN python3.12 -m pip install --requirement /tmp/requirements.txt

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin experiment \
    && install --directory --owner=experiment --group=experiment /artifacts /datasets /workspace

COPY smoke_test.py /opt/qwen-commit/smoke_test.py
USER experiment
WORKDIR /workspace

ENTRYPOINT ["python3.12"]
CMD ["/opt/qwen-commit/smoke_test.py"]
