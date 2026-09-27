# iScribe — single-container deployment.
#
# Everything runs in one process: the API, the static UI and local CPU
# inference. No external AI service is contacted at runtime, so clinical audio
# and generated notes never leave the host.
#
# The Whisper and spaCy models are baked into the image so a cold start needs
# no network and cannot be blocked by a hospital firewall.

FROM python:3.12-slim

# ffmpeg/ffprobe: faster-whisper decodes via bundled PyAV, but ffprobe is used
# for duration probing and ffmpeg is needed if pyannote diarization is enabled.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/models

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir \
       https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

COPY scribe_engine/ ./scribe_engine/
COPY service/ ./service/

# Bake the STT weights in. Build-time download, so runtime is fully offline.
ARG WHISPER_MODEL=base
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('${WHISPER_MODEL}', device='cpu', compute_type='int8')" \
    && python -c "import spacy; spacy.load('en_core_web_sm')"

# Run unprivileged. /data is the only writable location the app needs.
RUN useradd --create-home --uid 10001 iscribe \
    && mkdir -p /data \
    && chown -R iscribe:iscribe /data /opt/models
USER iscribe

ENV ISCRIBE_ENV=production \
    ISCRIBE_HOST=0.0.0.0 \
    ISCRIBE_PORT=8123 \
    ISCRIBE_DATA_DIR=/data \
    HF_HUB_OFFLINE=1

EXPOSE 8123
VOLUME ["/data"]

# Liveness only — /api/ready additionally reports model state.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8123/api/health || exit 1

CMD ["python", "-m", "service"]
