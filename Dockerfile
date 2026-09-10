FROM python:3.11-slim as builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements/requirements-prod.txt .

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-prod.txt

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 integralindx && \
    mkdir -p /app /app/data && \
    chown -R integralindx:integralindx /app

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

COPY --chown=integralindx:integralindx src /app/src
COPY --chown=integralindx:integralindx corpus /app/corpus
COPY --chown=integralindx:integralindx config /app/config

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    ARTIFACT_REPO=noamshh/integralindx-encoder

USER integralindx

# The model, database and embedding cache total ~200MB and are not in git.
# They are pulled from the Hub so a clean clone builds without extra setup.
RUN python -c "\
from huggingface_hub import hf_hub_download; \
import os, shutil; \
r = os.environ['ARTIFACT_REPO']; \
d = lambda f: hf_hub_download(repo_id=r, filename=f); \
os.makedirs('/app/data/models/inference', exist_ok=True); \
os.makedirs('/app/data/embeddings/ii-cl-19m_prod', exist_ok=True); \
shutil.copy(d('integral.db'), '/app/data/integral.db'); \
shutil.copy(d('ii-cl-19m_prod.pt'), '/app/data/models/inference/ii-cl-19m_prod.pt'); \
[shutil.copy(d('embeddings/' + f), '/app/data/embeddings/ii-cl-19m_prod/' + f) \
 for f in ('embedder.json', 'embedder.meta.json', 'groups.pkl')]"

EXPOSE 8080

CMD ["uvicorn", "src.web.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--log-level", "info"]
