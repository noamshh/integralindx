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

COPY --chown=integralindx:integralindx data/integral.db /app/data/
COPY --chown=integralindx:integralindx data/models/inference /app/data/models/inference
COPY --chown=integralindx:integralindx data/embeddings/ii-cl-19m_prod /app/data/embeddings/ii-cl-19m_prod

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080

USER integralindx
EXPOSE 8080

CMD ["uvicorn", "src.web.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--log-level", "info"]
