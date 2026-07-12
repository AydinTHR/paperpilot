FROM python:3.12-slim

# Never write .pyc, always flush stdout (docker logs sees ticks immediately).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependency layer first so code changes don't re-install everything.
COPY requirements.txt requirements-dashboard.txt ./
RUN pip install -r requirements.txt -r requirements-dashboard.txt

COPY config/ config/
COPY src/ src/
COPY scripts/ scripts/
COPY pyproject.toml ./

# Journals and logs live on a bind mount so they survive image rebuilds.
VOLUME ["/app/data", "/app/logs"]

CMD ["python", "scripts/run_experiment.py", "--interval", "60"]
