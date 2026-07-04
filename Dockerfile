FROM python:3.11-slim AS base
ENV PYTHONUNBUFFERED=1

# git is needed at runtime for the "index a GitHub URL" feature (Phase 10)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (separate layer -> cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install the package itself
COPY pyproject.toml .
COPY codelens/ ./codelens/
RUN pip install --no-cache-dir -e .

# Directory where FAISS index files get written
RUN mkdir -p /app/data/indexes

EXPOSE 8000

CMD ["uvicorn", "codelens.api:app", "--host", "0.0.0.0", "--port", "8000"]