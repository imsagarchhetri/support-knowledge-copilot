FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# system deps for PyMuPDF wheels are bundled; nothing extra needed for slim
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY eval ./eval
COPY ui ./ui
COPY data ./data
COPY pyproject.toml .

EXPOSE 8000
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
