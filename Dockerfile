# Educational CPU-only inference container, deliberately NOT an industrial controls service.
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir ".[api]"
RUN useradd --uid 10001 --create-home appuser
USER appuser
EXPOSE 8080
# Mount checkpoint at /app/artifacts/cosmos3_toy.pt (read only).
CMD ["uvicorn", "cosmos3_lab.serve:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
