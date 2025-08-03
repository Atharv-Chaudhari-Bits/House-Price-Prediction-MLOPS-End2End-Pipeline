FROM python:3.12-slim

WORKDIR /app

# Install any system-level deps needed by libraries (minimal)
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy everything into image
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "California-Housing-Price-Prediction.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
