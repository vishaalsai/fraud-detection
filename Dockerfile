FROM python:3.10-slim

WORKDIR /app

# Install dependencies first (layer-cached separately from source)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Download model artifacts from GitHub (pkl files are stored via Git LFS on GitHub
# and cannot be committed to HuggingFace Spaces directly)
RUN apt-get update && apt-get install -y --no-install-recommends wget \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p models \
    && wget -q "https://github.com/vishaalsai/fraud-detection/raw/main/models/xgboost_tuned.pkl" -O models/xgboost_tuned.pkl \
    && wget -q "https://github.com/vishaalsai/fraud-detection/raw/main/models/scaler.pkl" -O models/scaler.pkl \
    && wget -q "https://github.com/vishaalsai/fraud-detection/raw/main/models/evaluation_results.json" -O models/evaluation_results.json

# Make startup script executable
RUN chmod +x start.sh

EXPOSE 7860

CMD ["./start.sh"]
