FROM python:3.10-slim

WORKDIR /app

# Install dependencies first (layer-cached separately from source)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Make startup script executable
RUN chmod +x start.sh

EXPOSE 7860

CMD ["./start.sh"]
