# Použijeme ofiko Python image
FROM python:3.11-slim

# DŮLEŽITÉ: Nainstalujeme systémové závislosti + FFmpeg
RUN apt-get update && apt-get install -y \
    ffmpeg \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Nastavíme pracovní složku
WORKDIR /app

# Zkopírujeme requirements a nainstalujeme Python knihovny
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Zkopírujeme zbytek kódu
COPY . .

# Spustíme Flask server
CMD ["python", "app.py"]