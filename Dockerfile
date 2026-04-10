FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt_get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "main.py"]