# Minimal Python image for inference
FROM python:3.10-slim

# Güvenlik: non-root kullanıcı oluştur
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /home/app

WORKDIR /home/app

# Bağımlılıkları yükle (cache optimizasyonu için önce requirements)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Sadece kaynak kodu kopyala (models/ KOPYALANMAZ!)
COPY src/ ./src/

# Sahipliği düzenle
RUN chown -R app:app ./src

# Non-root kullanıcı ile çalıştır
USER app

# Port aç
EXPOSE 8000

# Uygulamayı başlat
CMD ["python", "-m", "uvicorn", "src.inference.app:app", "--host", "0.0.0.0", "--port", "8000"]
