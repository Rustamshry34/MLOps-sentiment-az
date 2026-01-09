# Stage 1: Build — sadece inference için gerekli dosyaları kopyala
FROM python:3.10-slim

# Güvenlik: non-root kullanıcı oluştur
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /home/app

# Çalışma dizinini ayarla
WORKDIR /home/app

# Bağımlılıkları kopyala ve yükle (cache avantajı için önce requirements)
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Kodu ve modelleri kopyala
COPY src/ ./src/


# Sahipliği düzenle
RUN chown -R app:app ./src ./models

# Non-root kullanıcı ile çalıştır
USER app

# Port aç
EXPOSE 8000

# Uygulamayı başlat
CMD ["python", "-m", "uvicorn", "src.inference.app:app", "--host", "0.0.0.0", "--port", "8000"]
