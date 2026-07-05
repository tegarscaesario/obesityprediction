# Gunakan Python slim image sebagai base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set working directory di dalam container
WORKDIR /app

# Salin file requirements.txt terlebih dahulu
COPY requirements.txt /app/

# Instal dependensi Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Salin seluruh isi folder proyek ke dalam container
COPY . /app/

# Expose port untuk Streamlit (8501) dan FastAPI (8000)
EXPOSE 8501 8000

# Perintah default (akan menjalankan Streamlit)
CMD ["streamlit", "run", "app_streamlit.py", "--server.address=0.0.0.0", "--server.port=8501"]
