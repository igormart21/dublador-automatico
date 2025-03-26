# Use uma imagem base com Python 3.11
FROM python:3.11-slim

# Instala FFmpeg e outras dependências do sistema
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Define o diretório de trabalho
WORKDIR /app

# Copia os arquivos de requisitos primeiro
COPY requirements.txt .

# Instala as dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Copia o resto do código
COPY . .

# Cria diretórios necessários
RUN mkdir -p /data/uploads /data/processed

# Define variáveis de ambiente
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV UPLOAD_DIR=/data/uploads
ENV PROCESSED_DIR=/data/processed
ENV FFMPEG_PATH=/usr/bin/ffmpeg

# Expõe a porta
EXPOSE 8000

# Comando para iniciar a aplicação
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"] 