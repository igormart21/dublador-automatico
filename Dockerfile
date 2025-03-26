FROM python:3.9-slim

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Criar diretórios de trabalho
WORKDIR /app
RUN mkdir -p uploads processed

# Copiar arquivos do projeto
COPY requirements.txt .
COPY main.py .
COPY templates/ templates/
COPY static/ static/

# Instalar dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Expor porta
EXPOSE 8000

# Comando para iniciar a aplicação
CMD ["python", "main.py"] 