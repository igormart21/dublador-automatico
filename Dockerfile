FROM python:3.9-slim

# Instalar dependências do sistema
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Criar diretórios de trabalho
WORKDIR /app
RUN mkdir -p uploads processed static templates

# Copiar arquivos do projeto
COPY requirements.txt .
COPY main.py .

# Instalar dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar diretórios estáticos e templates
COPY static/ /app/static/
COPY templates/ /app/templates/

# Expor porta
EXPOSE 8000

# Comando para iniciar a aplicação
CMD ["python", "main.py"] 