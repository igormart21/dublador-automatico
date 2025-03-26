# Dublador de Vídeos

Uma aplicação web para dublar vídeos de português para espanhol usando IA.

## Funcionalidades

- Upload de vídeos
- Transcrição automática usando Whisper
- Tradução de português para espanhol
- Geração de áudio em espanhol
- Combinação de vídeo original com áudio traduzido

## Requisitos

- Python 3.8+
- FFmpeg
- Dependências listadas em `requirements.txt`

## Instalação Local

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/dublador-videos.git
cd dublador-videos
```

2. Crie um ambiente virtual e ative-o:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
.\venv\Scripts\activate  # Windows
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

4. Instale o FFmpeg:
- Mac: `brew install ffmpeg`
- Linux: `sudo apt-get install ffmpeg`
- Windows: Baixe do site oficial e adicione ao PATH

5. Execute a aplicação:
```bash
uvicorn main:app --reload
```

A aplicação estará disponível em `http://localhost:8000`

## Deploy no Railway

1. Crie uma conta no [Railway](https://railway.app/)
2. Conecte seu repositório GitHub
3. Crie um novo projeto no Railway
4. Selecione o repositório
5. Configure as variáveis de ambiente necessárias
6. Deploy automático será iniciado

## Uso

1. Acesse a interface web
2. Faça upload de um vídeo em português
3. Aguarde o processamento
4. Baixe o vídeo dublado em espanhol

## Tecnologias Utilizadas

- FastAPI
- Whisper (OpenAI)
- Transformers (Hugging Face)
- TTS (Coqui)
- FFmpeg

## Licença

MIT 