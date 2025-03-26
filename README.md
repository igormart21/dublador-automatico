# Dublador de Vídeos

Uma aplicação web para dublar vídeos usando IA, que transcreve o áudio, traduz e gera uma nova dublagem.

## Funcionalidades

- Upload de vídeos (MP4, AVI, MOV, MKV)
- Transcrição automática do áudio usando Whisper
- Tradução do texto usando mBART
- Geração de áudio dublado usando TTS
- Interface web responsiva e intuitiva
- Barra de progresso em tempo real
- Download automático do vídeo dublado

## Requisitos

- Python 3.9+
- FFmpeg
- Docker e Docker Compose (opcional)

## Instalação

### Usando Docker (Recomendado)

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/dublador-videos.git
cd dublador-videos
```

2. Inicie os containers:
```bash
docker-compose up --build
```

3. Acesse a aplicação em `http://localhost:8000`

### Instalação Manual

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/dublador-videos.git
cd dublador-videos
```

2. Crie um ambiente virtual:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

4. Configure as variáveis de ambiente:
```bash
cp .env.example .env
# Edite o arquivo .env com suas configurações
```

5. Inicie o servidor:
```bash
python main.py
```

6. Acesse a aplicação em `http://localhost:8000`

## Uso

1. Acesse a interface web
2. Arraste e solte um vídeo ou clique para selecionar
3. Aguarde o processamento
4. Faça o download do vídeo dublado

## Estrutura do Projeto

```
dublador-videos/
├── main.py              # Servidor Flask
├── requirements.txt     # Dependências Python
├── Dockerfile          # Configuração do container
├── docker-compose.yml  # Orquestração dos containers
├── .env               # Variáveis de ambiente
├── .gitignore        # Arquivos ignorados pelo git
├── README.md         # Este arquivo
├── static/           # Arquivos estáticos
│   ├── styles.css    # Estilos CSS
│   └── script.js     # JavaScript
├── templates/        # Templates HTML
│   └── index.html    # Página principal
├── uploads/          # Diretório de uploads
└── processed/        # Diretório de vídeos processados
```

## Contribuindo

1. Faça um fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## Contato

Seu Nome - [@seu_twitter](https://twitter.com/seu_twitter) - email@exemplo.com

Link do Projeto: [https://github.com/seu-usuario/dublador-videos](https://github.com/seu-usuario/dublador-videos) 