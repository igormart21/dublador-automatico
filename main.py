from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional
import os
import uuid
import aiofiles
from pathlib import Path
import whisper
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, AutoModelForSeq2SeqLM, AutoTokenizer
from TTS.api import TTS
import torch
from fastapi.middleware.cors import CORSMiddleware
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import shutil
import ffmpeg
import tempfile
from pydub import AudioSegment
import numpy as np
from moviepy.editor import VideoFileClip, AudioFileClip
import json
from flask import Flask, request, jsonify, render_template, send_file
from dotenv import load_dotenv
from celery import Celery
import redis

# Carregar variáveis de ambiente
load_dotenv()

# Configurar logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AppState:
    def __init__(self):
        self.is_ready = False
        self.models = {
            "whisper": None,
            "translation": None,
            "translation_tokenizer": None,
            "tts": None,
            "loading_status": {
                "whisper": "not_loaded",
                "translation": "not_loaded",
                "tts": "not_loaded"
            }
        }
        self.tasks = {}

app_state = AppState()
app = FastAPI(title="Dublador de Vídeos")

# Configuração de templates e arquivos estáticos
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Configuração do CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Criação dos diretórios necessários
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))
PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", "processed"))
UPLOAD_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)

class TaskStatus(BaseModel):
    status: str
    message: str
    progress: int = 0
    download_url: Optional[str] = None

# Dicionário para armazenar o status das tarefas
tasks = {}

# Configuração do Celery
redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
celery = Celery('tasks', broker=redis_url)
celery.conf.update(
    result_backend=redis_url,
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)

# Configurações
MAX_FILE_SIZE = int(os.getenv('MAX_FILE_SIZE', 100000000))  # 100MB
ALLOWED_VIDEO_TYPES = os.getenv('ALLOWED_VIDEO_TYPES', 'video/mp4,video/avi,video/mov,video/mkv').split(',')

def load_whisper():
    try:
        logger.info("Carregando modelo Whisper...")
        app_state.models["whisper"] = whisper.load_model("base")
        app_state.models["loading_status"]["whisper"] = "loaded"
        logger.info("Modelo Whisper carregado com sucesso!")
    except Exception as e:
        logger.error(f"Erro ao carregar modelo Whisper: {str(e)}")
        app_state.models["loading_status"]["whisper"] = "error"
        raise

def load_translation():
    try:
        logger.info("Carregando modelo de tradução...")
        # Usar um modelo mais robusto para português-espanhol
        model_name = "facebook/mbart-large-50-many-to-many-mmt"
        logger.info(f"Carregando modelo: {model_name}")
        
        try:
            app_state.models["translation"] = MBartForConditionalGeneration.from_pretrained(
                model_name,
                local_files_only=False,
                torch_dtype=torch.float32
            )
            logger.info("Modelo de tradução carregado com sucesso")
        except Exception as e:
            logger.error(f"Erro ao carregar modelo de tradução: {str(e)}")
            raise
        
        try:
            app_state.models["translation_tokenizer"] = MBart50TokenizerFast.from_pretrained(
                model_name,
                local_files_only=False
            )
            logger.info("Tokenizer carregado com sucesso")
        except Exception as e:
            logger.error(f"Erro ao carregar tokenizer: {str(e)}")
            raise
        
        # Configurar o tokenizer para português-espanhol
        try:
            app_state.models["translation_tokenizer"].src_lang = "pt_XX"
            app_state.models["translation_tokenizer"].tgt_lang = "es_XX"
            logger.info("Configuração de idiomas concluída")
        except Exception as e:
            logger.error(f"Erro ao configurar idiomas: {str(e)}")
            raise
        
        # Testar a tradução
        try:
            test_text = "Olá, como você está?"
            logger.info(f"Testando tradução com texto: {test_text}")
            
            inputs = app_state.models["translation_tokenizer"](
                test_text,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            logger.info("Texto tokenizado com sucesso")
            
            translated = app_state.models["translation"].generate(
                **inputs,
                forced_bos_token_id=app_state.models["translation_tokenizer"].lang_code_to_id["es_XX"]
            )
            logger.info("Tradução gerada com sucesso")
            
            translated_text = app_state.models["translation_tokenizer"].decode(
                translated[0],
                skip_special_tokens=True
            )
            logger.info(f"Resultado do teste: {translated_text}")
            
        except Exception as e:
            logger.error(f"Erro no teste de tradução: {str(e)}")
            raise
        
        app_state.models["loading_status"]["translation"] = "loaded"
        logger.info("Modelo de tradução carregado e testado com sucesso!")
    except Exception as e:
        logger.error(f"Erro fatal ao carregar modelo de tradução: {str(e)}")
        app_state.models["loading_status"]["translation"] = "error"
        raise

def load_tts():
    try:
        logger.info("Carregando modelo TTS...")
        app_state.models["tts"] = TTS(model_name="tts_models/es/css10/vits")
        app_state.models["loading_status"]["tts"] = "loaded"
        logger.info("Modelo TTS carregado com sucesso!")
    except Exception as e:
        logger.error(f"Erro ao carregar modelo TTS: {str(e)}")
        app_state.models["loading_status"]["tts"] = "error"
        raise

async def load_models():
    """Carrega os modelos em threads separadas"""
    with ThreadPoolExecutor(max_workers=3) as executor:
        loop = asyncio.get_event_loop()
        await asyncio.gather(
            loop.run_in_executor(executor, load_whisper),
            loop.run_in_executor(executor, load_translation),
            loop.run_in_executor(executor, load_tts)
        )
    
    # Verifica se todos os modelos foram carregados com sucesso
    if all(status == "loaded" for status in app_state.models["loading_status"].values()):
        app_state.is_ready = True
        logger.info("Todos os modelos foram carregados com sucesso! O servidor está pronto para uso.")
    else:
        logger.error("Alguns modelos não foram carregados corretamente. Verifique os logs para mais detalhes.")

@app.on_event("startup")
async def startup_event():
    """Evento executado quando o servidor inicia"""
    asyncio.create_task(load_models())

@app.middleware("http")
async def check_server_ready(request: Request, call_next):
    """Middleware para verificar se o servidor está pronto antes de processar requisições"""
    if not app_state.is_ready and request.url.path not in ["/status", "/docs", "/openapi.json"]:
        return JSONResponse(
            status_code=503,
            content={
                "detail": "O servidor está iniciando. Por favor, aguarde...",
                "loading_status": app_state.models["loading_status"]
            }
        )
    response = await call_next(request)
    return response

@app.get("/status")
async def get_status():
    """Retorna o status do carregamento dos modelos"""
    return JSONResponse(content={
        "is_ready": app_state.is_ready,
        "loading_status": app_state.models["loading_status"]
    })

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Página inicial do sistema"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "selectedFile": None,
        "is_ready": app_state.is_ready,
        "loading_status": app_state.models["loading_status"]
    })

@app.post("/upload/")
async def upload_video(video: UploadFile = File(...)):
    try:
        # Verificar se os modelos estão carregados
        if app_state.models["loading_status"]["whisper"] != "loaded":
            raise HTTPException(status_code=503, detail="Modelo Whisper não está carregado")
        if app_state.models["loading_status"]["translation"] != "loaded":
            raise HTTPException(status_code=503, detail="Modelo de tradução não está carregado")
        if app_state.models["loading_status"]["tts"] != "loaded":
            raise HTTPException(status_code=503, detail="Modelo TTS não está carregado")
            
        # Gerar ID único para a tarefa
        task_id = str(uuid.uuid4())
        
        # Criar diretório para o vídeo
        video_dir = UPLOAD_DIR / task_id
        video_dir.mkdir(exist_ok=True)
        
        # Salvar o vídeo
        video_path = video_dir / video.filename
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
            
        # Iniciar processamento em background
        asyncio.create_task(process_video(task_id, video_path, "es", "female"))
        
        return {"task_id": task_id, "message": "Vídeo recebido e processamento iniciado"}
        
    except Exception as e:
        logger.error(f"Erro no upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{task_id}")
async def get_task_status(task_id: str):
    """Verifica o status de uma tarefa"""
    if task_id not in app_state.tasks:
        raise HTTPException(status_code=404, detail="Tarefa não encontrada")
    return app_state.tasks[task_id]

@app.get("/download/{task_id}")
async def download_video(task_id: str):
    """Download do vídeo processado"""
    try:
        # Verificar se o arquivo existe
        file_path = PROCESSED_DIR / task_id / "output.mp4"
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Vídeo não encontrado")
            
        return FileResponse(str(file_path))
        
    except Exception as e:
        logger.error(f"Erro no download: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/test-translation/")
async def test_translation(text: str):
    """Rota para testar a tradução"""
    try:
        # Verificar se o modelo está carregado
        if app_state.models["loading_status"]["translation"] != "loaded":
            raise HTTPException(status_code=503, detail="Modelo de tradução não está carregado")
            
        # Configurar o tokenizer para português-espanhol
        app_state.models["translation_tokenizer"].src_lang = "pt_XX"
        app_state.models["translation_tokenizer"].tgt_lang = "es_XX"
        
        # Tokenizar o texto
        inputs = app_state.models["translation_tokenizer"](
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Gerar a tradução
        translated = app_state.models["translation"].generate(
            **inputs,
            forced_bos_token_id=app_state.models["translation_tokenizer"].lang_code_to_id["es_XX"],
            max_length=512,
            num_beams=5,
            early_stopping=True,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.2
        )
        
        # Decodificar a tradução
        translated_text = app_state.models["translation_tokenizer"].decode(
            translated[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        ).strip()
        
        # Log para debug
        logger.info(f"Texto original: {text}")
        logger.info(f"Texto traduzido: {translated_text}")
        
        return {
            "original": text,
            "translated": translated_text,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Erro no teste de tradução: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/test-full-process/")
async def test_full_process(text: str):
    """Rota para testar todo o processo de tradução e dublagem"""
    try:
        # Verificar se os modelos estão carregados
        if app_state.models["loading_status"]["translation"] != "loaded":
            raise HTTPException(status_code=503, detail="Modelo de tradução não está carregado")
        if app_state.models["loading_status"]["tts"] != "loaded":
            raise HTTPException(status_code=503, detail="Modelo TTS não está carregado")
            
        # 1. Testar tradução
        logger.info("Iniciando teste de tradução...")
        app_state.models["translation_tokenizer"].src_lang = "pt_XX"
        app_state.models["translation_tokenizer"].tgt_lang = "es_XX"
        
        # Tokenizar o texto
        inputs = app_state.models["translation_tokenizer"](
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Gerar a tradução
        translated = app_state.models["translation"].generate(
            **inputs,
            forced_bos_token_id=app_state.models["translation_tokenizer"].lang_code_to_id["es_XX"],
            max_length=512,
            num_beams=5,
            early_stopping=True,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.2
        )
        
        # Decodificar a tradução
        translated_text = app_state.models["translation_tokenizer"].decode(
            translated[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        ).strip()
        
        # 2. Testar geração de áudio
        logger.info("Iniciando teste de geração de áudio...")
        test_audio_path = PROCESSED_DIR / "test_audio.wav"
        app_state.models["tts"].tts_to_file(text=translated_text, file_path=str(test_audio_path))
        
        # Verificar se o arquivo de áudio foi gerado
        if not test_audio_path.exists():
            raise Exception("Falha ao gerar arquivo de áudio")
            
        if test_audio_path.stat().st_size == 0:
            raise Exception("Arquivo de áudio gerado está vazio")
        
        # Log para debug
        logger.info(f"Texto original: {text}")
        logger.info(f"Texto traduzido: {translated_text}")
        logger.info(f"Arquivo de áudio gerado: {test_audio_path}")
        
        # Limpar arquivo de teste
        try:
            if test_audio_path.exists():
                os.remove(test_audio_path)
        except Exception as e:
            logger.warning(f"Erro ao limpar arquivo de teste: {str(e)}")
        
        return {
            "original": text,
            "translated": translated_text,
            "audio_generated": True,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Erro no teste completo: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@celery.task(bind=True)
def process_video(self, task_id: str, video_path: Path, target_language: str, voice: str):
    try:
        # Atualizar status
        self.update_state(state='PROCESSING', meta={'progress': 10, 'message': 'Iniciando processamento...'})
        
        # Extrair áudio do vídeo
        app_state.tasks[task_id] = {
            "status": "processing",
            "message": "Extraindo áudio do vídeo...",
            "progress": 10
        }
        
        try:
            audio_path = video_path.parent / "audio.wav"
            stream = ffmpeg.input(str(video_path))
            stream = ffmpeg.output(stream, str(audio_path), acodec='pcm_s16le', ac=1, ar='16k')
            ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
            
            if not audio_path.exists():
                raise Exception("Falha ao extrair áudio do vídeo")
                
        except Exception as e:
            raise Exception(f"Erro ao extrair áudio: {str(e)}")
        
        # Transcrever áudio
        app_state.tasks[task_id].message = "Transcrevendo áudio..."
        app_state.tasks[task_id].progress = 30
        
        try:
            result = app_state.models["whisper"].transcribe(str(audio_path))
            transcription = result["text"].strip()
            
            if not transcription:
                raise Exception("Transcrição vazia")
                
        except Exception as e:
            raise Exception(f"Erro na transcrição: {str(e)}")
        
        # Traduzir texto
        app_state.tasks[task_id].message = "Traduzindo texto..."
        app_state.tasks[task_id].progress = 50
        
        try:
            # Verificar se o texto está vazio ou muito curto
            if len(transcription) < 2:
                raise Exception("Texto muito curto para tradução")
                
            # Limpar o texto de caracteres especiais
            transcription = transcription.strip()
            logger.info(f"Texto a ser traduzido: {transcription[:100]}...")
            
            # Configurar o tokenizer para português-espanhol
            app_state.models["translation_tokenizer"].src_lang = "pt_XX"
            app_state.models["translation_tokenizer"].tgt_lang = "es_XX"
            
            # Dividir o texto em sentenças menores para melhor tradução
            sentences = transcription.split('.')
            translated_sentences = []
            
            for sentence in sentences:
                if not sentence.strip():
                    continue
                    
                # Tokenizar o texto
                inputs = app_state.models["translation_tokenizer"](
                    sentence.strip(),
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                
                # Gerar a tradução
                translated = app_state.models["translation"].generate(
                    **inputs,
                    forced_bos_token_id=app_state.models["translation_tokenizer"].lang_code_to_id["es_XX"],
                    max_length=512,
                    num_beams=5,
                    early_stopping=True,
                    do_sample=True,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.95,
                    repetition_penalty=1.2
                )
                
                # Decodificar a tradução
                translated_text = app_state.models["translation_tokenizer"].decode(
                    translated[0],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                ).strip()
                
                translated_sentences.append(translated_text)
                
                # Log para debug
                logger.info(f"Sentença original: {sentence[:100]}...")
                logger.info(f"Sentença traduzida: {translated_text[:100]}...")
            
            # Juntar as sentenças traduzidas
            translated_text = '. '.join(translated_sentences)
            
            # Verificar se a tradução é válida
            if not translated_text or len(translated_text) < 2:
                raise Exception("Tradução resultou em texto inválido")
                
            if translated_text == transcription:
                raise Exception("A tradução é idêntica ao texto original")
                
        except Exception as e:
            raise Exception(f"Erro na tradução: {str(e)}")
        
        # Gerar áudio traduzido
        app_state.tasks[task_id].message = "Gerando áudio traduzido..."
        app_state.tasks[task_id].progress = 70
        
        try:
            translated_audio_path = video_path.parent / "translated_audio.wav"
            app_state.models["tts"].tts_to_file(text=translated_text, file_path=str(translated_audio_path))
            
            if not translated_audio_path.exists():
                raise Exception("Falha ao gerar áudio traduzido")
                
        except Exception as e:
            raise Exception(f"Erro ao gerar áudio traduzido: {str(e)}")
        
        # Combinar vídeo com áudio traduzido
        app_state.tasks[task_id].message = "Combinando vídeo com áudio traduzido..."
        app_state.tasks[task_id].progress = 90
        
        try:
            output_path = PROCESSED_DIR / task_id / "output.mp4"
            output_path.parent.mkdir(exist_ok=True)
            
            # Carregar vídeo e áudio
            video = VideoFileClip(str(video_path))
            audio = AudioFileClip(str(translated_audio_path))
            
            # Combinar vídeo com áudio traduzido
            final_video = video.set_audio(audio)
            
            # Salvar vídeo final
            final_video.write_videofile(
                str(output_path),
                codec='libx264',
                audio_codec='aac',
                temp_audiofile=str(video_path.parent / "temp-audio.m4a"),
                remove_temp=True
            )
            
            # Fechar clips
            video.close()
            audio.close()
            
            if not output_path.exists():
                raise Exception("Falha ao gerar vídeo final")
                
        except Exception as e:
            raise Exception(f"Erro ao combinar vídeo com áudio: {str(e)}")
        
        # Limpar arquivos temporários
        try:
            if audio_path.exists():
                os.remove(audio_path)
            if translated_audio_path.exists():
                os.remove(translated_audio_path)
            if video_path.exists():
                os.remove(video_path)
            shutil.rmtree(video_path.parent)
        except Exception as e:
            logger.warning(f"Erro ao limpar arquivos temporários: {str(e)}")
        
        # Atualizar status final
        app_state.tasks[task_id].status = "completed"
        app_state.tasks[task_id].message = "Processamento concluído!"
        app_state.tasks[task_id].progress = 100
        
    except Exception as e:
        logger.error(f"Erro no processamento do vídeo: {str(e)}")
        app_state.tasks[task_id].status = "error"
        app_state.tasks[task_id].message = f"Erro: {str(e)}"
        app_state.tasks[task_id].progress = 0

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 