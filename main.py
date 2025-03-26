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

# Configurar logging
logging.basicConfig(level=logging.INFO)
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
                "whisper": "pending",
                "translation": "pending",
                "tts": "pending"
            }
        }

app_state = AppState()
app = FastAPI(title="Sistema de Dublagem Automática")

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
UPLOAD_DIR = Path("uploads")
PROCESSED_DIR = Path("processed")
UPLOAD_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)

class TaskStatus(BaseModel):
    status: str
    message: str
    progress: int = 0
    download_url: Optional[str] = None

# Dicionário para armazenar o status das tarefas
tasks = {}

def load_whisper():
    try:
        logger.info("Carregando modelo Whisper...")
        app_state.models["whisper"] = whisper.load_model("base")
        app_state.models["loading_status"]["whisper"] = "loaded"
        logger.info("Modelo Whisper carregado com sucesso!")
    except Exception as e:
        logger.error(f"Erro ao carregar modelo Whisper: {str(e)}")
        app_state.models["loading_status"]["whisper"] = "error"

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
        app_state.models["tts"] = TTS("tts_models/pt/cv/vits")
        app_state.models["loading_status"]["tts"] = "loaded"
        logger.info("Modelo TTS carregado com sucesso!")
    except Exception as e:
        logger.error(f"Erro ao carregar modelo TTS: {str(e)}")
        app_state.models["loading_status"]["tts"] = "error"

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

@app.post("/upload-video/")
async def upload_video(
    video: UploadFile = File(...),
    target_language: str = "es",
    voice: str = "male"
):
    if not video.filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Formato de vídeo não suportado")
    
    # Gerar ID único para a tarefa
    task_id = str(uuid.uuid4())
    
    # Salvar o vídeo
    video_path = UPLOAD_DIR / f"{task_id}_{video.filename}"
    async with aiofiles.open(video_path, 'wb') as out_file:
        content = await video.read()
        await out_file.write(content)
    
    # Registrar tarefa
    tasks[task_id] = TaskStatus(
        status="processing",
        message="Iniciando processamento do vídeo...",
        progress=0
    )
    
    # Iniciar processamento em background
    asyncio.create_task(process_video(task_id, video_path, target_language, voice))
    
    return {"task_id": task_id}

@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    """Verifica o status de uma tarefa"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Tarefa não encontrada")
    return tasks[task_id]

@app.get("/download/{filename}")
async def download_video(filename: str):
    """Download do vídeo processado"""
    file_path = PROCESSED_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Arquivo não encontrado")
    return FileResponse(str(file_path))

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

async def process_video(task_id: str, video_path: Path, target_language: str, voice: str):
    try:
        # Verificar se o arquivo existe
        if not video_path.exists():
            raise Exception("Arquivo de vídeo não encontrado")
            
        # Verificar se o arquivo tem tamanho maior que 0
        if video_path.stat().st_size == 0:
            raise Exception("Arquivo de vídeo está vazio")
        
        # Atualizar status
        tasks[task_id].message = "Extraindo áudio do vídeo..."
        tasks[task_id].progress = 10
        
        # Extrair áudio do vídeo
        audio_path = UPLOAD_DIR / f"{task_id}_audio.wav"
        result = os.system(f'ffmpeg -i "{video_path}" -vn -acodec pcm_s16le -ar 44100 -ac 2 "{audio_path}"')
        
        if result != 0:
            raise Exception("Falha ao extrair áudio do vídeo")
            
        if not audio_path.exists():
            raise Exception("Arquivo de áudio não foi criado")
            
        if audio_path.stat().st_size == 0:
            raise Exception("Arquivo de áudio está vazio")
        
        # Transcrever áudio
        tasks[task_id].message = "Transcrevendo áudio..."
        tasks[task_id].progress = 30
        
        try:
            result = app_state.models["whisper"].transcribe(str(audio_path))
            if not result or "text" not in result:
                raise Exception("Falha na transcrição do áudio")
            transcription = result["text"].strip()
            
            if not transcription:
                raise Exception("Nenhum texto foi transcrito do áudio")
        except Exception as e:
            raise Exception(f"Erro na transcrição: {str(e)}")
        
        # Traduzir texto
        tasks[task_id].message = "Traduzindo texto..."
        tasks[task_id].progress = 50
        
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
        
        # Gerar áudio
        tasks[task_id].message = "Gerando áudio traduzido..."
        tasks[task_id].progress = 70
        
        try:
            output_path = PROCESSED_DIR / f"{task_id}_output.wav"
            app_state.models["tts"].tts_to_file(text=translated_text, file_path=str(output_path))
            
            if not output_path.exists():
                raise Exception("Arquivo de áudio traduzido não foi criado")
                
            if output_path.stat().st_size == 0:
                raise Exception("Arquivo de áudio traduzido está vazio")
        except Exception as e:
            raise Exception(f"Erro na geração do áudio: {str(e)}")
        
        # Combinar áudio com vídeo
        tasks[task_id].message = "Finalizando vídeo..."
        tasks[task_id].progress = 90
        
        try:
            output_video_path = PROCESSED_DIR / f"{task_id}_final.mp4"
            result = os.system(f'ffmpeg -i "{video_path}" -i "{output_path}" -c:v copy -c:a aac "{output_video_path}"')
            
            if result != 0:
                raise Exception("Falha ao combinar áudio com vídeo")
                
            if not output_video_path.exists():
                raise Exception("Arquivo de vídeo final não foi criado")
                
            if output_video_path.stat().st_size == 0:
                raise Exception("Arquivo de vídeo final está vazio")
        except Exception as e:
            raise Exception(f"Erro na finalização do vídeo: {str(e)}")
        
        # Limpar arquivos temporários
        try:
            if audio_path.exists():
                os.remove(audio_path)
            if output_path.exists():
                os.remove(output_path)
        except Exception as e:
            logger.warning(f"Erro ao limpar arquivos temporários: {str(e)}")
        
        # Finalizar
        tasks[task_id].status = "completed"
        tasks[task_id].message = "Processamento concluído!"
        tasks[task_id].progress = 100
        tasks[task_id].download_url = f"/download/{task_id}_final.mp4"
        
    except Exception as e:
        tasks[task_id].status = "error"
        tasks[task_id].message = f"Erro durante o processamento: {str(e)}"
        logger.error(f"Erro no processamento da tarefa {task_id}: {str(e)}")
        
        # Limpar arquivos temporários em caso de erro
        try:
            if audio_path.exists():
                os.remove(audio_path)
            if output_path.exists():
                os.remove(output_path)
        except:
            pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 