from celery import Celery
from typing import List, Dict, Optional
import os
from voice_manager import VoiceManager
from lip_sync import LipSync
import whisper
from transformers import MarianMTModel, MarianTokenizer
import ffmpeg
import json

# Configuração do Celery
celery_app = Celery('tasks', broker='redis://localhost:6379/0')
celery_app.conf.update(
    result_backend='redis://localhost:6379/0',
    task_serializer='json',
    result_serializer='json',
    accept_content=['json']
)

# Inicialização dos modelos
voice_manager = VoiceManager()
lip_sync = LipSync()

# Dicionário de modelos de tradução
TRANSLATION_MODELS = {
    'pt-es': 'Helsinki-NLP/opus-mt-pt-es',
    'pt-en': 'Helsinki-NLP/opus-mt-pt-en',
    'pt-fr': 'Helsinki-NLP/opus-mt-pt-fr',
    'en-pt': 'Helsinki-NLP/opus-mt-en-pt',
    'es-pt': 'Helsinki-NLP/opus-mt-es-pt',
    'fr-pt': 'Helsinki-NLP/opus-mt-fr-pt'
}

@celery_app.task
def process_video(
    video_path: str,
    target_language: str,
    voice_profile: Optional[str] = None,
    optimize_for_mobile: bool = False,
    add_subtitles: bool = False
) -> Dict:
    """Processa um vídeo para dublagem"""
    try:
        # Cria diretórios temporários
        os.makedirs('temp', exist_ok=True)
        os.makedirs('output', exist_ok=True)
        
        # 1. Extrai áudio
        audio_path = f"temp/audio_{os.path.basename(video_path)}.wav"
        stream = ffmpeg.input(video_path)
        stream = ffmpeg.output(stream, audio_path)
        ffmpeg.run(stream)
        
        # 2. Transcreve áudio
        whisper_model = whisper.load_model("large")
        result = whisper_model.transcribe(audio_path)
        transcription = result["text"]
        
        # 3. Traduz texto
        source_language = result["language"]
        model_key = f"{source_language}-{target_language}"
        
        if model_key not in TRANSLATION_MODELS:
            raise ValueError(f"Par de idiomas não suportado: {model_key}")
            
        translator_model = MarianMTModel.from_pretrained(TRANSLATION_MODELS[model_key])
        translator_tokenizer = MarianTokenizer.from_pretrained(TRANSLATION_MODELS[model_key])
        
        inputs = translator_tokenizer([transcription], return_tensors="pt", padding=True)
        translated = translator_model.generate(**inputs)
        translation = translator_tokenizer.decode(translated[0], skip_special_tokens=True)
        
        # 4. Gera áudio traduzido
        dubbed_audio_path = f"temp/dubbed_{os.path.basename(video_path)}.wav"
        voice_manager.generate_speech(
            text=translation,
            language=target_language,
            output_path=dubbed_audio_path,
            profile_name=voice_profile
        )
        
        # 5. Sincroniza lábios
        synced_video_path = f"temp/synced_{os.path.basename(video_path)}"
        lip_sync.adjust_video_timing(
            video_path=video_path,
            audio_path=dubbed_audio_path,
            output_path=synced_video_path
        )
        
        # 6. Otimiza para mobile se necessário
        if optimize_for_mobile:
            final_video_path = f"output/final_{os.path.basename(video_path)}"
            stream = ffmpeg.input(synced_video_path)
            stream = ffmpeg.output(
                stream,
                final_video_path,
                vcodec='libx264',
                acodec='aac',
                preset='fast',
                crf='28',
                movflags='+faststart'
            )
            ffmpeg.run(stream)
        else:
            final_video_path = synced_video_path
            
        # 7. Adiciona legendas se solicitado
        if add_subtitles:
            srt_path = f"temp/subtitles_{os.path.basename(video_path)}.srt"
            with open(srt_path, 'w', encoding='utf-8') as f:
                f.write(generate_srt(translation))
                
            video_with_subs = f"output/final_with_subs_{os.path.basename(video_path)}"
            stream = ffmpeg.input(final_video_path)
            stream = ffmpeg.output(
                stream,
                video_with_subs,
                vf=f"subtitles={srt_path}"
            )
            ffmpeg.run(stream)
            final_video_path = video_with_subs
            
        return {
            "status": "success",
            "video_path": final_video_path,
            "translation": translation
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }
        
    finally:
        # Limpa arquivos temporários
        for path in [audio_path, dubbed_audio_path, synced_video_path]:
            if os.path.exists(path):
                os.remove(path)

def generate_srt(text: str) -> str:
    """Gera arquivo SRT a partir do texto traduzido"""
    # Implementação simplificada - você pode melhorar a divisão do texto
    lines = text.split('. ')
    srt = ""
    for i, line in enumerate(lines, 1):
        start_time = format_time(i * 3)  # 3 segundos por linha
        end_time = format_time(i * 3 + 3)
        srt += f"{i}\n{start_time} --> {end_time}\n{line}\n\n"
    return srt

def format_time(seconds: int) -> str:
    """Formata tempo em formato SRT (HH:MM:SS,mmm)"""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d},000"

@celery_app.task
def batch_process_videos(
    video_paths: List[str],
    target_languages: List[str],
    voice_profile: Optional[str] = None,
    optimize_for_mobile: bool = False,
    add_subtitles: bool = False
) -> List[Dict]:
    """Processa múltiplos vídeos em lote"""
    results = []
    for video_path in video_paths:
        for target_language in target_languages:
            result = process_video.delay(
                video_path=video_path,
                target_language=target_language,
                voice_profile=voice_profile,
                optimize_for_mobile=optimize_for_mobile,
                add_subtitles=add_subtitles
            )
            results.append(result.get())  # Aguarda resultado
    return results 