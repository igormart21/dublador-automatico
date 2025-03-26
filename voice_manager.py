from TTS.api import TTS
import torch
import os
from typing import Dict, List, Optional
import json

class VoiceManager:
    def __init__(self):
        self.available_models = {
            'pt-br': {
                'model': 'tts_models/pt/cv/vits',
                'speaker_wav': None
            },
            'es': {
                'model': 'tts_models/es/css10/vits',
                'speaker_wav': None
            },
            'en': {
                'model': 'tts_models/en/ljspeech/vits',
                'speaker_wav': None
            },
            'fr': {
                'model': 'tts_models/fr/css10/vits',
                'speaker_wav': None
            },
            'de': {
                'model': 'tts_models/de/thorsten/vits',
                'speaker_wav': None
            },
            'it': {
                'model': 'tts_models/it/mai_female/vits',
                'speaker_wav': None
            },
            'ru': {
                'model': 'tts_models/ru/cv/vits',
                'speaker_wav': None
            },
            'zh-cn': {
                'model': 'tts_models/zh-CN/baker/vits',
                'speaker_wav': None
            }
        }
        
        self.loaded_models: Dict[str, TTS] = {}
        self.voice_profiles: Dict[str, dict] = self._load_voice_profiles()
        
    def _load_voice_profiles(self) -> Dict[str, dict]:
        """Carrega perfis de voz salvos"""
        if os.path.exists('voice_profiles.json'):
            with open('voice_profiles.json', 'r') as f:
                return json.load(f)
        return {}
    
    def save_voice_profile(self, profile_name: str, language: str, speaker_wav: str,
                          voice_settings: Optional[Dict] = None):
        """Salva um novo perfil de voz"""
        if language not in self.available_models:
            raise ValueError(f"Idioma {language} não suportado")
            
        profile = {
            'language': language,
            'speaker_wav': speaker_wav,
            'settings': voice_settings or {}
        }
        
        self.voice_profiles[profile_name] = profile
        
        with open('voice_profiles.json', 'w') as f:
            json.dump(self.voice_profiles, f, indent=2)
            
    def get_model(self, language: str) -> TTS:
        """Obtém ou carrega um modelo TTS para o idioma especificado"""
        if language not in self.available_models:
            raise ValueError(f"Idioma {language} não suportado")
            
        if language not in self.loaded_models:
            print(f"Carregando modelo para {language}...")
            model_path = self.available_models[language]['model']
            self.loaded_models[language] = TTS(model_name=model_path)
            
        return self.loaded_models[language]
    
    def generate_speech(self, text: str, language: str, output_path: str,
                       profile_name: Optional[str] = None,
                       speaker_wav: Optional[str] = None) -> str:
        """Gera fala a partir do texto usando o perfil especificado ou speaker_wav"""
        model = self.get_model(language)
        
        # Se um perfil foi especificado, usa suas configurações
        if profile_name and profile_name in self.voice_profiles:
            profile = self.voice_profiles[profile_name]
            speaker_wav = profile['speaker_wav']
            settings = profile['settings']
        else:
            settings = {}
            
        # Gera o áudio
        model.tts_to_file(
            text=text,
            file_path=output_path,
            speaker_wav=speaker_wav,
            language=language,
            **settings
        )
        
        return output_path
    
    def list_available_languages(self) -> List[str]:
        """Lista todos os idiomas disponíveis"""
        return list(self.available_models.keys())
    
    def list_voice_profiles(self) -> List[str]:
        """Lista todos os perfis de voz salvos"""
        return list(self.voice_profiles.keys())
    
    def clone_voice(self, audio_path: str, language: str, profile_name: str,
                   voice_settings: Optional[Dict] = None):
        """Clona uma voz a partir de um áudio de exemplo"""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Arquivo de áudio não encontrado: {audio_path}")
            
        self.save_voice_profile(
            profile_name=profile_name,
            language=language,
            speaker_wav=audio_path,
            voice_settings=voice_settings
        ) 