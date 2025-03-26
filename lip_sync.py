import cv2
import numpy as np
from scipy.signal import correlate
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

class LipSync:
    def __init__(self):
        # Carrega o modelo de reconhecimento de fala para análise de fonemas
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-xlsr-53")
        
    def analyze_audio(self, audio_path):
        """Analisa o áudio para detectar momentos de fala"""
        import soundfile as sf
        speech, sample_rate = sf.read(audio_path)
        
        # Processa o áudio
        input_values = self.processor(speech, sampling_rate=sample_rate, return_tensors="pt").input_values
        
        # Obtém as probabilidades de fonemas
        with torch.no_grad():
            logits = self.model(input_values).logits
        
        # Converte para timestamps
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        
        return self._get_phoneme_timestamps(logits, sample_rate)
    
    def _get_phoneme_timestamps(self, logits, sample_rate):
        """Converte logits em timestamps de fonemas"""
        # Simplificação: usa a energia do sinal para detectar momentos de fala
        energy = torch.sum(logits ** 2, dim=-1)
        timestamps = torch.nonzero(energy > energy.mean()).squeeze()
        
        return timestamps * (sample_rate / logits.shape[1])
    
    def adjust_video_timing(self, video_path, audio_path, output_path):
        """Ajusta o timing do vídeo para sincronizar com o áudio"""
        # Obtém timestamps de fala
        speech_timestamps = self.analyze_audio(audio_path)
        
        # Lê o vídeo
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Prepara o vídeo de saída
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, 
                            (int(cap.get(3)), int(cap.get(4))))
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Calcula o tempo atual do frame
            current_time = frame_count / fps
            
            # Encontra o timestamp de fala mais próximo
            nearest_speech = min(speech_timestamps, 
                               key=lambda x: abs(x - current_time))
            
            # Ajusta o frame se necessário
            if abs(nearest_speech - current_time) < 1/fps:
                # Aqui você pode adicionar efeitos de morphing ou ajustes sutis
                # para melhorar a sincronização labial
                pass
            
            out.write(frame)
            frame_count += 1
        
        cap.release()
        out.release()
        
        return output_path 