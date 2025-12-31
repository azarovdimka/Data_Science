# """
# Сервис для обработки аудиофайлов и получения транскрипций
# Транскрибация аудио в текст
# Поддержка форматов: MP3, WAV, M4A, OGG
# Конвертация в WAV
# Распознавание речи через Google Speech Recognition
# Валидация файлов (размер до 50MB)
# """

# try:
#     # import speech_recognition as sr
#     # from pydub import AudioSegment
#     AUDIO_AVAILABLE = True
# except ImportError:
#     AUDIO_AVAILABLE = False
#     print("Audio libraries not installed. Transcription unavailable.")

# import os
# from typing import Optional
# import asyncio

# class AudioService:
#     """Сервис для работы с аудиофайлами"""
    
#     def __init__(self):
#         if AUDIO_AVAILABLE:
#             self.recognizer = sr.Recognizer()
#         self.supported_formats = ['.mp3', '.wav', '.m4a', '.ogg']
    
    # def convert_to_wav(self, audio_path: str) -> str:
    #     """Конвертация аудиофайла в WAV формат"""
    #     file_ext = os.path.splitext(audio_path)[1].lower()
        
    #     if file_ext == '.wav':
    #         return audio_path
        
    #     try:
    #         # Загрузка аудиофайла
    #         if file_ext == '.mp3':
    #             audio = AudioSegment.from_mp3(audio_path)
    #         elif file_ext == '.m4a':
    #             audio = AudioSegment.from_file(audio_path, format="m4a")
    #         elif file_ext == '.ogg':
    #             audio = AudioSegment.from_ogg(audio_path)
    #         else:
    #             audio = AudioSegment.from_file(audio_path)
            
    #         # Конвертация в WAV
    #         wav_path = audio_path.rsplit('.', 1)[0] + '.wav'
    #         audio.export(wav_path, format="wav")
            
    #         return wav_path
            
    #     except Exception as e:
    #         raise Exception(f"Audio conversion error: {str(e)}")
    
    # async def transcribe_audio(self, audio_path: str) -> str:
    #     """Получение транскрипции из аудиофайла"""
    #     if not AUDIO_AVAILABLE:
    #         return "Audio libraries not installed. Transcription unavailable."
        
    #     try:
    #         # Конвертация в WAV если необходимо
    #         wav_path = self.convert_to_wav(audio_path)
            
    #         # Загрузка аудиофайла
    #         with sr.AudioFile(wav_path) as source:
    #             # Настройка для лучшего распознавания
    #             self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
    #             audio_data = self.recognizer.record(source)
            
    #         # Попытка распознавания с помощью Google Speech Recognition
    #         try:
    #             transcription = self.recognizer.recognize_google(
    #                 audio_data, 
    #                 language='ru-RU'
    #             )
    #             return transcription
    #         except sr.UnknownValueError:
    #             return "Could not recognize speech"
    #         except sr.RequestError as e:
    #             # Fallback на offline распознавание
    #             try:
    #                 transcription = self.recognizer.recognize_sphinx(
    #                     audio_data, 
    #                     language='ru-RU'
    #                 )
    #                 return transcription
    #             except:
    #                 return f"Recognition service error: {str(e)}"
            
    #     except Exception as e:
    #         raise Exception(f"Audio processing error: {str(e)}")
    #     finally:
    #         # Удаление временного WAV файла если он был создан
    #         if wav_path != audio_path and os.path.exists(wav_path):
    #             os.remove(wav_path)
    
    # def validate_audio_file(self, file_path: str) -> bool:
    #     """Валидация аудиофайла"""
    #     if not os.path.exists(file_path):
    #         return False
        
    #     file_ext = os.path.splitext(file_path)[1].lower()
    #     if file_ext not in self.supported_formats:
    #         return False
        
    #     # Проверка размера файла (максимум 50MB)
    #     file_size = os.path.getsize(file_path)
    #     max_size = 50 * 1024 * 1024  # 50MB
        
    #     return file_size <= max_size
    
    # async def process_audio_batch(self, audio_files: list) -> dict:
    #     """Обработка нескольких аудиофайлов"""
    #     results = {}
        
    #     for file_path in audio_files:
    #         try:
    #             if self.validate_audio_file(file_path):
    #                 transcription = await self.transcribe_audio(file_path)
    #                 results[file_path] = {
    #                     'success': True,
    #                     'transcription': transcription
    #                 }
    #             else:
    #                 results[file_path] = {
    #                     'success': False,
    #                     'error': 'Unsupported format or file size'
    #                 }
    #         except Exception as e:
    #             results[file_path] = {
    #                 'success': False,
    #                 'error': str(e)
    #             }
        
    #     return results