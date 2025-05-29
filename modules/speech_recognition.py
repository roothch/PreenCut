from config import (
    SPEECH_RECOGNITION_MODEL,
    WHISPER_MODEL_SIZE,
    FUNASR_MODEL_NAME,
    FASTER_WHISPER_MODEL,
    FASTER_WHISPER_DEVICE,
    FASTER_WHISPER_COMPUTE_TYPE
)
import os
import warnings
from typing import Union


class SpeechRecognizer:
    def __init__(self):
        self.model = self._load_model()

    def _load_model(self):
        """根据配置加载语音识别模型"""
        if SPEECH_RECOGNITION_MODEL == 'whisper':
            try:
                import whisper
                print(f"加载Whisper模型: {WHISPER_MODEL_SIZE}")
                return whisper.load_model(WHISPER_MODEL_SIZE)
            except ImportError:
                raise ImportError(
                    "Whisper not installed. Please install with 'pip install openai-whisper'")

        elif SPEECH_RECOGNITION_MODEL == 'faster-whisper':
            try:
                from faster_whisper import WhisperModel
                print(
                    f"加载Faster-Whisper模型: {FASTER_WHISPER_MODEL}, 设备: {FASTER_WHISPER_DEVICE}, 计算类型: {FASTER_WHISPER_COMPUTE_TYPE}")
                model = WhisperModel(
                    FASTER_WHISPER_MODEL,
                    device=FASTER_WHISPER_DEVICE,
                    compute_type=FASTER_WHISPER_COMPUTE_TYPE
                )
                return model
            except ImportError:
                raise ImportError(
                    "Faster-Whisper not installed. Please install with 'pip install faster-whisper'")

        elif SPEECH_RECOGNITION_MODEL == 'funasr':
            try:
                from funasr import AutoModel
                print(f"加载FunASR模型: {FUNASR_MODEL_NAME}")
                return AutoModel(model=FUNASR_MODEL_NAME)
            except ImportError:
                raise ImportError(
                    "FunASR not installed. Please install with 'pip install funasr'")
        else:
            raise ValueError(
                f"Unsupported speech recognition model: {SPEECH_RECOGNITION_MODEL}")

    def transcribe(self, audio_path: str) -> str:
        """将音频文件转录为文本"""
        # 确保音频文件存在
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if SPEECH_RECOGNITION_MODEL == 'whisper':
            result = self.model.transcribe(audio_path)
            return result["text"]

        elif SPEECH_RECOGNITION_MODEL == 'faster-whisper':
            # 使用Faster-Whisper进行转录
            segments, info = self.model.transcribe(
                audio_path,
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
                word_timestamps=False
            )

            # 收集所有文本
            full_text = " ".join([segment.text for segment in segments])
            return full_text

        elif SPEECH_RECOGNITION_MODEL == 'funasr':
            result = self.model.generate(input=audio_path)[0]
            return result[0] if result else ""

        return ""