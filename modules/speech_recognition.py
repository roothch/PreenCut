from config import (
    SPEECH_RECOGNITION_MODEL,
    WHISPERX_MODEL_SIZE,
    WHISPERX_DEVICE,
    WHISPERX_COMPUTE_TYPE,
    WHISPERX_GPU_IDS,
    WHISPERX_BATCH_SIZE
)
import os


class SpeechRecognizer:
    def __init__(self):
        self.model = self._load_model()

    def _load_model(self):
        """根据配置加载语音识别模型"""
        if SPEECH_RECOGNITION_MODEL == 'whisperx':
            try:
                import whisperx
                print(f"加载WhisperX模型: {WHISPERX_MODEL_SIZE}")
                asr_options = {
                    "initial_prompt": "是的，这个句子是为了增加标点。",
                }
                model = whisperx.load_model(WHISPERX_MODEL_SIZE,
                                            WHISPERX_DEVICE,
                                            device_index=WHISPERX_GPU_IDS,
                                            compute_type=WHISPERX_COMPUTE_TYPE,
                                            asr_options=asr_options)
                return model
            except ImportError:
                raise ImportError(
                    "WhisperX not installed. Please install with 'pip install whisperx'")

        else:
            raise ValueError(
                f"Unsupported speech recognition model: {SPEECH_RECOGNITION_MODEL}")

    def transcribe(self, audio_path: str) -> str:
        """将音频文件转录为文本"""
        # 确保音频文件存在
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if SPEECH_RECOGNITION_MODEL == 'whisperx':
            # 使用WhisperX进行转录
            import whisperx
            audio = whisperx.load_audio(audio_path)
            result = self.model.transcribe(
                audio,
                batch_size=WHISPERX_BATCH_SIZE,
            )
            return result

        return ""
