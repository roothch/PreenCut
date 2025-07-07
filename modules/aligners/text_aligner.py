from typing import Optional
from typing import List, Dict

from config import (
    ALIGNMENT_MODEL,
    WHISPER_DEVICE,
)


class TextAligner:
    """文本对齐器类，用于将文本与音频对齐，生成SRT字幕"""

    def __init__(self, language_code: Optional[str] = None):
        self.language_code = language_code or 'zh'  # 默认语言代码为中文
        self.model = self._load_model()

    def _load_model(self):
        if ALIGNMENT_MODEL == 'whisperx':
            try:
                import whisperx
                print(f"加载WhisperX对齐模型，语言{self.language_code}")
                model = whisperx.load_align_model(
                    language_code=self.language_code, device=WHISPER_DEVICE)
                return model
            except ImportError:
                raise ImportError(
                    "WhisperX not installed. Please install with 'pip install whisperx'")
        else:
            raise ValueError(
                f"Unsupported forced alignment model: {ALIGNMENT_MODEL}")

    def align(self, segments: List[Dict], audio_path: str) -> str:
        """将文本与音频对齐，生成SRT字幕"""
        if ALIGNMENT_MODEL == 'whisperx':
            # 使用WhisperX进行对齐
            import whisperx
            audio = whisperx.load_audio(audio_path)
            align_model, align_model_metadata = self.model
            result = whisperx.align(segments, align_model, align_model_metadata,
                                    audio, WHISPER_DEVICE,
                                    return_char_alignments=False)
            return result
        else:
            raise ValueError(
                f"Unsupported forced alignment model: {ALIGNMENT_MODEL}")
