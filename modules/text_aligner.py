from typing import Optional

from jieba.lac_small.predict import batch_size
from sqlalchemy.testing.suite.test_reflection import metadata

from config import (
    ALIGNMENT_MODEL,
    WHISPERX_DEVICE,
    WHISPERX_GPU_IDS,
    WHISPERX_ALIGN_BATCH_SIZE
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
                print("加载WhisperX对齐模型")
                model = whisperx.load_align_model(
                    language_code=self.language_code, device=WHISPERX_DEVICE,
                    device_index=WHISPERX_GPU_IDS)
                return model
            except ImportError:
                raise ImportError(
                    "WhisperX not installed. Please install with 'pip install whisperx'")
        else:
            raise ValueError(
                f"Unsupported forced alignment model: {ALIGNMENT_MODEL}")

    def align(self, segments: str, audio_path: str) -> str:
        """将文本与音频对齐，生成SRT字幕"""
        if ALIGNMENT_MODEL == 'whisperx':
            # 使用WhisperX进行对齐
            import whisperx
            audio = whisperx.load_audio(audio_path)
            align_model, align_model_metadata = self.model
            result = whisperx.align(segments, align_model, align_model_metadata,
                                    audio, WHISPERX_DEVICE,
                                    batch_size=WHISPERX_ALIGN_BATCH_SIZE,
                                    return_char_alignments=False)
            print('after align', result["segments"], flush=True)
            return "srt_content"
        else:
            raise ValueError(
                f"Unsupported forced alignment model: {ALIGNMENT_MODEL}")
