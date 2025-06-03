from typing import Optional

from config import (
    ALIGNMENT_MODEL,
    WHISPERX_DEVICE,
    WHISPERX_GPU_IDS
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

    def align(self, text: str, audio_path: str) -> str:
        """将文本与音频对齐，生成SRT字幕 (占位实现)"""
        # 这里只是一个占位实现，实际中应替换为您的对齐模型
        # 返回一个示例SRT格式的字幕
        srt_content = "1\n00:00:00,000 --> 00:00:05,000\n这是第一句字幕\n\n"
        srt_content += "2\n00:00:05,000 --> 00:00:10,000\n这是第二句字幕\n\n"
        srt_content += "3\n00:00:10,000 --> 00:00:15,000\n这是第三句字幕"
        return srt_content
