from typing import Optional
from typing import List, Dict
import torch
import re

from config import (
    ALIGNMENT_MODEL,
    WHISPER_DEVICE,
    ALIGNMENT_DEVICE
)


def process_ctc_text(segments: List[Dict], language_code: str) -> str:
    """处理CTC强制对齐的文本"""
    text = ' '.join([segment['text'] for segment in segments])
    if language_code == 'zh':
        # 中文文本处理：去除中文标点符号
        text = replace_chinese_punctuation_with_space(text)
    return text


def replace_chinese_punctuation_with_space(text: str) -> str:
    # 将所有匹配的中文标点替换为空格
    chinese_punctuation_pattern = re.compile(r'[，。；：？！…,]')
    result = chinese_punctuation_pattern.sub(' ', text)
    return result


def to_639_3(language_code: str) -> str:
    """将语言代码转换为ISO 639-3格式"""
    language_code = language_code.lower()
    if language_code == 'zh':
        return 'cmn'  # 中文
    elif language_code == 'en':
        return 'eng'  # 英语
    elif language_code == 'es':
        return 'spa'  # 西班牙语
    elif language_code == 'fr':
        return 'fra'  # 法语
    elif language_code == 'de':
        return 'deu'  # 德语
    elif language_code == 'ja':
        return 'jpn'
    elif language_code == 'ko':
        return 'kor'
    elif language_code == 'ru':
        return 'rus'
    elif language_code == 'ar':
        return 'ara'
    elif language_code == 'pt':
        return 'por'
    elif language_code == 'it':
        return 'ita'
    elif language_code == 'hi':
        return 'hin'
    else:
        return "eng"  # 默认返回英文代码


class TextAligner:
    """文本对齐器类，用于将文本与音频对齐"""

    def __init__(self, language_code: Optional[str] = None):
        self.language_code = language_code or 'zh'  # 默认语言代码为中文
        self.model = self._load_model()

    def _load_model(self):
        if ALIGNMENT_MODEL == 'whisperx':
            try:
                import whisperx
                print(
                    f"加载WhisperX对齐模型，语言{self.language_code}，设备{WHISPER_DEVICE}")
                model = whisperx.load_align_model(
                    language_code=self.language_code, device=WHISPER_DEVICE)
                return model
            except ImportError:
                raise ImportError(
                    "WhisperX not installed. Please install with 'pip install whisperx'")
        elif ALIGNMENT_MODEL == 'ctc-forced-aligner':
            try:
                from ctc_forced_aligner import load_alignment_model
                print(
                    f"加载CTC强制对齐模型，语言{self.language_code}，设备{ALIGNMENT_DEVICE}")
                model = load_alignment_model(
                    ALIGNMENT_DEVICE,
                    dtype=torch.float16 if ALIGNMENT_DEVICE == "cuda" else torch.float32,
                )
                return model
            except ImportError:
                raise ImportError(
                    "CTC Forced Aligner not installed. Please install with 'pip install git+https://github.com/MahmoudAshraf97/ctc-forced-aligner.git'")
        else:
            raise ValueError(
                f"Unsupported forced alignment model: {ALIGNMENT_MODEL}")

    def align(self, segments: List[Dict], audio_path: str) -> str:
        """将文本与音频对齐"""
        if ALIGNMENT_MODEL == 'whisperx':
            # 使用WhisperX进行对齐
            import whisperx
            audio = whisperx.load_audio(audio_path)
            align_model, align_model_metadata = self.model
            result = whisperx.align(segments, align_model, align_model_metadata,
                                    audio, WHISPER_DEVICE,
                                    return_char_alignments=False)
            return result
        elif ALIGNMENT_MODEL == 'ctc-forced-aligner':
            from ctc_forced_aligner import (
                load_audio,
                generate_emissions,
                preprocess_text,
                get_alignments,
                get_spans,
                postprocess_results,
            )

            alignment_model, alignment_tokenizer = self.model
            audio_waveform = load_audio(audio_path, alignment_model.dtype,
                                        alignment_model.device)
            text = process_ctc_text(segments, self.language_code)
            emissions, stride = generate_emissions(
                alignment_model, audio_waveform, batch_size=16
            )
            code_639_3 = to_639_3(self.language_code)
            tokens_starred, text_starred = preprocess_text(
                text,
                romanize=True,
                language=code_639_3,
            )
            segments, scores, blank_token = get_alignments(
                emissions,
                tokens_starred,
                alignment_tokenizer,
            )
            spans = get_spans(tokens_starred, segments, blank_token)
            result = postprocess_results(text_starred, spans, stride,
                                         scores)
            return {
                "segments": result,
            }

        else:
            raise ValueError(
                f"Unsupported forced alignment model: {ALIGNMENT_MODEL}")
