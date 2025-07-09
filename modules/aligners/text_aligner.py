from typing import Optional
from typing import List, Dict
import torch
import re
from modules.word_segmenter import WordSegmenter

from config import (
    ALIGNMENT_MODEL,
    WHISPER_DEVICE,
    ALIGNMENT_DEVICE
)


def process_ctc_text(segments: List[Dict], language_code: str,
                     word_segmenter: Optional[object],
                     max_line_length: int) -> str:
    """处理CTC强制对齐的文本"""
    line_list = []
    for segment in segments:
        line = segment['text'].strip()
        # 如果文本长度超过最大行长度，则进行分割
        if word_segmenter and len(line) > max_line_length:
            split_lines = split_long_line(line, word_segmenter, max_line_length)
            line_list.extend(split_lines)
        else:
            line_list.append(line)
    text = ' '.join([segment['text'] for segment in segments])
    # if language_code == 'zh':
    #     text = replace_chinese_punctuation_with_space(text)
    return text


def split_long_line(line: str, word_segmenter: WordSegmenter,
                    max_length: int) -> List[str]:
    output = []
    # 提取并添加特定符号中的词语到 jieba 词典
    extract_and_add_phrases(word_segmenter, line)

    line = line.strip()
    if line:
        # 1.不超过max_length
        if len(line) <= max_length:
            output.append(line)
        # 2. 超过max_length，先按标点切分
        else:
            sub_sentences = [s.strip() for s in re.split('[，。,;；：？?！…]', line)
                             if
                             s.strip()]
            for sentence in sub_sentences:
                if len(sentence) <= max_length:
                    output.append(sentence)
                # 3. 仍然超过max_line_width，使用jieba分词
                else:
                    spit_sentences = word_segmenter.split_long_sentence(
                        sentence, max_length)
                    output.extend(spit_sentences)

    return output


def extract_and_add_phrases(word_segmenter, text):
    pattern = '(《[^》]*》|"[^"]*"|\'[^\']*\'|‘[^’]*’|“[^”]*”|\([^\)]*\)|（[^）]*）|「[^」]*」)'
    phrases = re.findall(pattern, text)
    for phrase in phrases:
        # 将词语添加到分词器的词典中
        word_segmenter.add_word(phrase)


def replace_chinese_punctuation_with_space(text: str) -> str:
    # 将所有匹配的中文标点替换为句号
    chinese_punctuation_pattern = re.compile(r'[，。,;；：？?！…、]')
    result = chinese_punctuation_pattern.sub('。', text)
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
    else:
        return "eng"  # 默认返回英文代码


class TextAligner:
    """文本对齐器类，用于将文本与音频对齐"""

    def __init__(self, language_code: Optional[str] = None,
                 word_segmenter: Optional[object] = None,
                 max_line_length: int = 16):
        self.language_code = language_code or 'zh'  # 默认语言代码为中文
        self.model = self._load_model()
        self.word_segmenter = word_segmenter
        self.max_line_length = max_line_length

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
            text = process_ctc_text(segments, self.language_code,
                                    self.word_segmenter,
                                    self.max_line_length)
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
