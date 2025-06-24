from binascii import Error
from modules.speech_recognizers.faster_whisper_speech_recognizer import \
    FasterWhisperSpeechRecognizer
from modules.speech_recognizers.whisperx_speech_recognizer import \
    WhisperXSpeechRecognizer
from modules.speech_recognizers.speech_recognizer import SpeechRecognizer
from config import (
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_GPU_IDS,
    WHISPER_BATCH_SIZE,
    FASTER_WHISPER_BEAM_SIZE
)


class SpeechRecognizerFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_speech_recognizer_by_type(type, model_size) -> SpeechRecognizer:
        if type == 'faster-whisper':
            speech_recognizer = FasterWhisperSpeechRecognizer(
                model_size,
                WHISPER_DEVICE,
                compute_type=WHISPER_COMPUTE_TYPE,
                device_index=WHISPER_GPU_IDS,
                beam_size=FASTER_WHISPER_BEAM_SIZE
            )
        elif type == 'whisperx':
            speech_recognizer = WhisperXSpeechRecognizer(
                model_size,
                WHISPER_DEVICE,
                compute_type=WHISPER_COMPUTE_TYPE,
                device_index=WHISPER_GPU_IDS,
                batch_size=WHISPER_BATCH_SIZE
            )
        else:
            raise Error("not support speech recognizer type")
        return speech_recognizer
