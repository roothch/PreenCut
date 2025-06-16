from binascii import Error
from modules.speech_recognizers.faster_whisper_speech_recognizer import FasterWhisperSpeechRecognizer
from modules.speech_recognizers.whisperx_speech_recognizer import WhsiperXSpeechRecognizer
from modules.speech_recognizers.speech_recognizer import SpeechRecognizer
from config import (
    SPEECH_RECOGNIZER_TYPE,
    WHISPER_MODEL_SIZE,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_GPU_IDS,
    WHISPER_BATCH_SIZE,
)

class SpeechRecognizerFactory:
    def __init__(self):
        pass
    def getSpeechRecognizerByType(type, model_size) -> SpeechRecognizer:
        speechRecongnizer = None
        if type == 'faster_whisper':
            speechRecongnizer = FasterWhisperSpeechRecognizer(
                model_size, 
                WHISPER_DEVICE,
                compute_type=WHISPER_COMPUTE_TYPE,
                device_index = WHISPER_GPU_IDS,
                batch_size = WHISPER_BATCH_SIZE
            )
        elif type == 'whisperx':
            speechRecongnizer = WhsiperXSpeechRecognizer(
                model_size, 
                WHISPER_DEVICE, 
                compute_type = WHISPER_COMPUTE_TYPE,
                device_index =  WHISPER_GPU_IDS,
                batch_size = WHISPER_BATCH_SIZE
            )
        else:
            raise Error("not support sppech recongnizer type")   
        return speechRecongnizer