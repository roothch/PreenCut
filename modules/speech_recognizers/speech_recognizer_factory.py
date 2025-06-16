from binascii import Error
from modules.speech_recognizers.faster_whisper_speech_recognizer import FasterWhisperSpeechRecognizer
from modules.speech_recognizers.whisperx_speech_recognizer import WhsiperXSpeechRecognizer
from modules.speech_recognizers.speech_recognizer import SpeechRecognizer
from config import (
    SPEECH_RECOGNIZER_TYPE,
    WHISPERX_MODEL_SIZE,
    WHISPERX_DEVICE,
    WHISPERX_COMPUTE_TYPE,
    WHISPERX_GPU_IDS,
    WHISPERX_BATCH_SIZE
)

class SpeechRecognizerFactory:
    def __init__(self):
        pass
    def getSpeechRecognizerByType(type) -> SpeechRecognizer:
        speechRecongnizer = None
        if type == 'faster_whisper':
            speechRecongnizer = FasterWhisperSpeechRecognizer(
                WHISPERX_MODEL_SIZE, 
                WHISPERX_DEVICE,
                compute_type=WHISPERX_COMPUTE_TYPE,
                device_index = WHISPERX_GPU_IDS,
                batch_size = WHISPERX_BATCH_SIZE
            )
        elif type == 'whisperx':
            speechRecongnizer = WhsiperXSpeechRecognizer(
                WHISPERX_MODEL_SIZE, 
                WHISPERX_DEVICE, 
                compute_type = WHISPERX_COMPUTE_TYPE,
                device_index =  WHISPERX_GPU_IDS,
                batch_size = WHISPERX_BATCH_SIZE
            )
        else:
            raise Error("not support sppech recongnizer type")   
        return speechRecongnizer