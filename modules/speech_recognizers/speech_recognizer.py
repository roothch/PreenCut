
import os


class SpeechRecognizer:
    model_size = 'large_v2'
    device = 'cpu'
    compute_type = 'float32'
    batch_size = 16
    device_index = []
    opts = {} 

    def __init__(self, model_size, device, device_index, compute_type, batch_size, opts = {}):
       self.model_size = model_size
       self.device = device
       self.device_index = device_index
       self.compute_type = compute_type
       self.batch_size = batch_size

    def before_transcribe(self, audio_path):
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"audio file not found: {audio_path}")
    def transcribe(self, audio_path):
        pass