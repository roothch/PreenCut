from typing import Optional
from modules.speech_recognizers.speech_recognizer import SpeechRecognizer
import whisperx


class WhisperXSpeechRecognizer(SpeechRecognizer):
    def __init__(
            self,
            model_size,
            device,
            compute_type,
            device_index,
            batch_size,
            language: Optional[str] = None
    ):
        super().__init__(
            model_size,
            device,
            device_index=device_index,
            compute_type=compute_type,
            batch_size=batch_size,
        )
        print(f"加载Whisper模型: {self.model_size}")
        print(f"device = {self.device}")
        print(f"{self.model_size, self.device, self.compute_type, self.opts}")
        asr_options = {
            "initial_prompt": "是的，这个句子是为了增加标点。",
        }
        if self.device == 'cuda':
            self.model = whisperx.load_model(
                self.model_size,
                self.device,
                device_index=self.device_index,
                compute_type=self.compute_type,
                asr_options=asr_options
            )
        else:
            self.model = whisperx.load_model(
                self.model_size,
                self.device,
                compute_type=self.compute_type,
                asr_options=asr_options
            )

    def transcribe(self, audio_path):
        self.before_transcribe(audio_path)
        audio = whisperx.load_audio(audio_path)
        print(f"get audio data: {audio_path}")
        print(f"batch size = {self.batch_size}")
        result = self.model.transcribe(
            audio,
            batch_size=self.batch_size,
        )

        return result
