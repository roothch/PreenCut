import faster_whisper;
import os

from modules.speech_recognizers.speech_recognizer import SpeechRecognizer

class FasterWhisperSpeechRecognizer (SpeechRecognizer):
    beam_size = 5
    def __init__(
            self, 
            model_size, 
            device, 
            device_index, 
            compute_type,
            batch_size,
            beam_size
        ):
        super().__init__(model_size, device, device_index = device_index, compute_type = compute_type, batch_size=batch_size)
        if beam_size > 0:
            self.beam_size = beam_size
        print(f"加载Whisper模型: {self.model_size}")
        print(f"device = {self.device}")  
        print(f"{self.model_size, self.device, self.compute_type, self.opts}")
        if self.device == 'cpu' :
            self.model = faster_whisper.WhisperModel(self.model_size, device = self.device, compute_type=self.compute_type)
        else:      
            self.model = faster_whisper.WhisperModel(self.model_size,
                                        device = self.device,
                                        device_index=self.device_index,
                                        compute_type=self.compute_type)
    def transcribe(self, audio_path: str) -> str:
        """将音频文件转录为文本"""
        # 确保音频文件存在
        self.before_transcribe(audio_path)
        print(f"get audio data: {audio_path}")
        print(f"batch size = {self.batch_size}")
        audio = faster_whisper.decode_audio(audio_path)
        print("load audio success")
        segments, info = self.model.transcribe(
            audio,
            beam_size=self.beam_size,
            batch_size=self.batch_size
        )
        segmentList = []
        for segment in segments:
            # print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
            segmentList.append(segment)
        # format result
        result = {}
        result["language"] = info.language
        result["segments"] = segmentList
        return result
    
    
        