import faster_whisper

from modules.speech_recognizers.speech_recognizer import SpeechRecognizer


class FasterWhisperSpeechRecognizer(SpeechRecognizer):
    beam_size = 5

    def __init__(
            self,
            model_size,
            device,
            device_index,
            compute_type,
            batch_size=16,
            beam_size=5,
    ):
        super().__init__(model_size, device, device_index=device_index,
                         compute_type=compute_type,
                         batch_size=batch_size)
        if beam_size > 0:
            self.beam_size = beam_size
        print(f"加载Whisper模型: {self.model_size}")
        print(f"device = {self.device}")
        print(f"{self.model_size, self.device, self.compute_type, self.opts}")
        if self.device == 'cpu':
            self.model = faster_whisper.WhisperModel(self.model_size,
                                                     device=self.device,
                                                     compute_type=self.compute_type)
        else:
            self.model = faster_whisper.WhisperModel(self.model_size,
                                                     device=self.device,
                                                     device_index=self.device_index,
                                                     compute_type=self.compute_type)

    def transcribe(self, audio_path: str):
        """将音频文件转录为文本"""
        # 确保音频文件存在
        self.before_transcribe(audio_path)
        print(f"get audio data: {audio_path}")
        print(f"batch size = {self.batch_size}")
        audio = faster_whisper.decode_audio(audio_path)
        print("load audio success")
        segments, info = self.model.transcribe(
            audio,
            initial_prompt="Add punctuation after end of each line. 就比如說，我要先去吃飯。Segment at end of each sentence.",
            word_timestamps=True,
            beam_size=self.beam_size
        )
        segment_list = []
        for segment in segments:
            segment_list.append({
                'start': segment.start,
                'end': segment.end,
                'text': segment.text,
                'words': segment.words if segment.words else []
            })
        # format result
        result = {"language": info.language, "segments": segment_list}
        return result
