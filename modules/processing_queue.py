from queue import Queue
from threading import Thread, Lock
import os
from modules.speech_recognition import SpeechRecognizer
from modules.text_aligner import TextAligner
from modules.llm_processor import LLMProcessor
from modules.video_processor import VideoProcessor
from config import SPEECH_RECOGNITION_MODEL, WHISPERX_MODEL_SIZE
from typing import List, Dict, Optional


class ProcessingQueue:
    def __init__(self):
        self.queue = Queue()
        self.lock = Lock()
        self.results = {}
        self.worker = Thread(target=self._process_queue, daemon=True)
        self.worker.start()

    def add_task(self, task_id: str, files: List[str], prompt: Optional[str],
                 model_size: Optional[str] = None):
        """添加任务到队列"""
        with self.lock:
            self.results[task_id] = {
                "status": "queued",
                "files": files,
                "prompt": prompt,
                "model_size": model_size
            }
        self.queue.put(task_id)

    def _process_queue(self):
        """处理队列中的任务"""
        while True:
            task_id = self.queue.get()
            try:
                with self.lock:
                    self.results[task_id]["status"] = "processing"

                # 获取任务数据
                with self.lock:
                    files = self.results[task_id]["files"]
                    prompt = self.results[task_id]["prompt"]
                    model_size = self.results[task_id].get("model_size")

                # 如果指定了模型大小，临时更新配置
                if model_size and SPEECH_RECOGNITION_MODEL == 'whisperx':
                    global WHISPERX_MODEL_SIZE
                    original_model = WHISPERX_MODEL_SIZE
                    WHISPERX_MODEL_SIZE = model_size

                # 处理每个文件
                file_results = []
                recognizer = SpeechRecognizer()
                llm = LLMProcessor()

                for file_path in files:
                    # 提取音频（如果是视频）
                    if file_path.lower().endswith(
                            ('.mp4', '.avi', '.mov', '.mkv')):
                        audio_path = VideoProcessor.extract_audio(file_path)
                    else:
                        audio_path = file_path

                    # 语音识别
                    print(f"开始语音识别: {file_path}")
                    result = recognizer.transcribe(audio_path)
                    print(f"语音识别完成，文本长度: {len(result['segments'])}")
                    print(result['segments'], result['language'])

                    # 文本对齐
                    print("开始文本对齐...")
                    aligner = TextAligner(result['language'])
                    srt_content = aligner.align(segments, audio_path)
                    total_paragraphs = len(srt_content.split('\n\n'))
                    print(
                        f"生成SRT字幕，段落数: {total_paragraphs}")

                    # 调用大模型进行分段
                    print("调用大模型进行分段...")
                    segments = llm.segment_video(srt_content, prompt)
                    print(f"分段完成，段数: {len(segments)}")

                    # 保存结果
                    file_results.append({
                        "filename": os.path.basename(file_path),
                        "srt": srt_content,
                        "segments": segments
                    })

                # 恢复原始模型设置
                if model_size and SPEECH_RECOGNITION_MODEL == 'whisperx':
                    WHISPERX_MODEL_SIZE = original_model

                # 更新结果
                with self.lock:
                    self.results[task_id]["status"] = "completed"
                    self.results[task_id]["result"] = file_results

            except Exception as e:
                import traceback
                error_msg = traceback.format_exc()
                print(f"任务处理错误: {error_msg}")

                # 恢复原始模型设置
                if 'model_size' in locals() and model_size and SPEECH_RECOGNITION_MODEL == 'whisperx':
                    WHISPERX_MODEL_SIZE = original_model

                with self.lock:
                    self.results[task_id]["status"] = "error"
                    self.results[task_id]["error"] = str(e)
            finally:
                self.queue.task_done()

    def get_result(self, task_id: str) -> Dict:
        """获取任务结果"""
        with self.lock:
            return self.results.get(task_id, {"status": "not_found"})
