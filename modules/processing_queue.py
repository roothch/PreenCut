from queue import Queue
from threading import Thread, Lock
import os
import time
from modules.speech_recognition import SpeechRecognizer
from modules.text_aligner import TextAligner
from modules.llm_processor import LLMProcessor
from modules.video_processor import VideoProcessor
from config import SPEECH_RECOGNITION_MODEL, WHISPERX_MODEL_SIZE, \
    ENABLE_ALIGNMENT
from typing import List, Dict, Optional


class ProcessingQueue:
    def __init__(self):
        self.queue = Queue()
        self.lock = Lock()
        self.results = {}
        self.result_ttl = 24 * 60 * 60  # 结果保留时间，单位为秒（默认24小时）
        self.max_results = 100  # 最大保留结果数
        self.worker = Thread(target=self._process_queue, daemon=True)
        self.worker.start()
        # 启动清理线程
        self.cleanup_worker = Thread(target=self._cleanup_results, daemon=True)
        self.cleanup_worker.start()

    def add_task(self, task_id: str, files: List[str], prompt: Optional[str],
                 model_size: Optional[str] = None):
        """添加任务到队列"""
        with self.lock:
            self.results[task_id] = {
                "status": "queued",
                "files": files,
                "prompt": prompt,
                "model_size": model_size,
                "timestamp": time.time()  # 记录任务添加时间
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
                        audio_path = VideoProcessor.extract_audio(file_path,
                                                                  task_id)
                    else:
                        audio_path = file_path

                    # 语音识别
                    print(f"开始语音识别: {file_path}")
                    result = recognizer.transcribe(audio_path)
                    print(
                        f"语音识别完成，segments个数: {len(result['segments'])}")
                    print(result['segments'], result['language'])

                    if ENABLE_ALIGNMENT:
                        # 文本对齐
                        print("开始文本对齐...")
                        aligner = TextAligner(result['language'])
                        result = aligner.align(result["segments"], audio_path)
                        print(
                            f"对齐结果: {result}")

                    # 调用大模型进行分段
                    print("调用大模型进行分段...")
                    segments = llm.segment_video(result["segments"], prompt)
                    print(f"大模型分段完成，段数: {len(segments)}")
                    print(f"大模型分段完成，分段结果: {segments}")

                    # 保存结果
                    file_results.append({
                        "filename": os.path.basename(file_path),
                        "align_result": result,
                        "segments": segments,
                        "filepath": file_path
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
            result = self.results.get(task_id, {"status": "not_found"})
            if result["status"] != "not_found" and result["status"] != "queued":
                # 更新访问时间，避免被清理
                result["last_accessed"] = time.time()
            return result

    def _cleanup_results(self):
        """定期清理过期或过多的结果"""
        while True:
            time.sleep(1 * 60 * 60)  # 每1小时清理一次
            with self.lock:
                current_time = time.time()
                # 按时间排序的结果列表
                sorted_results = sorted(
                    self.results.items(),
                    key=lambda x: x[1].get("last_accessed", x[1]["timestamp"])
                )

                # 移除过期结果
                for task_id, result in list(sorted_results):
                    age = current_time - result.get("last_accessed",
                                                    result["timestamp"])
                    if age > self.result_ttl:
                        self.results.pop(task_id, None)

                # 限制最大结果数
                if len(self.results) > self.max_results:
                    # 删除最旧的结果
                    to_remove = len(self.results) - self.max_results
                    for task_id, _ in sorted_results[:to_remove]:
                        self.results.pop(task_id, None)
