import json
import os

from twelvelabs import TwelveLabs
from config import (
    TWELVELABS_MODEL_OPTIONS,
    TWELVELABS_API_KEY_ENV_NAME,
    TWELVELABS_MAX_CLIPS,
    TWELVELABS_INDEX_NAME,
)
from typing import List, Dict, Optional


class TwelveLabsProcessor:
    """使用 TwelveLabs 进行视频分段（可选后端）。

    与 LLMProcessor 不同，本处理器直接分析视频画面与音频：
    - Marengo (search): 当用户提供自然语言提示时，按语义检索相关片段，得到 start/end 时间。
    - Pegasus (analyze): 为每个片段（或整段视频）生成一句话摘要和主题标签。

    返回结构与 LLMProcessor.segment_video 一致：[{start, end, summary, tags}]，
    因此可以无缝接入现有的剪辑/导出流程。
    """

    def __init__(self, llm_model: str):
        for model in TWELVELABS_MODEL_OPTIONS:
            if model['label'] == llm_model:
                self.search_model = model['search_model']
                self.analyze_model = model['analyze_model']
                break
        else:
            # 允许在未显式配置 label 时使用默认模型
            self.search_model = 'marengo3.0'
            self.analyze_model = 'pegasus1.2'

        self.api_key = os.getenv(TWELVELABS_API_KEY_ENV_NAME)
        if not self.api_key:
            raise ValueError(
                f"TwelveLabs API key is not set, "
                f"please set the {TWELVELABS_API_KEY_ENV_NAME} environment variable"
            )
        self.client = TwelveLabs(api_key=self.api_key)
        self.max_clips = TWELVELABS_MAX_CLIPS

    def _ensure_index(self) -> str:
        """复用或创建一个同时包含 Marengo 与 Pegasus 模型的索引。"""
        for index in self.client.indexes.list():
            if index.index_name == TWELVELABS_INDEX_NAME:
                return index.id

        created = self.client.indexes.create(
            index_name=TWELVELABS_INDEX_NAME,
            models=[
                {"model_name": self.search_model,
                 "model_options": ["visual", "audio"]},
                {"model_name": self.analyze_model,
                 "model_options": ["visual", "audio"]},
            ],
        )
        return created.id

    def _index_video(self, index_id: str, video_path: str) -> str:
        """上传并索引视频，返回 video_id（索引过程可能较慢）。"""
        with open(video_path, "rb") as f:
            task = self.client.tasks.create(index_id=index_id, video_file=f)
        task = self.client.tasks.wait_for_done(task_id=task.id, sleep_interval=5)
        if task.status != "ready":
            raise RuntimeError(f"TwelveLabs 索引失败, 状态: {task.status}")
        return task.video_id

    def _pegasus_segments(self, video_id: str,
                          prompt: Optional[str]) -> List[Dict]:
        """用 Pegasus 对整段视频进行分段，得到 [{start, end, summary, tags}]。"""
        instruction = prompt + "。" if prompt else ""
        full_prompt = (
            f"{instruction}"
            "请将这个视频分成不超过"
            f"{self.max_clips}个有意义的片段，每个片段主题连贯。"
            "为每个片段给出开始时间(秒)、结束时间(秒)、一句话内容摘要和1-3个主题标签。"
            "严格返回JSON数组，每个元素包含四个键：start, end, summary, tags(字符串数组)。"
            "只返回JSON，不要其它文字。"
        )
        res = self.client.analyze(
            model_name=self.analyze_model,
            video_id=video_id,
            prompt=full_prompt,
            max_tokens=2048,
        )
        return _parse_segments(res.data)

    def _marengo_windows(self, index_id: str, video_id: str,
                         prompt: str) -> List[tuple]:
        """用 Marengo 语义检索，返回与提示相关的 (start, end) 时间窗。"""
        pager = self.client.search.query(
            index_id=index_id,
            query_text=prompt,
            search_options=["visual", "audio"],
            group_by="clip",
            page_limit=self.max_clips,
        )
        windows = []
        for item in pager:
            if item.video_id != video_id:
                continue
            if item.start is None or item.end is None:
                continue
            windows.append((float(item.start), float(item.end)))
        return windows

    def segment_video(self, video_path: str,
                      prompt: Optional[str] = None) -> List[Dict]:
        """对视频进行语义分段，返回 [{start, end, summary, tags}]。

        与 LLMProcessor.segment_video 不同，这里的第一个参数是视频文件路径，
        因为 TwelveLabs 直接分析视频画面而非字幕文本。

        流程：
        1. 上传并索引视频(同时建立 Marengo 与 Pegasus 模型索引)。
        2. Pegasus(analyze) 生成带摘要和标签的视频分段。
        3. 若提供了提示词，再用 Marengo(search) 语义检索相关时间窗，
           只保留与之重叠的片段，实现按需检索。
        """
        index_id = self._ensure_index()
        video_id = self._index_video(index_id, video_path)

        segments = self._pegasus_segments(video_id, prompt)

        if prompt:
            windows = self._marengo_windows(index_id, video_id, prompt)
            if windows:
                segments = [
                    seg for seg in segments
                    if _overlaps_any(seg["start"], seg["end"], windows)
                ]
        return segments


def _overlaps_any(start: float, end: float, windows: List[tuple]) -> bool:
    """片段 [start, end] 是否与任一检索窗口重叠。"""
    for w_start, w_end in windows:
        if start < w_end and end > w_start:
            return True
    return False


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _parse_segments(text: str) -> List[Dict]:
    """解析整段视频的分段结果。"""
    try:
        segments = json.loads(_strip_code_fence(text))
    except (json.JSONDecodeError, AttributeError):
        raise ValueError(f"处理 TwelveLabs 结果出错, 返回:{text}")

    normalized = []
    for seg in segments:
        tags = seg.get("tags", [])
        if isinstance(tags, str):
            tags = [tags]
        normalized.append({
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "summary": seg.get("summary", ""),
            "tags": tags,
        })
    return normalized
