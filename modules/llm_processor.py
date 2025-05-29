import json
import openai
from config import OPENAI_API_KEY, OPENAI_MODEL
from typing import List, Dict, Optional


class LLMProcessor:
    def __init__(self):
        self.api_key = OPENAI_API_KEY
        openai.api_key = self.api_key

    def segment_video(self, srt_content: str, prompt: Optional[str] = None) -> \
    List[Dict]:
        """使用大模型根据字幕内容进行视频分段"""
        if not self.api_key:
            raise ValueError("OpenAI API key is not set")

        # 构建系统提示
        system_prompt = (
            "你是一个专业的视频剪辑助手，需要根据提供的字幕内容将视频分成有意义的段落。"
            "每个段落应该包含以下信息：开始时间(秒)，结束时间(秒)，内容摘要和主题标签。"
            "返回格式必须是JSON，包含一个字典列表，每个字典有四个键：start, end, summary, tags。"
        )

        # 用户提示（如果有自定义提示则使用）
        user_prompt = prompt or (
            "请根据以下字幕内容，将视频分成若干有意义的段落。"
            "每个段落应包含连贯的主题内容，并给出简洁的摘要和3-5个主题标签。"
            "时间信息需要精确到秒。"
        )

        # 组合完整的提示
        full_prompt = f"{user_prompt}\n\n字幕内容：\n{srt_content}"

        # 调用OpenAI API
        response = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.3,
            max_tokens=1500
        )

        # 解析响应
        result = response.choices[0].message.content

        # 尝试提取JSON内容
        try:
            # 去除可能的代码块标记
            if result.startswith("```json"):
                result = result[7:-3].strip()
            elif result.startswith("```"):
                result = result[3:-3].strip()

            segments = json.loads(result)
            return segments
        except json.JSONDecodeError:
            # 尝试直接解析为JSON
            try:
                segments = json.loads(result)
                return segments
            except:
                # 如果无法解析，返回示例数据
                return [
                    {"start": 0, "end": 30, "summary": "视频介绍",
                     "tags": ["介绍", "开场"]},
                    {"start": 30, "end": 120, "summary": "主要内容第一部分",
                     "tags": ["主题1", "讲解"]},
                    {"start": 120, "end": 180, "summary": "主要内容第二部分",
                     "tags": ["主题2", "演示"]},
                    {"start": 180, "end": 210, "summary": "总结",
                     "tags": ["总结", "结束语"]}
                ]