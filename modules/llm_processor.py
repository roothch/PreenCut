import json
import os

from openai import OpenAI
from config import LLM_MODEL_OPTIONS
from typing import List, Dict, Optional


class LLMProcessor:
    def __init__(self, llm_model: str):
        for model in LLM_MODEL_OPTIONS:
            if model['label'] == llm_model:
                self.api_key = os.getenv(model['api_key_env_name'])
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url=model['base_url'],
                )
                self.model = model['model']
                self.temperature = model.get('temperature', 0.3)
                self.max_tokens = model.get('max_tokens', 4096)
                break

        if not hasattr(self, 'client'):
            raise ValueError(
                f"Unsupported LLM model: {llm_model}. Available models: "
                f"{', '.join([m['label'] for m in LLM_MODEL_OPTIONS])}"
            )

    def segment_video(self, subtitles: str,
                      prompt: Optional[str] = None) -> \
            List[Dict]:
        """使用大模型根据字幕内容进行视频分段"""
        if not self.api_key:
            raise ValueError("OpenAI API key is not set")

        # 构建系统提示
        system_prompt = (
            "你是一个专业的视频剪辑助手，需要根据提供的字幕内容和用户要求将视频处理成片段。"
            "每个片段应该包含以下信息：开始时间(秒)，结束时间(秒)，一句话的内容摘要和1-3个主题标签。"
            "不同片段的长度不要相差太多，单个片段最长不要超过总长度的30%。"
            "返回格式必须是JSON，包含一个字典列表，每个字典有四个键：start, end, summary, tags。"
        )

        # 用户提示（如果有自定义提示则使用）
        user_prompt = prompt or (
            "请根据以下字幕内容，将视频分成不超过10个有意义的片段。"
            "不同片段的长度不要相差太多，单个片段最长尽量不要超过总长度的30%。"
            "每个片段应包含连贯的主题内容，并给出一句话摘要和1-3个主题标签。"
            "时间信息需要精确到秒。"
        )

        # 组合完整的提示
        full_prompt = f"{user_prompt}\n\n字幕内容：\n{subtitles}"

        # 调用OpenAI API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_prompt}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens
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
                print("大模型返回的数据", result)
                segments = json.loads(result)
                return segments
            except:
                raise ValueError(
                    f"处理大模型结果出错, 大模型返回:{result}"
                )
