import os
import subprocess
from config import TEMP_FOLDER
from typing import List, Dict


class VideoProcessor:
    @staticmethod
    def extract_audio(video_path: str, task_id: str) -> str:
        """从视频中提取音频"""

        task_temp_dir = os.path.join(TEMP_FOLDER, task_id)
        os.makedirs(task_temp_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        audio_path = os.path.join(task_temp_dir, f"{base_name}.wav")

        cmd = [
            'ffmpeg', '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
            '-y', audio_path
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        return audio_path

    @staticmethod
    def clip_video(input_path: str, segments: List[Dict], output_path: str,
                   task_temp_dir: str):
        """根据分段剪辑视频"""
        # 创建临时文件列表
        clip_list = []

        for i, seg in enumerate(segments):
            clip_path = os.path.join(task_temp_dir, f"clip_{i}.mp4")
            cmd = [
                'ffmpeg', '-i', input_path,
                '-ss', str(seg['start']),
                '-to', str(seg['end']),
                '-c', 'copy', clip_path
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
            clip_list.append(clip_path)

        # 合并剪辑
        with open(os.path.join(task_temp_dir, "file_list.txt"), 'w') as f:
            for clip in clip_list:
                f.write(f"file '{clip}'\n")

        cmd = [
            'ffmpeg', '-f', 'concat', '-safe', '0',
            '-i', os.path.join(task_temp_dir, "file_list.txt"),
            '-c', 'copy', output_path
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)

        # 清理临时文件
        for clip in clip_list:
            os.remove(clip)
        os.remove(os.path.join(task_temp_dir, "file_list.txt"))
