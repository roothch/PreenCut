import shutil
import os
import subprocess
import json
import csv
import re
from modules.subtitles_processor import SubtitlesProcessor
from modules.subtitles_processor import format_timestamp
import torch
import gc


def clear_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def generate_safe_filename(filename, max_length=100):
    """生成安全的文件名，确保不超过指定长度"""
    safe_filename = filename
    for c in filename:
        if not c.isalnum() and c not in ['_', '.']:
            safe_filename = safe_filename.replace(c, '_')
    return safe_filename[:max_length]


def clear_directory_fast(directory_path):
    """通过重建目录快速清空内容"""
    shutil.rmtree(directory_path)
    os.makedirs(directory_path, exist_ok=True)


def seconds_to_hhmmss(seconds):
    """将秒数转换为 HH:MM:SS 格式"""
    seconds = float(seconds)  # 确保输入是浮点数
    hours = int(seconds // 3600)
    remaining_seconds = seconds % 3600
    minutes = int(remaining_seconds // 60)
    seconds_remaining = int(round(remaining_seconds % 60))  # 四舍五入秒数

    # 格式化为两位数，不足补零
    hhmmss = f"{hours:02d}:{minutes:02d}:{seconds_remaining:02d}"
    return hhmmss


def hhmmss_to_seconds(time_str):
    """将 HH:MM:SS 或 MM:SS 格式的时间转换为秒数（浮点数）"""
    parts = time_str.split(':')

    # 处理不同长度的情况（如 MM:SS 或 HH:MM:SS）
    if len(parts) == 2:  # MM:SS 格式
        minutes, seconds = map(float, parts)
        total_seconds = minutes * 60 + seconds
    elif len(parts) == 3:  # HH:MM:SS 格式
        hours, minutes, seconds = map(float, parts)
        total_seconds = hours * 3600 + minutes * 60 + seconds
    else:
        raise ValueError("时间格式应为 HH:MM:SS 或 MM:SS")

    return total_seconds


def get_audio_codec(input_file):
    """使用 ffprobe 检测音频编码"""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a:0",  # 只检查第一个音频流
        "-show_entries", "stream=codec_name",
        "-of", "json",
        input_file
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True)
    info = json.loads(result.stdout)
    return info["streams"][0]["codec_name"]


def write_to_srt(align_result, output_dir, max_line_length,
                 filename='字幕.srt'):
    '''
    :param align_result: whisperx对齐后的视/音频转文本结果
    :param output_dir: srt文件的保存目录
    :param max_line_length: 单组字幕的最大长度
    :param filename: srt文件名
    :return: srt文件所在目录
    '''

    # 确保目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 避免处理重名文件
    files_in_dir = os.listdir(output_dir)
    while filename in files_in_dir:
        filename = filename.split('.')[0] + '_duplicate' + '.srt'

    # 构造完整文件路径
    file_path = os.path.join(output_dir, filename)

    if align_result["language"] not in ["zh", "en"]:
        subtitles_processor = SubtitlesProcessor(align_result["segments"],
                                                 align_result["language"],
                                                 max_line_length=max_line_length,
                                                 min_char_length_splitter=5)
        subtitles_processor.save(file_path, advanced_splitting=True)
    else:
        # 定义标点符号列表，但不包含小数点
        punctuations = ['，', '、', '。', '！', ',', '!', '?', '？', ';', '；']

        with open(file_path, "w", encoding="utf-8") as f:
            srt_index = 1
            all_words = []

            for segment in align_result.get('segments', []):
                all_words.extend(segment['words'])

            if not all_words:
                print("文本对齐数据异常")
                return file_path

            current_group = []
            current_length = 0  # 跟踪当前字幕的长度(汉字/单词)

            for i, word in enumerate(all_words):
                text = word.get('word', '')
                current_group.append(word)

                # 计算当前字幕长度
                if align_result["language"] == "zh":
                    current_length += len(text)
                else:
                    current_length += 1

                # 检查是否达到最大长度
                if current_length >= max_line_length:
                    end_time = current_group[-1]["end"]
                    if text in punctuations:
                        current_group.pop()

                    start_time = current_group[0]["start"]
                    text_content = "".join(
                        [w.get('word', '') for w in current_group])

                    f.write(f"{srt_index}\n")
                    f.write(
                        f"{format_timestamp(start_time)} --> {format_timestamp(end_time)}\n")
                    f.write(f"{text_content}\n\n")

                    srt_index += 1
                    current_group = []
                    current_length = 0
                    continue

                # 检查是否为需要换行的标点符号
                elif text in punctuations:
                    end_time = current_group[-1]["end"]
                    current_group.pop()
                    if current_group:
                        start_time = current_group[0]["start"]
                        text_content = "".join(
                            [w.get('word', '') for w in current_group])

                        f.write(f"{srt_index}\n")
                        f.write(
                            f"{format_timestamp(start_time)} --> {format_timestamp(end_time)}\n")
                        f.write(f"{text_content}\n\n")

                        srt_index += 1
                        current_group = []
                        current_length = 0

                # 特殊处理小数点：如果是数字的一部分则不断行
                elif text == '.':
                    # 获取当前组的文本内容
                    current_text = "".join(
                        [w.get('word', '') for w in current_group])

                    # 检查小数点是否是浮点数的一部分
                    if re.search(r'\d\.\d', current_text):
                        # 是浮点数的一部分，不断行
                        continue
                    else:
                        end_time = current_group[-1]["end"]
                        # 不是浮点数的一部分，视为句子结束
                        current_group.pop()
                        if current_group:
                            start_time = current_group[0]["start"]
                            text_content = "".join(
                                [w.get('word', '') for w in current_group])

                            f.write(f"{srt_index}\n")
                            f.write(
                                f"{format_timestamp(start_time)} --> {format_timestamp(end_time)}\n")
                            f.write(f"{text_content}\n\n")

                            srt_index += 1
                            current_group = []
                            current_length = 0

            # 处理最后一组
            if current_group:
                start_time = current_group[0]["start"]
                end_time = current_group[-1]["end"]
                text_content = "".join(
                    [w.get('word', '') for w in current_group])
                f.write(f"{srt_index}\n")
                f.write(
                    f"{format_timestamp(start_time)} --> {format_timestamp(end_time)}\n")
                f.write(f"{text_content}\n\n")

    print(f'已保存字幕文件：{file_path}')

    return file_path


def write_to_csv(display_result: list, output_dir: str,
                 filename: str = "output.csv",
                 header: list = ["文件名", "开始时间", "结束时间", "时长",
                                 "内容摘要",
                                 "标签"]) -> str:
    """
    将 `display_result` 写入 CSV 文件，并返回文件路径。

    Args:
        display_result (list): 要写入的数据（二维列表，每行代表 CSV 的一行）
        output_dir (str): 输出目录
        filename (str, optional): 输出文件名，默认为 "output.csv"

    Returns:
        str: 生成的 CSV 文件路径
    """
    # 确保目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 避免处理重名文件
    suffix = filename.split('.')[1]
    files_in_dir = os.listdir(output_dir)
    while filename in files_in_dir:
        filename = filename.split('.')[0] + '_duplicate' + '.' + suffix

    # 构造完整文件路径
    file_path = os.path.join(output_dir, filename)

    # 写入 CSV 文件
    with open(file_path, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        # 写入表头（可选，如果需要列名可以在这里添加）
        if header:
            writer.writerow(header)

        # 写入数据行
        writer.writerows(display_result)

    return file_path
