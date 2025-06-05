import shutil
import os

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