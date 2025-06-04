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