import os
import torch


def get_available_gpus():
    """获取所有可用的GPU设备"""
    if torch.cuda.is_available():
        return list(range(torch.cuda.device_count()))
    return []


def get_device_config():
    """设置设备配置"""
    gpus = get_available_gpus()

    # 检查环境变量是否指定了GPU
    cuda_visible = os.getenv('CUDA_VISIBLE_DEVICES', '')
    if cuda_visible:
        try:
            # 解析环境变量中的GPU索引
            selected_gpus = [int(x.strip()) for x in cuda_visible.split(',') if
                             x.strip()]
            return 'cuda', selected_gpus
        except ValueError:
            pass

    # 如果没有指定但检测到GPU
    if gpus:
        return 'cuda', gpus

    # 默认使用CPU
    return 'cpu', []


# 文件上传配置
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = ['mp4', 'avi', 'mov', 'mkv', 'mp3', 'wav', 'flac']
MAX_FILE_SIZE = 5 * 1024 * 1024 * 1024  # 5GB

# 临时文件夹
TEMP_FOLDER = "temp"

# 输出文件夹
OUTPUT_FOLDER = "output"

# 语音识别模型配置
SPEECH_RECOGNITION_MODEL = 'whisperx'
DEVICE_TYPE, AVAILABLE_GPUS = get_device_config()
# WhisperX配置
WHISPERX_MODEL_SIZE = 'large-v2'  # 模型大小 (tiny, base, small, medium, large, large-v2, large-v3)
WHISPERX_DEVICE = DEVICE_TYPE
WHISPERX_GPU_IDS = AVAILABLE_GPUS
WHISPERX_COMPUTE_TYPE = 'float16' if WHISPERX_DEVICE == 'cuda' else 'float32'  # float16, float32, int8
WHISPERX_BATCH_SIZE = 16  # 批处理大小

# 语音文字对齐模型
ALIGNMENT_MODEL = 'whisperx'  # 使用的对齐模型

# OpenAI配置
OPENAI_BASE_URL = "https://api.lkeap.cloud.tencent.com/v1"  # OpenAI API base_url
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # 从环境变量获取
OPENAI_MODEL = "deepseek-v3-0324"  # 使用模型

# 创建必要的目录
for folder in [UPLOAD_FOLDER, TEMP_FOLDER, OUTPUT_FOLDER]:
    os.makedirs(folder, exist_ok=True)
