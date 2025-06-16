import os
import torch

# 设置Gradio临时目录
# os.environ['GRADIO_TEMP_DIR'] = '/data/tmp/gradio'


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
ALLOWED_EXTENSIONS = ['mp4', 'avi', 'mov', 'mkv', 'ts', 'mxf', 'mp3', 'wav',
                      'flac']
MAX_FILE_SIZE = 10 * 1024 * 1024 * 1024  # 10GB
MAX_FILE_NUMBERS = 10  # 最大文件数量

# 临时文件夹
TEMP_FOLDER = "temp"

# 输出文件夹
OUTPUT_FOLDER = "output"

# 语音识别模型配置
SPEECH_RECOGNIZER_TYPE = 'faster_whisper'

DEVICE_TYPE, AVAILABLE_GPUS = get_device_config()
# WhisperX配置
WHISPERX_MODEL_SIZE = 'large-v2'  # 模型大小 (tiny, base, small, medium, large, large-v2, large-v3)
WHISPERX_DEVICE = DEVICE_TYPE
WHISPERX_GPU_IDS = AVAILABLE_GPUS
WHISPERX_COMPUTE_TYPE = 'float16' if WHISPERX_DEVICE == 'cuda' else 'float32'  # float16, float32, int8
WHISPERX_BATCH_SIZE = 5  # 批处理大小

# 语音文字对齐模型
ENABLE_ALIGNMENT = False  # 是否启用对齐
ALIGNMENT_MODEL = 'whisperx'  # 使用的对齐模型

# OpenAI API配置
LLM_MODEL_OPTIONS = [
    {
        "model": "deepseek-v3-0324",
        "base_url": "https://api.lkeap.cloud.tencent.com/v1",
        "api_key_env_name": "DEEPSEEK_V3_API_KEY",
        "label": "DeepSeek-V3-0324",
        "max_tokens": 4096,
        "temperature": 0.3,
    },
    {
        "model": "doubao-1-5-pro-32k-250115",
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "api_key_env_name": "DOUBAO_1_5_PRO_API_KEY",
        "label": "豆包",
        "max_tokens": 4096,
        "temperature": 0.3,
    }
]

# 创建必要的目录
for folder in [TEMP_FOLDER, OUTPUT_FOLDER]:
    os.makedirs(folder, exist_ok=True)
