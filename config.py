import os

# 文件上传配置
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = ['mp4', 'avi', 'mov', 'mkv', 'mp3', 'wav', 'flac']
MAX_FILE_SIZE = 5 * 1024 * 1024 * 1024  # 5GB

# 临时文件夹
TEMP_FOLDER = "temp"

# 输出文件夹
OUTPUT_FOLDER = "output"

# 语音识别模型配置
SPEECH_RECOGNITION_MODEL = 'faster-whisper'  # 'whisper', 'funasr' 或 'faster-whisper'
WHISPER_MODEL_SIZE = 'base'  # whisper模型大小 (tiny, base, small, medium, large)
FASTER_WHISPER_MODEL = 'large-v3'  # faster-whisper模型 (tiny, base, small, medium, large, large-v2, large-v3)
FASTER_WHISPER_DEVICE = 'cuda' if os.getenv('CUDA_VISIBLE_DEVICES') else 'cpu'  # cuda 或 cpu
FASTER_WHISPER_COMPUTE_TYPE = 'float16' if FASTER_WHISPER_DEVICE == 'cuda' else 'float32'  # float16, float32, int8
FUNASR_MODEL_NAME = 'paraformer-zh'  # funasr模型名称

# OpenAI配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # 从环境变量获取
OPENAI_MODEL = "gpt-3.5-turbo"  # 使用模型

# 创建必要的目录
for folder in [UPLOAD_FOLDER, TEMP_FOLDER, OUTPUT_FOLDER]:
    os.makedirs(folder, exist_ok=True)