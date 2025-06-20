from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr
import uvicorn


from web.gradio_ui import create_gradio_interface
from web.api import router as api_router
import config

import logging

# 屏蔽访问日志
block_endpoints = "/"


class LogFilter(logging.Filter):
    def filter(self, record):
        if record.args and len(record.args) >= 3:
            if str(record.args[2]).startswith(block_endpoints):
                return False
        return True


uvicorn_logger = logging.getLogger("uvicorn.access")
uvicorn_logger.addFilter(LogFilter())


app = FastAPI()

# 跨域中间件
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# 添加api路由
app.include_router(api_router)

# 检查GPU可用性
try:
    import torch

    if torch.cuda.is_available():
        print("✅ 检测到GPU可用，将使用GPU加速")
    else:
        print("⚠️ 未检测到GPU，将使用CPU运行")
except ImportError:
    print("⚠️ 无法导入torch，GPU状态未知")

# 打印当前配置
print("\n当前配置:")
print(f"  语音识别处理模块: {config.SPEECH_RECOGNIZER_TYPE}")
print(f"  模型大小: {config.WHISPER_MODEL_SIZE}")
print(f"  计算设备: {config.WHISPER_DEVICE}")
print(f"  计算类型: {config.WHISPER_COMPUTE_TYPE}")
print(f"  使用GPU: {config.WHISPER_GPU_IDS}")
print(f"  批处理大小: {config.WHISPER_BATCH_SIZE}")

# 创建Gradio界面
gradio_app = create_gradio_interface()
app = gr.mount_gradio_app(app, gradio_app, path="/web")

if __name__ == "__main__":
    # 启动应用
    uvicorn.run(app, host="localhost", port=7860)
