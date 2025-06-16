from web.gradio_ui import create_gradio_interface
import config

if __name__ == "__main__":
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
    app = create_gradio_interface()

    # 启动应用
    app.launch(
        server_name="localhost",
        server_port=7860,
        share=False
    )