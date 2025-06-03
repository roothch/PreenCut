import os
import time
import gradio as gr
from config import (
    UPLOAD_FOLDER,
    TEMP_FOLDER,
    OUTPUT_FOLDER,
    ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE,
    SPEECH_RECOGNITION_MODEL,
    WHISPERX_MODEL_SIZE
)
from modules.processing_queue import ProcessingQueue
from modules.video_processor import VideoProcessor
from typing import List, Dict, Tuple, Optional
import subprocess

# 全局实例
processing_queue = ProcessingQueue()


def check_uploaded_file(file) -> str:
    """检查上传的文件是否符合要求"""
    filename = os.path.basename(file.name)

    # 检查文件大小
    file_size = os.path.getsize(file.name)
    if file_size > MAX_FILE_SIZE:
        raise gr.Error(f"文件大小超过限制 ({file_size} > {MAX_FILE_SIZE})")

    # 检查文件格式
    ext = os.path.splitext(filename)[1][1:].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise gr.Error(f"不支持的文件格式: {ext}")

    return file.name


def process_files(files: List, prompt: Optional[str] = None,
                  model_size: Optional[str] = None) -> Tuple[str, Dict]:
    """处理上传的文件"""
    # 保存文件
    saved_paths = []
    for file in files:
        saved_paths.append(check_uploaded_file(file))

    # 创建任务ID
    task_id = f"task_{int(time.time() * 1000)}"

    # 添加到处理队列
    processing_queue.add_task(task_id, saved_paths, prompt, model_size)

    return task_id, {"status": "已加入队列，请稍候..."}


def check_status(task_id: str) -> Dict:
    """检查任务状态"""
    result = processing_queue.get_result(task_id)

    if result["status"] == "completed":
        # 整理结果以便显示
        display_result = []
        for file_result in result["result"]:
            segments = []
            for seg in file_result["segments"]:
                segments.append([file_result["filename"],
                                 f"{seg['start']:.1f}秒",
                                 f"{seg['end']:.1f}秒",
                                 f"{seg['end'] - seg['start']:.1f}秒",
                                 seg["summary"],
                                 ", ".join(seg["tags"]) if isinstance(
                                     seg["tags"], list) else seg["tags"]])
                # segments.append({
                #     "文件名": file_result["filename"],
                #     "开始时间": f"{seg['start']:.1f}秒",
                #     "结束时间": f"{seg['end']:.1f}秒",
                #     "时长": f"{seg['end'] - seg['start']:.1f}秒",
                #     "内容摘要": seg["summary"],
                #     "标签": ", ".join(seg["tags"]) if isinstance(seg["tags"],
                #                                                  list) else seg[
                #         "tags"]
                # })

            display_result.extend(segments)

        print(display_result)

        # return {
        #     "status": "处理完成",
        #     "result": display_result,
        #     "raw_result": result["result"]
        # }
        return (
            {"status": "处理完成", "raw_result": result["result"],
             "result": display_result, },
            display_result,
            display_result
        )

    elif result["status"] == "error":
        return (
            {"status": f"错误: {result.get('error', '未知错误')}"},
            [], []
        )
        # return {"status": f"错误: {result.get('error', '未知错误')}"}

    return (
        {"status": "处理中..."},
        [],
        []
    )


def clip_and_download(raw_result: Dict, selected_segments: List[Dict]) -> str:
    """剪辑并下载选择的片段"""
    if not raw_result or "raw_result" not in raw_result:
        raise gr.Error("无效的处理结果")

    # 组织文件分段
    file_segments = {}
    for file_data in raw_result["raw_result"]:
        file_segments[file_data["filename"]] = file_data["segments"]

    # 收集用户选择的分段
    selected_clips = []
    for seg in selected_segments:
        filename = seg["文件名"]
        start = float(seg["开始时间"].replace("秒", ""))
        end = float(seg["结束时间"].replace("秒", ""))

        # 找到对应的原始分段
        for original_seg in file_segments[filename]:
            if abs(original_seg["start"] - start) < 0.5 and abs(
                    original_seg["end"] - end) < 0.5:
                selected_clips.append({
                    "filename": filename,
                    "start": original_seg["start"],
                    "end": original_seg["end"],
                    "filepath": os.path.join(UPLOAD_FOLDER, filename)
                })
                break

    # 按文件分组
    clips_by_file = {}
    for clip in selected_clips:
        if clip["filename"] not in clips_by_file:
            clips_by_file[clip["filename"]] = []
        clips_by_file[clip["filename"]].append({
            "start": clip["start"],
            "end": clip["end"]
        })

    # 处理每个文件
    output_files = []
    for filename, segments in clips_by_file.items():
        input_path = os.path.join(UPLOAD_FOLDER, filename)
        output_path = os.path.join(OUTPUT_FOLDER, f"clipped_{filename}")
        VideoProcessor.clip_video(input_path, segments, output_path)
        output_files.append(output_path)

    # 如果只有一个文件，直接返回
    if len(output_files) == 1:
        return output_files[0]

    # 合并多个文件
    combined_path = os.path.join(OUTPUT_FOLDER, "combined_output.mp4")

    # 创建文件列表
    with open(os.path.join(TEMP_FOLDER, "combine_list.txt"), 'w') as f:
        for file in output_files:
            f.write(f"file '{file}'\n")

    # 合并视频
    cmd = [
        'ffmpeg', '-f', 'concat', '-safe', '0',
        '-i', os.path.join(TEMP_FOLDER, "combine_list.txt"),
        '-c', 'copy', combined_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)

    return combined_path


def reanalyze_with_prompt(raw_result: Dict, new_prompt: str) -> Dict:
    """使用新的提示重新分析"""
    if not raw_result or "raw_result" not in raw_result:
        raise gr.Error("无效的处理结果")

    # 使用新提示重新处理
    from modules.llm_processor import LLMProcessor
    llm = LLMProcessor()
    updated_results = []

    for file_data in raw_result["raw_result"]:
        new_segments = llm.segment_video(file_data["align_result"], new_prompt)
        updated_results.append({
            "filename": file_data["filename"],
            "align_result": file_data["align_result"],
            "segments": new_segments
        })

    # 整理显示结果
    display_result = []
    for file_result in updated_results:
        segments = []
        for seg in file_result["segments"]:
            segments.append({
                "文件名": file_result["filename"],
                "开始时间": f"{seg['start']:.1f}秒",
                "结束时间": f"{seg['end']:.1f}秒",
                "时长": f"{seg['end'] - seg['start']:.1f}秒",
                "内容摘要": seg["summary"],
                "标签": ", ".join(seg["tags"]) if isinstance(seg["tags"],
                                                             list) else seg[
                    "tags"]
            })
        display_result.extend(segments)

    return {
        "status": "重新分析完成",
        "result": display_result,
        "raw_result": updated_results
    }


def create_gradio_interface():
    """创建Gradio界面"""
    with gr.Blocks(title="AI视频剪辑工具", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🎬 AI智能视频剪辑工具")
        gr.Markdown(
            "上传视频/音频文件，AI将自动识别内容、生成字幕、智能分段，并允许您选择片段进行剪辑。")

        with gr.Row():
            with gr.Column(scale=2):
                file_upload = gr.Files(
                    label="上传视频/音频文件",
                    file_count="multiple"
                )

                with gr.Accordion("高级设置", open=False):
                    model_size = gr.Dropdown(
                        choices=["large-v2", "large-v3", "large", "medium",
                                 "small", "base", "tiny"],
                        value=WHISPERX_MODEL_SIZE,
                        label="语音识别模型大小",
                        visible=(SPEECH_RECOGNITION_MODEL == 'whisperx')
                    )

                prompt_input = gr.Textbox(
                    label="自定义分析提示 (可选)",
                    placeholder="例如：找出所有关于产品演示的片段",
                    lines=2
                )
                process_btn = gr.Button("开始处理", variant="primary")

                with gr.Row():
                    status_display = gr.JSON(label="处理状态")
                    task_id = gr.Textbox(visible=False)

                raw_result = gr.JSON(visible=False)

            with gr.Column(scale=3):
                with gr.Tab("分析结果"):
                    result_table = gr.Dataframe(
                        headers=["文件名", "开始时间", "结束时间", "时长",
                                 "内容摘要", "标签"],
                        datatype=["str", "str", "str", "str", "str", "str"],
                        interactive=False,
                        wrap=True
                    )

                with gr.Tab("重新分析"):
                    new_prompt = gr.Textbox(
                        label="输入新的分析提示",
                        placeholder="例如：找出所有包含技术术语的片段",
                        lines=2
                    )
                    reanalyze_btn = gr.Button("重新分析", variant="secondary")

                with gr.Tab("剪辑选项"):
                    segment_selection = gr.Dataframe(
                        headers=["文件名", "开始时间", "结束时间", "时长",
                                 "内容摘要", "标签"],
                        datatype=["str", "str", "str", "str", "str", "str"],
                        interactive=True,
                        wrap=True,
                        type="array",
                        label="选择要保留的片段 (按住Ctrl可多选)"
                    )
                    clip_btn = gr.Button("剪辑并下载", variant="primary")
                    download_output = gr.File(label="下载剪辑结果")

        # 定时器，用于轮询状态
        timer = gr.Timer(2, active=False)
        timer.tick(check_status, task_id,
                   outputs=[status_display, result_table, segment_selection])

        # 事件处理
        process_btn.click(
            process_files,
            inputs=[file_upload, prompt_input, model_size],
            outputs=[task_id, status_display]
        ).then(
            lambda: gr.Timer(active=True),
            inputs=None,
            outputs=timer,
            show_progress="hidden"
        ).then(
            lambda x: x,
            inputs=status_display,
            outputs=raw_result
        ).then(
            lambda x: x.get("result", []) if x and "result" in x else [],
            inputs=status_display,
            outputs=result_table
        ).then(
            lambda x: x.get("result", []) if x and "result" in x else [],
            inputs=status_display,
            outputs=segment_selection
        )

        reanalyze_btn.click(
            reanalyze_with_prompt,
            inputs=[raw_result, new_prompt],
            outputs=status_display
        ).then(
            lambda x: x,
            inputs=status_display,
            outputs=raw_result
        ).then(
            lambda x: x.get("result", []) if x and "result" in x else [],
            inputs=status_display,
            outputs=result_table
        ).then(
            lambda x: x.get("result", []) if x and "result" in x else [],
            inputs=status_display,
            outputs=segment_selection
        )

        clip_btn.click(
            clip_and_download,
            inputs=[raw_result, segment_selection],
            outputs=download_output
        )

        return app
