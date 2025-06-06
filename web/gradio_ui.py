import os
import uuid
import time
import random
import zipfile
import gradio as gr
from config import LLM_MODEL_OPTIONS

from config import (
    TEMP_FOLDER,
    OUTPUT_FOLDER,
    ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE,
    SPEECH_RECOGNITION_MODEL,
    WHISPERX_MODEL_SIZE
)
from modules.processing_queue import ProcessingQueue
from modules.video_processor import VideoProcessor
from utils import seconds_to_hhmmss, hhmmss_to_seconds, clear_directory_fast \
    , generate_safe_filename
from typing import List, Dict, Tuple, Optional
import subprocess

# 全局实例
processing_queue = ProcessingQueue()
CHECKBOX_CHECKED = '<span style="display: flex; width: 16px; height: 16px; border: 2px solid blue; background:#4B6BFB ;font-weight: bold;color:white;align-items:center;justify-content:center">✓</span>'
CHECKBOX_UNCHECKED = '<span style="display: flex; width: 16px; height: 16px; border: 2px solid blue;font-weight: bold;color:white;align-items:center;justify-content:center"></span>'


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


def process_files(files: List, llm_model: str,
                  prompt: Optional[str] = None,
                  whisper_model_size: Optional[str] = None) -> Tuple[
    str, Dict]:
    """处理上传的文件"""

    print('当前results', processing_queue.results, flush=True)

    # 保存文件
    saved_paths = []
    for file in files:
        saved_paths.append(check_uploaded_file(file))

    # 创建唯一任务ID
    task_id = f"task_{uuid.uuid4().hex}"

    # 添加到处理队列
    processing_queue.add_task(task_id, saved_paths, llm_model, prompt,
                              whisper_model_size)

    return task_id, {"status": "已加入队列，请稍候..."}


def check_status(task_id: str) -> Tuple[Dict, List, List, gr.Timer]:
    """检查任务状态"""
    result = processing_queue.get_result(task_id)

    if result["status"] == "completed":
        # 整理结果以便显示
        display_result = []
        clip_result = []
        for file_result in result["result"]:
            for seg in file_result["segments"]:
                row = [file_result["filename"],
                       f"{seconds_to_hhmmss(seg['start'])}",
                       f"{seconds_to_hhmmss(seg['end'])}",
                       f"{seconds_to_hhmmss(seg['end'] - seg['start'])}",
                       seg["summary"],
                       ", ".join(seg["tags"]) if isinstance(
                           seg["tags"], list) else seg["tags"]]
                clip_row = row.copy()
                clip_row.insert(0, CHECKBOX_UNCHECKED)  # 添加选择框
                display_result.append(row)
                clip_result.append(clip_row)

        return (
            {"task_id": task_id, "status": "处理完成",
             "raw_result": result["result"],
             "result": display_result, },
            display_result,
            clip_result,
            gr.Timer(active=False)
        )

    elif result["status"] == "error":
        return (
            {"task_id": task_id,
             "status": f"错误: {result.get('error', '未知错误')}"},
            [], [], gr.update()
        )
    elif result["status"] == "queued":
        return (
            {"task_id": task_id,
             "status": f"排队中, 前面还有{processing_queue.get_queue_size() + 1}个任务"},
            [], [], gr.update()
        )

    if task_id:
        return (
            {"task_id": task_id, "status": "处理中...",
             "status_info": result.get("status_info", "")},
            [], [], gr.update()
        )
    else:
        return (
            {"task_id": "", "status": ""},
            [], [], gr.update()
        )


def select_clip(segment_selection: List[List], evt: gr.SelectData) -> List[
    List]:
    """选择剪辑片段"""
    selected_row = segment_selection[evt.index[0]]
    # 切换选择状态
    selected_row[0] = CHECKBOX_CHECKED \
        if selected_row[0] == CHECKBOX_UNCHECKED else CHECKBOX_UNCHECKED
    return segment_selection


def clip_and_download(status_display: Dict,
                      segment_selection: List[List], download_mode: str) -> str:
    """剪辑并下载选择的片段"""
    if not status_display or "raw_result" not in status_display:
        raise gr.Error("无效的处理结果")

    # 获取任务ID用于创建唯一目录
    task_id = status_display.get("task_id",
                                 f"temp_{int(time.time() * 1000)}_{random.randint(1000, 9999)}")
    task_temp_dir = os.path.join(TEMP_FOLDER, task_id)
    task_output_dir = os.path.join(OUTPUT_FOLDER, task_id)

    if os.path.exists(task_output_dir):
        clear_directory_fast(task_output_dir)
    else:
        os.makedirs(task_output_dir, exist_ok=True)
    if os.path.exists(task_temp_dir):
        clear_directory_fast(task_temp_dir)
    else:
        os.makedirs(task_temp_dir, exist_ok=True)

    # 组织文件分段
    file_segments = {}
    for file_data in status_display["raw_result"]:
        file_segments[file_data["filename"]] = {
            "segments": file_data["segments"],
            "filepath": file_data["filepath"],
            "ext": os.path.splitext(file_data["filepath"])[1]  # 获取原始文件扩展名
        }

    selected_segments = [seg for seg in segment_selection if
                         seg[0] == CHECKBOX_CHECKED]

    # 处理"合并成一个文件"的情况
    if download_mode == "合并成一个文件":
        # 检查所有片段格式是否一致
        formats = set()
        for seg in selected_segments:
            filename = seg[1]
            file_ext = file_segments[filename]['ext']
            formats.add(file_ext.lower())

        if len(formats) > 1:
            raise gr.Error(
                "无法合并: 所选片段包含多种格式: " + ", ".join(formats))

    selected_clips = []
    for seg in selected_segments:
        filename = seg[1]
        start = hhmmss_to_seconds(seg[2])
        end = hhmmss_to_seconds(seg[3])

        # 找到对应的原始分段
        for original_seg in file_segments[filename]['segments']:
            if abs(original_seg["start"] - start) < 0.5 and abs(
                    original_seg["end"] - end) < 0.5:
                selected_clips.append({
                    "filename": filename,
                    "start": original_seg["start"],
                    "end": original_seg["end"],
                    "filepath": file_segments[filename]['filepath'],
                    "ext": file_segments[filename]['ext']  # 添加扩展名
                })
                break

    # 按文件分组
    clips_by_file = {}
    for clip in selected_clips:
        if clip["filename"] not in clips_by_file:
            clips_by_file[clip["filename"]] = {
                "filepath": clip["filepath"],
                "ext": clip["ext"],
                "segments": []
            }
        clips_by_file[clip["filename"]]['segments'].append({
            "start": clip["start"],
            "end": clip["end"],
        })

    # 处理每个文件
    output_files = []
    for filename, segments in clips_by_file.items():
        input_path = segments['filepath']
        # 生成安全的目录名(一个文件可能有多个片段，放在以这个文件名为名的目录下)
        safe_filename = generate_safe_filename(filename)
        output_folder = os.path.join(task_output_dir, safe_filename)
        os.makedirs(output_folder, exist_ok=True)
        single_file_clips = VideoProcessor.clip_video(input_path,
                                                      segments['segments'],
                                                      output_folder,
                                                      segments['ext'])
        output_files.extend(single_file_clips)

    # 如果只有一个文件，直接返回
    if len(output_files) == 1:
        return output_files[0]

    # 根据用户选择的模式处理
    if download_mode == "合并成一个文件":
        # 合并多个文件
        ext = clips_by_file[next(iter(clips_by_file))]['ext']  # 获取第一个文件的扩展名
        combined_path = os.path.join(task_output_dir, f"combined_output{ext}")

        # 创建文件列表
        with open(os.path.join(task_temp_dir, "combine_list.txt"), 'w') as f:
            for file in output_files:
                f.write(f"file '../../{file}'\n")

        # 合并视频
        cmd = [
            'ffmpeg', '-f', 'concat', '-safe', '0',
            '-i', os.path.join(task_temp_dir, "combine_list.txt"),
            '-c', 'copy', combined_path
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg error: {e.stderr.decode('utf-8')}")
            raise gr.Error(f"文件合并失败: {str(e)}")

        return combined_path

    # 打包成zip文件
    else:
        # 创建zip文件
        zip_path = os.path.join(task_output_dir, "clipped_segments.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file_path in output_files:
                # 在zip文件中使用相对路径
                arcname = os.path.basename(file_path)
                zipf.write(file_path, arcname)

        return zip_path


def start_reanalyze() -> Dict:
    return {
        'status': '请稍候，正在使用新的提示重新分析...',
    }


def reanalyze_with_prompt(task_id: str, reanalyze_llm_model: str,
                          new_prompt: str) -> Tuple[
    Dict, List[List], List[List]]:
    """使用新的提示重新分析"""
    if not task_id:
        raise gr.Error("无效的任务ID")
    task_result = processing_queue.get_result(task_id)
    if not task_result or "result" not in task_result:
        raise gr.Error("没有可以重新分析的内容")

    if not new_prompt:
        raise gr.Error("请输入新的分析提示")

    if not reanalyze_llm_model:
        raise gr.Error("请选择大语言模型")

    try:
        # 使用新提示重新处理
        from modules.llm_processor import LLMProcessor
        llm = LLMProcessor(reanalyze_llm_model)
        updated_results = []

        for file_data in task_result["result"]:
            new_segments = llm.segment_video(file_data["align_result"],
                                             new_prompt)
            updated_results.append({
                "filename": file_data["filename"],
                "filepath": file_data["filepath"],
                "align_result": file_data["align_result"],
                "segments": new_segments
            })

        # 整理结果以便显示
        display_result = []
        clip_result = []
        for file_result in updated_results:
            for seg in file_result["segments"]:
                row = [file_result["filename"],
                       f"{seconds_to_hhmmss(seg['start'])}",
                       f"{seconds_to_hhmmss(seg['end'])}",
                       f"{seconds_to_hhmmss(seg['end'] - seg['start'])}",
                       seg["summary"],
                       ", ".join(seg["tags"]) if isinstance(
                           seg["tags"], list) else seg["tags"]]
                clip_row = row.copy()
                clip_row.insert(0, CHECKBOX_UNCHECKED)  # 添加选择框
                display_result.append(row)
                clip_result.append(clip_row)

        return ({
                    "task_id": task_id,
                    "status": "重新分析完成，请在分析结果中查看",
                    "result": display_result,
                    "raw_result": updated_results
                }, display_result, clip_result)

    except Exception as e:
        print(f"重新分析失败: {str(e)}")
        task_result["status"] = "error"
        task_result["status_info"] = f"重新分析失败: {str(e)}"
        return task_result, [], []


def create_gradio_interface():
    """创建Gradio界面"""
    with (gr.Blocks(title="PreenCut", theme=gr.themes.Soft()) as app):
        gr.Markdown("# 🎬 PreenCut-AI视频剪辑助手")
        gr.Markdown(
            "上传包含语音的视频/音频文件，AI将自动识别语音内容、智能分段，并允许您输入自然语言进行检索。")

        with gr.Row():
            with gr.Column(scale=2):
                file_upload = gr.Files(
                    label="上传视频/音频文件",
                    file_count="multiple"
                )

                with gr.Accordion("高级设置", open=False):
                    llm_model = gr.Dropdown(
                        choices=[model['label'] for model in LLM_MODEL_OPTIONS],
                        value="豆包", label="大语言模型")
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
                    reanalyze_llm_model = gr.Dropdown(
                        choices=[model['label'] for model in LLM_MODEL_OPTIONS],
                        value="豆包", label="大语言模型")
                    reanalyze_btn = gr.Button("重新分析", variant="secondary")

                with gr.Tab("剪辑选项"):
                    segment_selection = gr.Dataframe(
                        headers=["选择", "文件名", "开始时间", "结束时间",
                                 "时长",
                                 "内容摘要", "标签"],
                        datatype='html',
                        interactive=False,
                        wrap=True,
                        type="array",
                        label="选择要保留的片段"
                    )
                    segment_selection.select(select_clip,
                                             inputs=segment_selection,
                                             outputs=segment_selection)
                    # 添加下载模式选择
                    download_mode = gr.Radio(
                        choices=["打包成zip文件", "合并成一个文件"],
                        label="选择多个文件时的处理方式",
                        value="打包成zip文件"
                    )
                    clip_btn = gr.Button("剪辑", variant="primary")
                    download_output = gr.File(label="下载剪辑结果")

        # 定时器，用于轮询状态
        timer = gr.Timer(2, active=False)
        timer.tick(check_status, task_id,
                   outputs=[status_display, result_table, segment_selection,
                            timer])

        # 事件处理
        process_btn.click(
            process_files,
            inputs=[file_upload, llm_model, prompt_input, model_size],
            outputs=[task_id, status_display]
        ).then(
            lambda: gr.Timer(active=True),
            inputs=None,
            outputs=timer,
            show_progress="hidden"
        )

        reanalyze_btn.click(
            start_reanalyze,
            inputs=None,
            outputs=status_display,
        ).then(
            reanalyze_with_prompt,
            inputs=[task_id, reanalyze_llm_model, new_prompt],
            outputs=[status_display, result_table, segment_selection],
            show_progress="hidden"
        )

        clip_btn.click(
            clip_and_download,
            inputs=[status_display, segment_selection, download_mode],
            outputs=download_output
        )

        return app
