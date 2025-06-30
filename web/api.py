from fastapi import APIRouter, UploadFile, HTTPException
from modules.processing_queue import ProcessingQueue
from typing import Optional
import os
import uuid
from pydantic import BaseModel
from datetime import date
from pathlib import Path

temp_dir = os.environ["GRADIO_TEMP_DIR"]

processing_queue = ProcessingQueue()
files_dict = {}

router = APIRouter(prefix="/api")


class createTransribeTaskBody(BaseModel):
    whisper_model_size: str
    llm_model: str
    prompt: Optional[str] = None
    file_path: str


@router.post("/upload")
def upload(file: UploadFile):
    today = date.today()
    date_str = date.strftime(today, "%Y/%m/%d")
    file_ext = Path(file.filename).suffix
    uuid_ex = str(uuid.uuid1()).replace("-", "")
    save_dir = f"{temp_dir}/files/{date_str}"
    if not Path(save_dir).exists():
        Path(save_dir).mkdir(parents=True)
    save_path = f"{save_dir}/{uuid_ex}{file_ext}"
    with file.file as src, open(save_path, "wb+") as dst:
        dst.write(src.read())
    return {"file_path": save_path}


@router.post("/tasks")
def createTranscribeTask(body: createTransribeTaskBody):
    task_id = f"task_{uuid.uuid4().hex}"
    file_path = body.file_path
    if not file_path.startswith(f"{temp_dir}/files"):
        raise HTTPException(status_code=403,
                            detail="file is not permit to visit")
    if not Path(file_path).exists():
        raise HTTPException(status_code=400, detail="file not found")
    print(f"添加任务: {task_id}, 文件路径: {file_path}")
    # 添加到处理队列
    processing_queue.add_task(
        task_id, [file_path], body.llm_model, body.prompt,
        body.whisper_model_size
    )
    return {"task_id": task_id}


@router.get("/tasks/{task_id}")
def queryTranscribeTask(task_id: str):
    result = processing_queue.get_result(task_id)
    return result
