import asyncio
import logging
from fastapi import BackgroundTasks, FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Import our unified predict pipeline
from predict import predict_image
from exceptions import PreprocessingError, ModelExecutionError
from api.task_store import TaskStore, TaskResult

logger = logging.getLogger(__name__)
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024
UPLOAD_READ_CHUNK_BYTES = 1024 * 1024


async def _read_image_bytes(file: UploadFile) -> bytes:
    chunks = []
    total_size = 0
    while chunk := await file.read(UPLOAD_READ_CHUNK_BYTES):
        total_size += len(chunk)
        if total_size > MAX_UPLOAD_SIZE_BYTES:
            raise HTTPException(status_code=413, detail="Uploaded image is too large.")
        chunks.append(chunk)
    return b"".join(chunks)

app = FastAPI(
    title="PixelTruth API",
    description="Deepfake detection API that classifies an image as Real or Fake.",
    version="1.0.0",
)

# Allow CORS for external web integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global in-memory task store for async inference.
task_store = TaskStore()


def _format_inference_response(result: dict) -> dict:
    return {
        "verdict": result["label"],
        "confidence": result["confidence"],
        "raw_scores": result["raw"],
    }


def _run_inference_task(task_id: str, image_bytes: bytes) -> None:
    task_store.mark_running(task_id)
    try:
        result = predict_image(image_bytes)
        task_store.mark_completed(task_id, result)
    except Exception as exc:
        logger.error("Background inference task failed", exc_info=exc)
        task_store.mark_failed(task_id, str(exc))

@app.post("/api/detect")
async def detect_image(file: UploadFile = File(...)):
    """
    Accepts an uploaded image file and returns deepfake detection results.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        image_bytes = await _read_image_bytes(file)
        result = await asyncio.to_thread(predict_image, image_bytes)
        return _format_inference_response(result)

    except HTTPException:
        raise
    except PreprocessingError as e:
        logger.error(f"Preprocessing error: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except ModelExecutionError as e:
        logger.error(f"Model error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during model execution.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")


@app.post("/api/detect/async", status_code=202)
async def detect_image_async(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        image_bytes = await _read_image_bytes(file)
        task_id = task_store.create_task()
        background_tasks.add_task(_run_inference_task, task_id, image_bytes)
        return {"task_id": task_id}

    except HTTPException:
        raise
    except PreprocessingError as e:
        logger.error(f"Preprocessing error: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except ModelExecutionError as e:
        logger.error(f"Model error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during model execution.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")


@app.get("/api/task/{task_id}", response_model=TaskResult)
async def get_task_status(task_id: str):
    task = task_store.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found.")
    return task
