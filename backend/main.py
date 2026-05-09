from fastapi import FastAPI, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, StreamingResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from PIL import Image
import csv
from io import StringIO
from io import BytesIO
import os
import time

# Import your modules
from model_manager import ModelManager
from augmentation import apply_augmentation
from inference import run_inference
from config import AVAILABLE_MODELS, AVAILABLE_AUGMENTATIONS
from evaluation import evaluate_outputs, measure_inference_time

app = FastAPI()

# Mount static directory
app.mount("/static", StaticFiles(directory="../static"), name="static")

# Templates directory
templates = Jinja2Templates(directory="../frontend")

# Initialize model manager
# model_manager = ModelManager()


# ==============================
# HOME PAGE
# ==============================
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "models": AVAILABLE_MODELS,
            "augmentations": AVAILABLE_AUGMENTATIONS
        }
    )


# ==============================
# PROCESS ROUTE
# ==============================
@app.post("/process")
async def process_image(
    request: Request,
    file: UploadFile,
    model_name: str = Form(...),
    augmentation_method: str = Form(...),
    prompt: str = Form(...),
    ground_truth: str = Form(...)
):

    async def event_generator():
        import asyncio

        yield "data: PROCESS ROUTE HIT\n\n"
        await asyncio.sleep(0.01)

        contents = await file.read()
        filename = file.filename.lower()

        # Ensure upload folder exists
        os.makedirs("../static/uploads", exist_ok=True)

        media_path_disk = f"../static/uploads/{file.filename}"
        media_path_web = f"/static/uploads/{file.filename}"

        # Save uploaded file
        with open(media_path_disk, "wb") as f:
            f.write(contents)

        # Detect media type
        if filename.endswith((".png", ".jpg", ".jpeg", ".bmp")):
            media_type = "image"
        elif filename.endswith((".mp4", ".avi", ".mov")):
            media_type = "video"
        else:
            yield "data: ERROR: Unsupported file type\n\n"
            return

        yield f"data: Media type detected: {media_type}\n\n"
        await asyncio.sleep(0.01)

        # ==========================
        # IMAGE PIPELINE
        # ==========================
        if media_type == "image":

            image = Image.open(BytesIO(contents)).convert("RGB")
            
            yield f"data: Applying augmentation: {augmentation_method}\n\n"
            await asyncio.sleep(0.01)

            start_aug = time.time()
            # Apply augmentation
            augmented_image = apply_augmentation(image, augmentation_method, prompt)

            # Save augmented image
            augmented_path_disk = "../static/uploads/augmented_" + file.filename
            augmented_path_web = "/static/uploads/augmented_" + file.filename
            augmented_image.save(augmented_path_disk)

            yield "data: Loading model...\n\n"
            await asyncio.sleep(0.01)

            # Load model
            # model = model_manager.get_model(model_name)
            model = model_name  # Placeholder since we're directly calling inference

            yield "data: Running augmented inference...\n\n"
            await asyncio.sleep(0.01)

            # -------- Augmented inference --------
            augmented_output = run_inference(model, augmented_path_disk, prompt , "augmented")
            end_aug = time.time()

            yield "data: Running original inference...\n\n"
            await asyncio.sleep(0.01)

            # -------- Original inference --------
            start_original = time.time()
            original_output = run_inference(model, media_path_disk, prompt , "original")
            end_original = time.time()

            print("Start Time:", start_original, "End Time:", end_original)
            print("Start Time:", start_aug, "End Time:", end_aug)
            # Measure inference times
            inference_time_original = measure_inference_time(start_original, end_original)
            inference_time_augmented = measure_inference_time(start_aug, end_aug)

            yield "data: Evaluating outputs...\n\n"
            await asyncio.sleep(0.01)

            # Run evaluation
            evaluation_results = evaluate_outputs(
                original_output,
                augmented_output,
                inference_time_original,
                inference_time_augmented,
                ground_truth
            )

        # ==========================
        # VIDEO PIPELINE (Placeholder)
        # ==========================
        else:
            original_output = "[VIDEO PROCESSING NOT IMPLEMENTED YET]"
            augmented_output = "[VIDEO PROCESSING NOT IMPLEMENTED YET]"
            inference_time_original = 0
            inference_time_augmented = 0
            evaluation_results = {
                "original_bert_score": "N/A",
                "original_bleu_score": "N/A",
                "augmented_bert_score": "N/A",
                "augmented_bleu_score": "N/A",
                "token_count_original": 0,
                "token_count_augmented": 0,
                "latency_diff": 0
            }
            augmented_path_web = None

        # ==========================
        # RENDER DASHBOARD
        # ==========================
        yield "data: Rendering dashboard...\n\n"
        await asyncio.sleep(0.01)

        context = {
            "request": request,
            "media_type": media_type,
            "media_path": media_path_web,
            "augmented_media_path": augmented_path_web,
            "original_output": original_output,
            "augmented_output": augmented_output,
            "inference_time_original": inference_time_original,
            "inference_time_augmented": inference_time_augmented,
            "original_bert_score": evaluation_results["original_bert_score"],
            "original_bleu_score": evaluation_results["original_bleu_score"],
            "augmented_bert_score": evaluation_results["augmented_bert_score"],
            "augmented_bleu_score": evaluation_results["augmented_bleu_score"],
            "token_count_original": evaluation_results["token_count_original"],
            "token_count_augmented": evaluation_results["token_count_augmented"],
            "latency_diff": evaluation_results["latency_diff"],
            "ground_truth": ground_truth
        }

        print("Done, Rendering Dashboard")
        html = templates.env.get_template('dashboard.html').render(context)
        encoded_html = "data: " + html.replace("\n", "\ndata: ") + "\n\n"

        yield "data: DONE\n\n"
        yield encoded_html

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/download_metrics")
async def download_metrics(
    ground_truth: str = "",
    augmented_output: str = "",
    original_output: str = "",
    original_bert_score: str = "",
    original_bleu_score: str = "",
    augmented_bert_score: str = "",
    augmented_bleu_score: str = "",
    token_count_original: int = 0,
    token_count_augmented: int = 0,
    latency_diff: float = 0.0,
    inference_time_original: float = 0.0,
    inference_time_augmented: float = 0.0
):
    """
    Generates a downloadable CSV report for evaluation metrics
    """

    # Prepare CSV in memory
    output = StringIO()
    writer = csv.writer(output)

    # Write header dynamically
    writer.writerow([
        "Metric",
        "Value"
    ])

    # Collect metrics dynamically
    metrics_dict = {
        "Ground Truth": ground_truth,
        "Original Output": original_output,
        "Augmented Output": augmented_output,
        "Original BERTScore": original_bert_score,
        "Original BLEU Score": original_bleu_score,
        "Augmented BERTScore": augmented_bert_score,
        "Augmented BLEU Score": augmented_bleu_score,
        "Original Token Count": token_count_original,
        "Augmented Token Count": token_count_augmented,
        "Latency Difference (ms)": latency_diff,
        "Original Inference Time (ms)": inference_time_original,
        "Augmented Inference Time (ms)": inference_time_augmented
    }

    for k, v in metrics_dict.items():
        writer.writerow([k, v])

    # Reset buffer to beginning
    output.seek(0)

    # Return as downloadable file
    return StreamingResponse(
        output,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=evaluation_metrics.csv"}
    )