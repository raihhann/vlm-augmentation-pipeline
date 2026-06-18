from fastapi import FastAPI, UploadFile, Form, Request, File
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from PIL import Image
import pandas as pd
from io import BytesIO
import os
import time

# Import your modules
from model_manager import ModelManager
from augmentation import apply_augmentation
from Augmentify.extract.video_extractor import extract_video_keyframes
from inference import run_inference
from config import AVAILABLE_MODELS, AVAILABLE_AUGMENTATIONS
from evaluation import evaluate_outputs, measure_inference_time

app = FastAPI()

app.mount("/static", StaticFiles(directory="../static"), name="static")
templates = Jinja2Templates(directory="../frontend")

# In-memory storage to persist matrix evaluation metrics across requests
GLOBAL_SESSION_STORAGE_MATRIX = {}

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

@app.post("/process")
async def process_media_matrix(
    request: Request,
    processing_mode: str = Form(...),
    model_name: str = Form(...),
    # Image Mode inputs (Received as dynamic parallel arrays from multi-card form)
    image_files: list[UploadFile] = File(None),
    image_prompts: list[str] = Form(None),
    image_ground_truths: list[str] = Form(None),
    augmentation_methods: list[str] = Form(None),
    # Video Mode single inputs fallback
    file: UploadFile = File(None),
    video_sampling_method: str = Form("1"),
    augmentation_method_video: str = Form("none"),
    prompt: str = Form(None),
    ground_truth: str = Form(None)
):

    async def batch_event_generator():
        import asyncio
        session_token = f"matrix_session_{int(time.time())}"
        GLOBAL_SESSION_STORAGE_MATRIX[session_token] = []
        
        yield "data: BATCH ROW RUNWAY INITIALIZED\n\n"
        await asyncio.sleep(0.01)

        os.makedirs("../static/uploads", exist_ok=True)
        model = model_name
        
        # ==========================================
        # 📷 BATCH MULTI-IMAGE MULTI-AUGMENTATION MATRIX
        # ==========================================
        if processing_mode == "image":
            files_pool = image_files if image_files else []
            augs_pool = augmentation_methods if augmentation_methods else ["none"]
            
            total_cells = len(files_pool) * len(augs_pool)
            yield f"data: Batch parameters matched. Processing execution grid matrix size: {total_cells} Cells\n\n"
            await asyncio.sleep(0.01)

            # Iterative Matrix Loop Step
            for idx, uploaded_file in enumerate(files_pool):
                raw_bytes = await uploaded_file.read()
                filename = uploaded_file.filename
                
                # Capture the specific unique pair coordinates for this specific item index
                current_prompt = image_prompts[idx] if idx < len(image_prompts) else "Describe the image."
                current_gt = image_ground_truths[idx] if idx < len(image_ground_truths) else ""

                # Write individual unique original asset onto disk
                media_path_disk = f"../static/uploads/{filename}"
                media_path_web = f"/static/uploads/{filename}"
                with open(media_path_disk, "wb") as buffer:
                    buffer.write(raw_bytes)

                for single_aug in augs_pool:
                    yield f"data: File [{idx + 1}/{len(files_pool)}]: {filename} ──> Checking Method: {single_aug}...\n\n"
                    await asyncio.sleep(0.01)

                    # Initialize fresh image handle out of memory bytes
                    pil_image = Image.open(BytesIO(raw_bytes)).convert("RGB")
                    
                    # Run your UNTOUCHED 2D augmentation module dynamically passing local variables
                    start_aug = time.time()
                    augmented_image = apply_augmentation(pil_image, single_aug, current_prompt)
                    
                    # Output the unique cell result
                    augmented_path_disk = f"../static/uploads/cell_{idx}_{single_aug}_{filename}"
                    augmented_path_web = f"/static/uploads/cell_{idx}_{single_aug}_{filename}"
                    augmented_image.save(augmented_path_disk)

                    # Model inference executions
                    augmented_output = run_inference(model, augmented_path_disk, current_prompt, "augmented")
                    end_aug = time.time()

                    start_original = time.time()
                    original_output = run_inference(model, media_path_disk, current_prompt, "original")
                    end_original = time.time()

                    t_orig = measure_inference_time(start_original, end_original)
                    t_proc = measure_inference_time(start_aug, end_aug)

                    # Evaluate metrics using local ground truth string matching
                    eval_res = evaluate_outputs(original_output, augmented_output, t_orig, t_proc, current_gt)

                    # Save to local session log state dictionary
                    GLOBAL_SESSION_STORAGE_MATRIX[session_token].append({
                        "file_name": filename,
                        "variant_method": single_aug,
                        "media_type": "image",
                        "media_path_web": media_path_web,
                        "augmented_media_path": augmented_path_web,
                        "original_output": original_output,
                        "processed_output": augmented_output,
                        "time_original": t_orig,
                        "time_processed": t_proc,
                        "latency_diff": eval_res["latency_diff"],
                        "original_bert": eval_res["original_bert_score"],
                        "original_bleu": eval_res["original_bleu_score"],
                        "processed_bert": eval_res["augmented_bert_score"],
                        "processed_bleu": eval_res["augmented_bleu_score"],
                        "token_original": eval_res["token_count_original"],
                        "token_processed": eval_res["token_count_augmented"]
                    })

        # ==========================================
        # 🎬 MODULAR VIDEO SINGLE TARGET SEQUENCER LOOP
        # ==========================================
        else:
            raw_bytes = await file.read()
            filename = file.filename
            media_path_disk = f"../static/uploads/{filename}"
            media_path_web = f"/static/uploads/{filename}"
            
            with open(media_path_disk, "wb") as buffer:
                buffer.write(raw_bytes)

            video_out_disk = f"../static/uploads/extracted_{session_token}"
            video_aug_disk = f"../static/uploads/augmented_{session_token}"
            os.makedirs(video_aug_disk, exist_ok=True)

            yield f"data: Stage 1: Running video keyframe extraction (Method {video_sampling_method})...\n\n"
            await asyncio.sleep(0.01)

            start_aug = time.time()
            extracted_frames = extract_video_keyframes(
                video_path=media_path_disk, method=video_sampling_method, query=prompt, output_dir=video_out_disk
            )

            yield f"data: Stage 2: Applying '{augmentation_method_video}' augmentation across all extracted frames...\n\n"
            await asyncio.sleep(0.01)

            augmented_frames = []
            for f_path in extracted_frames:
                f_img = Image.open(f_path).convert("RGB")
                aug_f_img = apply_augmentation(f_img, augmentation_method_video, prompt)
                aug_f_path = os.path.join(video_aug_disk, f"aug_{os.path.basename(f_path)}")
                aug_f_img.save(aug_f_path)
                augmented_frames.append(aug_f_path)

            augmented_output = run_inference(model, augmented_frames[0], prompt, "augmented")
            end_aug = time.time()

            start_original = time.time()
            original_output = run_inference(model, media_path_disk, prompt, "original")
            end_original = time.time()

            t_orig = measure_inference_time(start_original, end_original)
            t_proc = measure_inference_time(start_aug, end_aug)
            eval_res = evaluate_outputs(original_output, augmented_output, t_orig, t_proc, ground_truth)

            extracted_keyframes_web = [p.replace("../static", "/static").replace("\\", "/") for p in extracted_frames]
            augmented_keyframes_web = [p.replace("../static", "/static").replace("\\", "/") for p in augmented_frames]

            GLOBAL_SESSION_STORAGE_MATRIX[session_token].append({
                "file_name": filename,
                "variant_method": f"Method {video_sampling_method} + {augmentation_method_video}",
                "media_type": "video",
                "media_path_web": media_path_web,
                "extracted_keyframes": extracted_keyframes_web,
                "augmented_media_path": augmented_keyframes_web[0],
                "original_output": original_output,
                "processed_output": augmented_output,
                "time_original": t_orig,
                "time_processed": t_proc,
                "latency_diff": eval_res["latency_diff"],
                "original_bert": eval_res["original_bert_score"],
                "original_bleu": eval_res["original_bleu_score"],
                "processed_bert": eval_res["augmented_bert_score"],
                "processed_bleu": eval_res["augmented_bleu_score"],
                "token_original": eval_res["token_count_original"],
                "token_processed": eval_res["token_count_augmented"]
            })

        # ==========================================
        # 🗺️ DASHBOARD GENERATION AND DELIVERY RENDER
        # ==========================================
        yield "data: Compiling dashboard metrics results grid matrix views...\n\n"
        await asyncio.sleep(0.01)

        context = {
            "request": request,
            "session_token": session_token,
            "dataset_matrix": GLOBAL_SESSION_STORAGE_MATRIX[session_token]
        }

        html = templates.env.get_template('dashboard.html').render(context)
        encoded_html = "data: " + html.replace("\n", "\ndata: ") + "\n\n"

        yield "data: DONE\n\n"
        yield encoded_html

    return StreamingResponse(batch_event_generator(), media_type="text/event-stream")

@app.get("/download_metrics")
async def export_excel_metrics_matrix(session_token: str = ""):
    dataset = GLOBAL_SESSION_STORAGE_MATRIX.get(session_token, [])
    if not dataset:
        return HTMLResponse(content="<h3>Error: Target Matrix Context Expired.</h3>", status_code=404)

    df = pd.DataFrame(dataset)
    
    # Prune out local layout preview configuration tracking paths
    export_columns = [
        "file_name", "variant_method", "media_type", "time_original", 
        "time_processed", "latency_diff", "original_bert", "original_bleu", 
        "processed_bert", "processed_bleu", "token_original", "token_processed"
    ]
    df_filtered = df[export_columns].copy()
    df_filtered.columns = [c.replace("_", " ").title() for c in export_columns]

    output_stream = BytesIO()
    with pd.ExcelWriter(output_stream, engine="openpyxl") as writer:
        df_filtered.to_excel(writer, index=False, sheet_name="VLM Batch Evaluation Matrix")

    output_stream.seek(0)
    return StreamingResponse(
        output_stream,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename=VLM_Robustness_Batch_Report.xlsx"}
    )