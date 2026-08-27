"""Serve the web interface and execute image and video evaluation workflows.

This FastAPI application accepts uploaded media or spreadsheet datasets,
applies augmentations and keyframe extractors, streams progress, and exports
the resulting evaluation matrices as HTML, CSV, or Excel reports.
"""

import asyncio
from io import BytesIO
import os
import time
from typing import List, Optional

from Augmentify.FramesExtraction import extract_frames
from augmentation import apply_augmentation
from config import (
    AVAILABLE_AUGMENTATIONS,
    AVAILABLE_EXTRACTIONS,
    AVAILABLE_MODELS,
)
from evaluation import evaluate_outputs, measure_inference_time
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from inference import run_inference
import pandas as pd
from PIL import Image

app = FastAPI()

app.mount("/static", StaticFiles(directory="../static"), name="static")
templates = Jinja2Templates(directory="../frontend")

GLOBAL_SESSION_STORAGE_MATRIX = {}


def to_web_url(disk_path: str) -> str:
    """Converts local disk paths into relative /static/ web URLs."""
    norm = disk_path.replace("\\", "/")
    if "/static/" in norm:
        return "/static/" + norm.split("/static/", 1)[1]
    return norm


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "models": AVAILABLE_MODELS,
            "augmentations": AVAILABLE_AUGMENTATIONS,
            "extractions": AVAILABLE_EXTRACTIONS,
        },
    )


@app.get("/excel_studio", response_class=HTMLResponse)
async def excel_studio_page(request: Request):
    return templates.TemplateResponse(
        "excel_evaluator.html",
        {
            "request": request,
            "models": AVAILABLE_MODELS,
            "augmentations": AVAILABLE_AUGMENTATIONS,
            "extractions": AVAILABLE_EXTRACTIONS,
        },
    )


@app.post("/process")
async def process_media_matrix(
    request: Request,
    processing_mode: str = Form(...),
    model_name: str = Form(...),
    # Image Mode inputs
    image_files: Optional[List[UploadFile]] = File(default=None),
    image_prompts: Optional[List[str]] = Form(default=None),
    image_ground_truths: Optional[List[str]] = Form(default=None),
    augmentation_methods: Optional[List[str]] = Form(default=None),
    # Video Mode matrix array inputs
    video_files: Optional[List[UploadFile]] = File(default=None),
    video_prompts: Optional[List[str]] = Form(default=None),
    video_ground_truths: Optional[List[str]] = Form(default=None),
    video_sampling_methods: Optional[List[str]] = Form(default=None),
    max_frames_video: Optional[int] = Form(default=8),
    augmentation_methods_video: Optional[List[str]] = Form(default=None),
    # Excel Dataset Batch Mode inputs
    excel_file: Optional[UploadFile] = File(default=None),
    excel_image_dir: Optional[str] = Form(default=None),
    excel_col_filename: Optional[str] = Form(default=None),
    excel_col_prompt: Optional[str] = Form(default=None),
    excel_col_gt: Optional[str] = Form(default=None),
    excel_extra_cols: Optional[List[str]] = Form(default=None),
):
    async def batch_event_generator():
        session_token = f"matrix_session_{int(time.time())}"
        GLOBAL_SESSION_STORAGE_MATRIX[session_token] = []

        yield "data: BATCH ROW RUNWAY INITIALIZED\n\n"
        await asyncio.sleep(0.01)

        os.makedirs("../static/uploads", exist_ok=True)
        os.makedirs("../static/cache", exist_ok=True)
        model = model_name
        baseline_cache = {}

        # ==========================================================
        # 📊 1. EXCEL BATCH IMAGE DATASET MODE
        # ==========================================================
        if processing_mode in ["excel", "excel_image"]:
            yield "data: Reading image dataset file...\n\n"
            await asyncio.sleep(0.01)

            excel_bytes = await excel_file.read()
            df = pd.read_excel(BytesIO(excel_bytes))

            augs_pool = augmentation_methods if augmentation_methods else ["none"]
            extra_cols_list = excel_extra_cols if excel_extra_cols else []
            total_rows = len(df)
            total_runs = total_rows * len(augs_pool)
            cache_csv_path = f"../static/cache/dataset_eval_{session_token}.csv"

            yield (
                f"data: Loaded {total_rows} rows. Total evaluation runs: {total_runs}. "
                f"Cache file: dataset_eval_{session_token}.csv\n\n"
            )
            await asyncio.sleep(0.01)

            for idx, row in df.iterrows():
                img_name = str(row[excel_col_filename]).strip()
                current_prompt = str(row[excel_col_prompt]).strip()
                current_gt = str(row[excel_col_gt]).strip()

                extra_data_dict = {
                    col: str(row[col]) for col in extra_cols_list if col in row
                }
                media_path_disk = os.path.join(excel_image_dir, img_name)

                if not os.path.exists(media_path_disk):
                    yield f"data: [Warning] Image file not found: {media_path_disk}. Skipping row...\n\n"
                    await asyncio.sleep(0.01)
                    continue

                try:
                    pil_image = Image.open(media_path_disk).convert("RGB")
                except Exception as e:
                    yield f"data: [Error] Could not open image {img_name}: {str(e)}. Skipping row...\n\n"
                    await asyncio.sleep(0.01)
                    continue

                cache_key = (media_path_disk, current_prompt)
                if cache_key not in baseline_cache:
                    yield f"data: Processing Baseline for [{idx + 1}/{total_rows}]: {img_name}...\n\n"
                    await asyncio.sleep(0.01)

                    start_original = time.time()
                    original_output = run_inference(
                        model, media_path_disk, current_prompt, "original"
                    )
                    end_original = time.time()
                    t_orig = measure_inference_time(start_original, end_original)
                    baseline_cache[cache_key] = (original_output, t_orig)
                else:
                    original_output, t_orig = baseline_cache[cache_key]

                for single_aug in augs_pool:
                    yield f"data: Processing [{idx + 1}/{total_rows}]: {img_name} ──> Method: {single_aug}...\n\n"
                    await asyncio.sleep(0.01)

                    try:
                        start_aug = time.time()
                        augmented_image = apply_augmentation(
                            pil_image, single_aug, current_prompt
                        )

                        augmented_path_disk = f"../static/uploads/batch_{idx}_{single_aug}_{img_name}"
                        augmented_path_web = f"/static/uploads/batch_{idx}_{single_aug}_{img_name}"
                        augmented_image.save(augmented_path_disk)

                        augmented_output = run_inference(
                            model, augmented_path_web, current_prompt, "augmented"
                        )
                        end_aug = time.time()
                        t_proc = measure_inference_time(start_aug, end_aug)

                        eval_res = evaluate_outputs(
                            original_output,
                            augmented_output,
                            t_orig,
                            t_proc,
                            current_gt,
                            image_input=media_path_disk,
                        )

                        row_record = {
                            "Image Filename": img_name,
                            "Evaluation Method": single_aug,
                            "Evaluation Prompt": current_prompt,
                            "Ground Truth Reference": current_gt,
                            **extra_data_dict,
                            "Baseline Output": original_output,
                            "Augmented Output": augmented_output,
                            "Baseline Latency (s)": t_orig,
                            "Augmented Latency (s)": t_proc,
                            "Latency Delta Difference (s)": eval_res["latency_diff"],
                            "Baseline BERTScore": eval_res["original_bert_score"],
                            "Baseline BLEU Score": eval_res["original_bleu_score"],
                            "Baseline CLIPScore": eval_res["original_clip_score"],
                            "Augmented BERTScore": eval_res["augmented_bert_score"],
                            "Augmented BLEU Score": eval_res["augmented_bleu_score"],
                            "Augmented CLIPScore": eval_res["augmented_clip_score"],
                            "Baseline Tokens": eval_res["token_count_original"],
                            "Augmented Tokens": eval_res["token_count_augmented"],
                        }

                        GLOBAL_SESSION_STORAGE_MATRIX[session_token].append(row_record)
                        row_df = pd.DataFrame([row_record])
                        file_exists = os.path.exists(cache_csv_path)
                        row_df.to_csv(
                            cache_csv_path,
                            mode="a",
                            index=False,
                            header=not file_exists,
                        )
                    except Exception as exc:
                        yield f"data: [Pipeline Error] Exception on {img_name} under {single_aug}: {str(exc)}. Skipping step...\n\n"
                        await asyncio.sleep(0.01)
                        continue

            yield f"data: DOWNLOAD_TOKEN:{session_token}\n\n"
            yield "data: DONE\n\n"

        # ==========================================================
        # 🎬 2. EXCEL BATCH VIDEO DATASET MODE
        # ==========================================================
        elif processing_mode == "excel_video":
            yield "data: Reading video dataset file...\n\n"
            await asyncio.sleep(0.01)

            excel_bytes = await excel_file.read()
            df = pd.read_excel(BytesIO(excel_bytes))

            samplers_pool = (
                video_sampling_methods if video_sampling_methods else ["bolt"]
            )
            v_augs_pool = (
                augmentation_methods_video if augmentation_methods_video else ["none"]
            )
            extra_cols_list = excel_extra_cols if excel_extra_cols else []
            total_rows = len(df)
            total_runs = total_rows * len(samplers_pool) * len(v_augs_pool)
            cache_csv_path = f"../static/cache/video_dataset_eval_{session_token}.csv"

            yield (
                f"data: Loaded {total_rows} video dataset rows. Total evaluation runs: {total_runs}. "
                f"Cache file: video_dataset_eval_{session_token}.csv\n\n"
            )
            await asyncio.sleep(0.01)

            for idx, row in df.iterrows():
                vid_name = str(row[excel_col_filename]).strip()
                current_prompt = str(row[excel_col_prompt]).strip()
                current_gt = str(row[excel_col_gt]).strip()

                extra_data_dict = {
                    col: str(row[col]) for col in extra_cols_list if col in row
                }
                media_path_disk = os.path.join(excel_image_dir, vid_name)

                if not os.path.exists(media_path_disk):
                    yield f"data: [Warning] Video file not found: {media_path_disk}. Skipping row...\n\n"
                    await asyncio.sleep(0.01)
                    continue

                cache_key = (media_path_disk, current_prompt)
                if cache_key not in baseline_cache:
                    yield f"data: Processing Baseline for [{idx + 1}/{total_rows}]: {vid_name}...\n\n"
                    await asyncio.sleep(0.01)

                    start_original = time.time()
                    original_output = run_inference(
                        model, media_path_disk, current_prompt, "original"
                    )
                    end_original = time.time()
                    t_orig = measure_inference_time(start_original, end_original)
                    baseline_cache[cache_key] = (original_output, t_orig)
                else:
                    original_output, t_orig = baseline_cache[cache_key]

                for current_sampler in samplers_pool:
                    video_out_disk = f"../static/uploads/batch_vid_{idx}_{current_sampler}_{int(time.time())}"

                    yield (
                        f"data: [{idx + 1}/{total_rows}] {vid_name} ──>"
                        f" Extracting via {current_sampler} (Max K={max_frames_video})...\n\n"
                    )
                    await asyncio.sleep(0.01)

                    try:
                        extracted_frames = extract_frames(
                            video_path=media_path_disk,
                            method=current_sampler,
                            max_frames=max_frames_video,
                            prompt=current_prompt,
                            output_folder=video_out_disk,
                        )
                    except Exception as ext_err:
                        yield f"data: [Extraction Error] {current_sampler} on {vid_name}: {str(ext_err)}. Skipping...\n\n"
                        await asyncio.sleep(0.01)
                        continue

                    if not extracted_frames:
                        continue

                    num_extracted = len(extracted_frames)

                    for current_aug in v_augs_pool:
                        yield (
                            f"data: [{idx + 1}/{total_rows}] {vid_name} ──>"
                            f" [{current_sampler}] Applying Filter: {current_aug}...\n\n"
                        )
                        await asyncio.sleep(0.01)

                        try:
                            start_aug = time.time()
                            video_aug_disk = f"../static/uploads/batch_aug_{idx}_{current_sampler}_{current_aug}_{int(time.time())}"
                            os.makedirs(video_aug_disk, exist_ok=True)

                            augmented_frames_disk_pool = []
                            for f_path in extracted_frames:
                                f_img = Image.open(f_path).convert("RGB")
                                aug_f_img = apply_augmentation(
                                    f_img, current_aug, current_prompt
                                )
                                aug_f_path = os.path.join(
                                    video_aug_disk, f"aug_{os.path.basename(f_path)}"
                                )
                                aug_f_img.save(aug_f_path)
                                augmented_frames_disk_pool.append(aug_f_path)

                            augmented_output = run_inference(
                                model,
                                augmented_frames_disk_pool[0],
                                current_prompt,
                                "augmented",
                            )
                            end_aug = time.time()
                            t_proc = measure_inference_time(start_aug, end_aug)

                            eval_res = evaluate_outputs(
                                original_output,
                                augmented_output,
                                t_orig,
                                t_proc,
                                current_gt,
                                image_input=extracted_frames[0],
                            )

                            row_record = {
                                "Video Filename": vid_name,
                                "Keyframe Strategy": current_sampler,
                                "Extracted Frames Count": num_extracted,
                                "Max Frames Budget": max_frames_video,
                                "2D Post-Augmentation": current_aug,
                                "Evaluation Prompt": current_prompt,
                                "Ground Truth Reference": current_gt,
                                **extra_data_dict,
                                "Baseline Output": original_output,
                                "Augmented Output": augmented_output,
                                "Baseline Latency (s)": t_orig,
                                "Augmented Latency (s)": t_proc,
                                "Latency Delta Difference (s)": eval_res["latency_diff"],
                                "Baseline BERTScore": eval_res["original_bert_score"],
                                "Baseline BLEU Score": eval_res["original_bleu_score"],
                                "Baseline CLIPScore": eval_res["original_clip_score"],
                                "Augmented BERTScore": eval_res["augmented_bert_score"],
                                "Augmented BLEU Score": eval_res["augmented_bleu_score"],
                                "Augmented CLIPScore": eval_res["augmented_clip_score"],
                                "Baseline Tokens": eval_res["token_count_original"],
                                "Augmented Tokens": eval_res["token_count_augmented"],
                            }

                            GLOBAL_SESSION_STORAGE_MATRIX[session_token].append(row_record)
                            row_df = pd.DataFrame([row_record])
                            file_exists = os.path.exists(cache_csv_path)
                            row_df.to_csv(
                                cache_csv_path,
                                mode="a",
                                index=False,
                                header=not file_exists,
                            )
                        except Exception as exc:
                            yield f"data: [Pipeline Error] Exception on {vid_name} under {current_sampler}+{current_aug}: {str(exc)}. Skipping...\n\n"
                            await asyncio.sleep(0.01)
                            continue

            yield f"data: DOWNLOAD_TOKEN:{session_token}\n\n"
            yield "data: DONE\n\n"

        # ==========================================================
        # 📷 3. BATCH MULTI-IMAGE UI MATRIX (MANUAL)
        # ==========================================================
        elif processing_mode == "image":
            files_pool = image_files if image_files else []
            augs_pool = augmentation_methods if augmentation_methods else ["none"]
            total_cells = len(files_pool) * len(augs_pool)

            yield f"data: Processing execution grid matrix size: {total_cells} Cells\n\n"
            await asyncio.sleep(0.01)

            for idx, uploaded_file in enumerate(files_pool):
                raw_bytes = await uploaded_file.read()
                filename = uploaded_file.filename
                current_prompt = (
                    image_prompts[idx]
                    if idx < len(image_prompts)
                    else "Describe the image."
                )
                current_gt = (
                    image_ground_truths[idx] if idx < len(image_ground_truths) else ""
                )

                media_path_disk = f"../static/uploads/{filename}"
                media_path_web = f"/static/uploads/{filename}"
                with open(media_path_disk, "wb") as buffer:
                    buffer.write(raw_bytes)

                cache_key = (media_path_web, current_prompt)
                if cache_key not in baseline_cache:
                    yield f"data: File [{idx + 1}/{len(files_pool)}]: {filename} ──> Processing Baseline Output...\n\n"
                    await asyncio.sleep(0.01)

                    start_original = time.time()
                    original_output = run_inference(
                        model, media_path_web, current_prompt, "original"
                    )
                    end_original = time.time()
                    t_orig = measure_inference_time(start_original, end_original)
                    baseline_cache[cache_key] = (original_output, t_orig)
                else:
                    original_output, t_orig = baseline_cache[cache_key]

                pil_image = Image.open(BytesIO(raw_bytes)).convert("RGB")

                for single_aug in augs_pool:
                    yield f"data: File [{idx + 1}/{len(files_pool)}]: {filename} ──> Method: {single_aug}...\n\n"
                    await asyncio.sleep(0.01)

                    start_aug = time.time()
                    augmented_image = apply_augmentation(
                        pil_image, single_aug, current_prompt
                    )

                    augmented_path_disk = f"../static/uploads/cell_{idx}_{single_aug}_{filename}"
                    augmented_path_web = f"/static/uploads/cell_{idx}_{single_aug}_{filename}"
                    augmented_image.save(augmented_path_disk)

                    augmented_output = run_inference(
                        model, augmented_path_web, current_prompt, "augmented"
                    )
                    end_aug = time.time()
                    t_proc = measure_inference_time(start_aug, end_aug)

                    eval_res = evaluate_outputs(
                        original_output,
                        augmented_output,
                        t_orig,
                        t_proc,
                        current_gt,
                        image_input=pil_image,
                    )

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
                        "original_clip": eval_res["original_clip_score"],
                        "processed_bert": eval_res["augmented_bert_score"],
                        "processed_bleu": eval_res["augmented_bleu_score"],
                        "processed_clip": eval_res["augmented_clip_score"],
                        "token_original": eval_res["token_count_original"],
                        "token_processed": eval_res["token_count_augmented"],
                    })

            yield "data: Compiling dashboard metrics results grid matrix views...\n\n"
            await asyncio.sleep(0.01)

            context = {
                "request": request,
                "session_token": session_token,
                "dataset_matrix": GLOBAL_SESSION_STORAGE_MATRIX[session_token],
            }
            html = templates.env.get_template("dashboard.html").render(context)
            encoded_html = "data: " + html.replace("\n", "\ndata: ") + "\n\n"
            yield "data: DONE\n\n"
            yield encoded_html

        # ==========================================================
        # 🎬 4. BATCH MULTI-VIDEO UI MATRIX (MANUAL)
        # ==========================================================
        elif processing_mode == "video":
            v_files_pool = video_files if video_files else []
            samplers_pool = (
                video_sampling_methods if video_sampling_methods else ["bolt"]
            )
            v_augs_pool = (
                augmentation_methods_video if augmentation_methods_video else ["none"]
            )
            total_v_cells = len(v_files_pool) * len(samplers_pool) * len(v_augs_pool)

            yield f"data: Video evaluation matrix size: {total_v_cells} Work Cells\n\n"
            await asyncio.sleep(0.01)

            for idx, uploaded_file in enumerate(v_files_pool):
                raw_bytes = await uploaded_file.read()
                filename = uploaded_file.filename
                current_prompt = (
                    video_prompts[idx]
                    if idx < len(video_prompts)
                    else "Analyze video frames"
                )
                current_gt = (
                    video_ground_truths[idx] if idx < len(video_ground_truths) else ""
                )

                media_path_disk = f"../static/uploads/{filename}"
                media_path_web = f"/static/uploads/{filename}"
                with open(media_path_disk, "wb") as buffer:
                    buffer.write(raw_bytes)

                cache_key = (media_path_web, current_prompt)
                if cache_key not in baseline_cache:
                    yield f"data: Video [{idx + 1}/{len(v_files_pool)}]: {filename} ──> Processing Baseline Output...\n\n"
                    await asyncio.sleep(0.01)

                    start_original = time.time()
                    original_output = run_inference(
                        model, media_path_web, current_prompt, "original"
                    )
                    end_original = time.time()
                    t_orig = measure_inference_time(start_original, end_original)
                    baseline_cache[cache_key] = (original_output, t_orig)
                else:
                    original_output, t_orig = baseline_cache[cache_key]

                for current_sampler in samplers_pool:
                    video_out_disk = f"../static/uploads/extracted_{idx}_{current_sampler}_{int(time.time())}"

                    yield (
                        f"data: Video [{idx + 1}/{len(v_files_pool)}]: {filename} ──>"
                        f" Keyframe Strategy: {current_sampler} (Max K={max_frames_video})...\n\n"
                    )
                    await asyncio.sleep(0.01)

                    extracted_frames = extract_frames(
                        video_path=media_path_disk,
                        method=current_sampler,
                        max_frames=max_frames_video,
                        prompt=current_prompt,
                        output_folder=video_out_disk,
                    )

                    if not extracted_frames:
                        continue

                    num_extracted = len(extracted_frames)

                    for current_aug in v_augs_pool:
                        yield (
                            f"data: Video [{idx + 1}/{len(v_files_pool)}]: {filename}"
                            f" [{current_sampler}] ──> Applying 2D Filter: {current_aug}...\n\n"
                        )
                        await asyncio.sleep(0.01)

                        video_aug_disk = f"../static/uploads/augmented_{idx}_{current_sampler}_{current_aug}_{int(time.time())}"
                        os.makedirs(video_aug_disk, exist_ok=True)

                        start_aug = time.time()
                        augmented_frames_web_pool = []
                        for f_path in extracted_frames:
                            f_img = Image.open(f_path).convert("RGB")
                            aug_f_img = apply_augmentation(
                                f_img, current_aug, current_prompt
                            )
                            aug_f_path = os.path.join(
                                video_aug_disk, f"aug_{os.path.basename(f_path)}"
                            )
                            aug_f_img.save(aug_f_path)

                            augmented_frames_web_pool.append(to_web_url(aug_f_path))

                        augmented_output = run_inference(
                            model,
                            augmented_frames_web_pool[0],
                            current_prompt,
                            "augmented",
                        )
                        end_aug = time.time()
                        t_proc = measure_inference_time(start_aug, end_aug)

                        eval_res = evaluate_outputs(
                            original_output,
                            augmented_output,
                            t_orig,
                            t_proc,
                            current_gt,
                            image_input=extracted_frames[0],
                        )

                        extracted_keyframes_web = [to_web_url(p) for p in extracted_frames]

                        GLOBAL_SESSION_STORAGE_MATRIX[session_token].append({
                            "file_name": filename,
                            "variant_method": f"{current_sampler} + {current_aug}",
                            "media_type": "video",
                            "media_path_web": media_path_web,
                            "extracted_keyframes": extracted_keyframes_web,
                            "extracted_frames_count": num_extracted,
                            "max_frames_budget": max_frames_video,
                            "augmented_keyframes": augmented_frames_web_pool,
                            "augmented_media_path": augmented_frames_web_pool[0],
                            "original_output": original_output,
                            "processed_output": augmented_output,
                            "time_original": t_orig,
                            "time_processed": t_proc,
                            "latency_diff": eval_res["latency_diff"],
                            "original_bert": eval_res["original_bert_score"],
                            "original_bleu": eval_res["original_bleu_score"],
                            "original_clip": eval_res["original_clip_score"],
                            "processed_bert": eval_res["augmented_bert_score"],
                            "processed_bleu": eval_res["augmented_bleu_score"],
                            "processed_clip": eval_res["augmented_clip_score"],
                            "token_original": eval_res["token_count_original"],
                            "token_processed": eval_res["token_count_augmented"],
                        })

            yield "data: Compiling dashboard metrics results grid matrix views...\n\n"
            await asyncio.sleep(0.01)

            context = {
                "request": request,
                "session_token": session_token,
                "dataset_matrix": GLOBAL_SESSION_STORAGE_MATRIX[session_token],
            }
            html = templates.env.get_template("dashboard.html").render(context)
            encoded_html = "data: " + html.replace("\n", "\ndata: ") + "\n\n"
            yield "data: DONE\n\n"
            yield encoded_html

    return StreamingResponse(
        batch_event_generator(), media_type="text/event-stream"
    )


@app.get("/download_csv/{session_token}")
async def download_csv_report(session_token: str):
    cache_img = f"../static/cache/dataset_eval_{session_token}.csv"
    cache_vid = f"../static/cache/video_dataset_eval_{session_token}.csv"
    cache_csv_path = cache_img if os.path.exists(cache_img) else cache_vid

    if os.path.exists(cache_csv_path):
        with open(cache_csv_path, "rb") as f:
            csv_bytes = f.read()
        return StreamingResponse(
            BytesIO(csv_bytes),
            media_type="text/csv",
            headers={
                "Content-Disposition": (
                    f"attachment; filename=VLM_Batch_Evaluation_Results_{session_token}.csv"
                )
            },
        )

    dataset = GLOBAL_SESSION_STORAGE_MATRIX.get(session_token, [])
    if not dataset:
        return HTMLResponse(
            "<h3>Error: Cached dataset file or session expired.</h3>", 404
        )

    df = pd.DataFrame(dataset)
    output_stream = BytesIO()
    df.to_csv(output_stream, index=False)
    output_stream.seek(0)
    return StreamingResponse(
        output_stream,
        media_type="text/csv",
        headers={
            "Content-Disposition": (
                f"attachment; filename=VLM_Batch_Evaluation_Results_{session_token}.csv"
            )
        },
    )


@app.get("/download_metrics")
async def export_excel_metrics_matrix(session_token: str = ""):
    dataset = GLOBAL_SESSION_STORAGE_MATRIX.get(session_token, [])
    if not dataset:
        return HTMLResponse(
            content="<h3>Error: Target Matrix Context Expired.</h3>", status_code=404
        )

    df = pd.DataFrame(dataset)
    export_columns = [
        "file_name",
        "variant_method",
        "media_type",
        "extracted_frames_count",
        "max_frames_budget",
        "time_original",
        "time_processed",
        "latency_diff",
        "original_bert",
        "original_bleu",
        "original_clip",
        "processed_bert",
        "processed_bleu",
        "processed_clip",
        "token_original",
        "token_processed",
    ]
    valid_cols = [c for c in export_columns if c in df.columns]
    df_filtered = df[valid_cols].copy()
    df_filtered.columns = [c.replace("_", " ").title() for c in valid_cols]

    output_stream = BytesIO()
    with pd.ExcelWriter(output_stream, engine="openpyxl") as writer:
        df_filtered.to_excel(
            writer, index=False, sheet_name="VLM Batch Evaluation Matrix"
        )

    output_stream.seek(0)
    return StreamingResponse(
        output_stream,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            "Content-Disposition": (
                "attachment; filename=VLM_Robustness_Batch_Report.xlsx"
            )
        },
    )