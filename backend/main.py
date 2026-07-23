from io import BytesIO
import os
import time
from Augmentify.extract.video_extractor import extract_video_keyframes
from augmentation import apply_augmentation
from config import AVAILABLE_AUGMENTATIONS, AVAILABLE_MODELS
from evaluation import evaluate_outputs, measure_inference_time
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from inference import run_inference
from PIL import Image

import pandas as pd

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
          "augmentations": AVAILABLE_AUGMENTATIONS,
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
      },
  )


@app.post("/process")
async def process_media_matrix(
    request: Request,
    processing_mode: str = Form(...),
    model_name: str = Form(...),
    # Image Mode inputs
    image_files: list[UploadFile] = File(None),
    image_prompts: list[str] = Form(None),
    image_ground_truths: list[str] = Form(None),
    augmentation_methods: list[str] = Form(None),
    # Video Mode matrix array inputs
    video_files: list[UploadFile] = File(None),
    video_prompts: list[str] = Form(None),
    video_ground_truths: list[str] = Form(None),
    video_sampling_methods: list[str] = Form(None),
    augmentation_methods_video: list[str] = Form(None),
    # Excel Dataset Batch Mode inputs
    excel_file: UploadFile = File(None),
    excel_image_dir: str = Form(None),
    excel_col_filename: str = Form(None),
    excel_col_prompt: str = Form(None),
    excel_col_gt: str = Form(None),
    excel_extra_cols: list[str] = Form(None),
):

  async def batch_event_generator():
    import asyncio

    session_token = f"matrix_session_{int(time.time())}"
    GLOBAL_SESSION_STORAGE_MATRIX[session_token] = []

    yield "data: BATCH ROW RUNWAY INITIALIZED\n\n"
    await asyncio.sleep(0.01)

    os.makedirs("../static/uploads", exist_ok=True)
    os.makedirs("../static/cache", exist_ok=True)  # Create cache folder
    model = model_name

    # ==========================================================
    # 📊 EXCEL BATCH DATASET MODE (INCREMENTAL DISK CACHING)
    # ==========================================================
    if processing_mode == "excel":
      yield "data: Reading dataset file...\n\n"
      await asyncio.sleep(0.01)

      excel_bytes = await excel_file.read()
      df = pd.read_excel(BytesIO(excel_bytes))

      augs_pool = augmentation_methods if augmentation_methods else ["none"]
      extra_cols_list = excel_extra_cols if excel_extra_cols else []
      total_rows = len(df)
      total_runs = total_rows * len(augs_pool)

      # Dedicated incremental CSV cache file on disk
      cache_csv_path = f"../static/cache/dataset_eval_{session_token}.csv"

      yield (
          f"data: Loaded {total_rows} rows. Total evaluation runs:"
          f" {total_runs}. Cache file: dataset_eval_{session_token}.csv\n\n"
      )
      await asyncio.sleep(0.01)

      for idx, row in df.iterrows():
        img_name = str(row[excel_col_filename]).strip()
        current_prompt = str(row[excel_col_prompt]).strip()
        current_gt = str(row[excel_col_gt]).strip()

        # Extract extra dataset metadata columns requested by user
        extra_data_dict = {}
        for col in extra_cols_list:
          if col in row:
            extra_data_dict[col] = str(row[col])

        # Resolve image disk path from local folder path
        media_path_disk = os.path.join(excel_image_dir, img_name)

        # ERROR CONTROL 1: Missing image check
        if not os.path.exists(media_path_disk):
          yield (
              f"data: [Warning] Image file not found: {media_path_disk}."
              " Skipping row...\n\n"
          )
          await asyncio.sleep(0.01)
          continue

        # ERROR CONTROL 2: Image opening check
        try:
          pil_image = Image.open(media_path_disk).convert("RGB")
        except Exception as e:
          yield (
              f"data: [Error] Could not open image {img_name}: {str(e)}."
              " Skipping row...\n\n"
          )
          await asyncio.sleep(0.01)
          continue

        for single_aug in augs_pool:
          yield (
              f"data: Processing [{idx + 1}/{total_rows}]: {img_name} ──>"
              f" Method: {single_aug}...\n\n"
          )
          await asyncio.sleep(0.01)

          # ERROR CONTROL 3: Pipeline Execution & Model Inference Protection
          try:
            start_aug = time.time()
            augmented_image = apply_augmentation(
                pil_image, single_aug, current_prompt
            )

            augmented_path_disk = (
                f"../static/uploads/batch_{idx}_{single_aug}_{img_name}"
            )
            augmented_path_web = (
                f"/static/uploads/batch_{idx}_{single_aug}_{img_name}"
            )
            augmented_image.save(augmented_path_disk)

            augmented_output = run_inference(
                model, augmented_path_web, current_prompt, "augmented"
            )
            end_aug = time.time()

            start_original = time.time()
            original_output = run_inference(
                model, media_path_disk, current_prompt, "original"
            )
            end_original = time.time()

            t_orig = measure_inference_time(start_original, end_original)
            t_proc = measure_inference_time(start_aug, end_aug)

            eval_res = evaluate_outputs(
                original_output, augmented_output, t_orig, t_proc, current_gt
            )

            row_record = {
                "Image Filename": img_name,
                "Evaluation Method": single_aug,
                "Evaluation Prompt": current_prompt,
                "Ground Truth Reference": current_gt,
                **extra_data_dict,  # Extra dataset columns
                "Baseline Output": original_output,
                "Augmented Output": augmented_output,
                "Baseline Latency (s)": t_orig,
                "Augmented Latency (s)": t_proc,
                "Latency Delta Difference (s)": eval_res["latency_diff"],
                "Baseline BERTScore": eval_res["original_bert_score"],
                "Baseline BLEU Score": eval_res["original_bleu_score"],
                "Augmented BERTScore": eval_res["augmented_bert_score"],
                "Augmented BLEU Score": eval_res["augmented_bleu_score"],
                "Baseline Tokens": eval_res["token_count_original"],
                "Augmented Tokens": eval_res["token_count_augmented"],
            }

            # 1. Store in active session memory
            GLOBAL_SESSION_STORAGE_MATRIX[session_token].append(row_record)

            # 2. 💾 INCREMENTAL DISK CACHING: Save row directly to CSV immediately!
            row_df = pd.DataFrame([row_record])
            file_exists = os.path.exists(cache_csv_path)
            row_df.to_csv(
                cache_csv_path, mode="a", index=False, header=not file_exists
            )

          except Exception as exc:
            yield (
                f"data: [Pipeline Error] Exception on {img_name} under"
                f" {single_aug}: {str(exc)}. Skipping step...\n\n"
            )
            await asyncio.sleep(0.01)
            continue

      # Signal client to download the final cached CSV
      yield f"data: DOWNLOAD_TOKEN:{session_token}\n\n"
      yield "data: DONE\n\n"

    # ==========================================================
    # 📷 BATCH MULTI-IMAGE MULTI-AUGMENTATION MATRIX
    # ==========================================================
    elif processing_mode == "image":
      files_pool = image_files if image_files else []
      augs_pool = augmentation_methods if augmentation_methods else ["none"]

      total_cells = len(files_pool) * len(augs_pool)
      yield (
          "data: Batch parameters matched. Processing execution grid matrix"
          f" size: {total_cells} Cells\n\n"
      )
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

        for single_aug in augs_pool:
          yield (
              f"data: File [{idx + 1}/{len(files_pool)}]: {filename} ──>"
              f" Checking Method: {single_aug}...\n\n"
          )
          await asyncio.sleep(0.01)

          pil_image = Image.open(BytesIO(raw_bytes)).convert("RGB")

          start_aug = time.time()
          augmented_image = apply_augmentation(
              pil_image, single_aug, current_prompt
          )

          augmented_path_disk = (
              f"../static/uploads/cell_{idx}_{single_aug}_{filename}"
          )
          augmented_path_web = (
              f"/static/uploads/cell_{idx}_{single_aug}_{filename}"
          )
          augmented_image.save(augmented_path_disk)

          augmented_output = run_inference(
              model, augmented_path_web, current_prompt, "augmented"
          )
          end_aug = time.time()

          start_original = time.time()
          original_output = run_inference(
              model, media_path_web, current_prompt, "original"
          )
          end_original = time.time()

          t_orig = measure_inference_time(start_original, end_original)
          t_proc = measure_inference_time(start_aug, end_aug)

          eval_res = evaluate_outputs(
              original_output, augmented_output, t_orig, t_proc, current_gt
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
              "processed_bert": eval_res["augmented_bert_score"],
              "processed_bleu": eval_res["augmented_bleu_score"],
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
    # 🎬 BATCH MULTI-VIDEO MULTI-SAMPLING MULTI-AUGMENTATION MATRIX
    # ==========================================================
    else:
      v_files_pool = video_files if video_files else []
      samplers_pool = (
          video_sampling_methods if video_sampling_methods else ["1"]
      )
      v_augs_pool = (
          augmentation_methods_video if augmentation_methods_video else ["none"]
      )

      total_v_cells = len(v_files_pool) * len(samplers_pool) * len(v_augs_pool)
      yield (
          "data: Video parameters matched. Processing complex evaluation"
          f" matrix size: {total_v_cells} Work Cells\n\n"
      )
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

        for current_sampler in samplers_pool:
          video_out_disk = f"../static/uploads/extracted_{idx}_{current_sampler}_{int(time.time())}"

          yield (
              f"data: Video [{idx + 1}/{len(v_files_pool)}]: {filename} ──>"
              f" Keyframe Strategy: Method {current_sampler}...\n\n"
          )
          await asyncio.sleep(0.01)

          extracted_frames = extract_video_keyframes(
              video_path=media_path_disk,
              method=current_sampler,
              query=current_prompt,
              output_dir=video_out_disk,
          )

          if not extracted_frames:
            continue

          for current_aug in v_augs_pool:
            yield (
                f"data: Video [{idx + 1}/{len(v_files_pool)}]: {filename}"
                f" [Method {current_sampler}] ──> Applying 2D Filter:"
                f" {current_aug}...\n\n"
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

              web_accessible_url = aug_f_path.replace(
                  "../static", "/static"
              ).replace("\\", "/")
              augmented_frames_web_pool.append(web_accessible_url)

            augmented_output = run_inference(
                model, augmented_frames_web_pool[0], current_prompt, "augmented"
            )
            end_aug = time.time()

            start_original = time.time()
            original_output = run_inference(
                model, media_path_web, current_prompt, "original"
            )
            end_original = time.time()

            t_orig = measure_inference_time(start_original, end_original)
            t_proc = measure_inference_time(start_aug, end_aug)
            eval_res = evaluate_outputs(
                original_output, augmented_output, t_orig, t_proc, current_gt
            )

            extracted_keyframes_web = [
                p.replace("../static", "/static").replace("\\", "/")
                for p in extracted_frames
            ]

            GLOBAL_SESSION_STORAGE_MATRIX[session_token].append({
                "file_name": filename,
                "variant_method": f"Method {current_sampler} + {current_aug}",
                "media_type": "video",
                "media_path_web": media_path_web,
                "extracted_keyframes": extracted_keyframes_web,
                "augmented_keyframes": augmented_frames_web_pool,
                "augmented_media_path": augmented_frames_web_pool[0],
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
  # Check if disk cache file exists first
  cache_csv_path = f"../static/cache/dataset_eval_{session_token}.csv"

  if os.path.exists(cache_csv_path):
    with open(cache_csv_path, "rb") as f:
      csv_bytes = f.read()

    return StreamingResponse(
        BytesIO(csv_bytes),
        media_type="text/csv",
        headers={
            "Content-Disposition": (
                f"attachment;"
                f" filename=VLM_Batch_Evaluation_Results_{session_token}.csv"
            )
        },
    )

  # Fallback to session storage if cache file isn't present
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
              f"attachment;"
              f" filename=VLM_Batch_Evaluation_Results_{session_token}.csv"
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
      "time_original",
      "time_processed",
      "latency_diff",
      "original_bert",
      "original_bleu",
      "processed_bert",
      "processed_bleu",
      "token_original",
      "token_processed",
  ]
  df_filtered = df[export_columns].copy()
  df_filtered.columns = [c.replace("_", " ").title() for c in export_columns]

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