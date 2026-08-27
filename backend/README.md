# Backend

The `backend` directory contains the FastAPI service and the Python modules that run inference, image augmentation, video frame extraction, and output evaluation.

## Runtime entry point

[`main.py`](main.py) creates the FastAPI application. It serves the frontend pages, accepts uploaded media and spreadsheet datasets, streams progress, stores session results, and exposes CSV/Excel downloads.

Start it from this directory so the relative paths used by the application resolve correctly:

```powershell
python -m uvicorn main:app --reload
```

The application expects the repository-level `frontend/` and `static/` directories to be available one level above `backend/`.

## Request flow

For an image run, the service saves the upload, performs baseline inference, applies the selected augmentation, performs processed inference, computes evaluation metrics, and adds the result to the current session.

For a video run, it performs baseline inference, extracts keyframes with the selected strategy, applies any selected image augmentation to those frames, runs processed inference, and records the comparison.

For spreadsheet workflows, rows provide media paths, prompts, ground truths, and optional metadata. Results are appended to runtime CSV caches as the batch progresses, which allows completed work to be exported after processing.

## Python modules

| File | Responsibility |
| --- | --- |
| [`main.py`](main.py) | FastAPI routes, upload handling, processing generators, dashboards, and report downloads. |
| [`config.py`](config.py) | Lists available Ollama models, enabled image augmentations, and registered extraction methods. |
| [`inference.py`](inference.py) | Calls Ollama for VLM responses and handles the `mock` inference option. |
| [`augmentation.py`](augmentation.py) | Dispatches a configured image method to an Augmentify implementation. |
| [`evaluation.py`](evaluation.py) | Computes BERTScore, BLEU, CLIPScore, latency differences, and token counts. |
| [`model_manager.py`](model_manager.py) | Provides an Ollama model-preloading helper; it is not currently used by the main route path. |
| [`Augmentify/`](Augmentify/README.md) | Contains augmentation and video extraction implementations. |

## API routes

- `GET /` serves the manual evaluation interface.
- `GET /excel_studio` serves the spreadsheet batch interface.
- `POST /process` starts a processing workflow and streams Server-Sent Events.
- `GET /download_csv/{session_token}` returns the session as CSV.
- `GET /download_metrics?session_token=...` returns the session as an Excel workbook.

## Configuration and models

The models used by `inference.py` are local Ollama models. The configured names currently include `qwen3-vl:4b`, `llava-phi3`, and `mock`. The evaluation module downloads or loads BERTScore and CLIP resources as needed. Model-based augmentations use weights stored in or downloaded to the repository-level `models/` directory.

The two `.pt` files directly under `backend/` are legacy or backend-local model assets. Other resources are organized under `models/`; avoid duplicating large weights without a clear reason.

## Runtime output

Uploaded and processed media are written to `../static/uploads/`. Batch CSV caches are written to `../static/cache/`. These are experiment artifacts, not source files. The in-memory `GLOBAL_SESSION_STORAGE_MATRIX` is cleared when the server restarts, although cached batch CSV files may remain available.

## Adding backend functionality

Keep orchestration in `main.py` and place algorithmic work in `Augmentify`, `inference.py`, or `evaluation.py` according to its responsibility. Add a module-level docstring to every new Python file, update `config.py` only for methods that are actually callable, and verify the complete workflow with a small representative media file.
