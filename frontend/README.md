# Frontend

The `frontend` directory contains the browser pages served by the FastAPI backend. These are standalone HTML interfaces that use embedded JavaScript to submit media or dataset requests, consume streamed progress events, and display evaluation results.

## Pages

| File | Purpose |
| --- | --- |
| [`index.html`](index.html) | Manual image and video evaluation studio. |
| [`excel_evaluator.html`](excel_evaluator.html) | Spreadsheet-driven image and video batch evaluation. |
| [`dashboard.html`](dashboard.html) | Results view for outputs, selected frames, and evaluation metrics. |

## Manual studio

`index.html` supports image and video modes. Users can upload multiple files, assign prompts and ground truths, choose a configured VLM, select image augmentations or video keyframe extractors, and set the maximum number of video frames. The page sends the form to `POST /process` and displays server-sent progress messages while the matrix is running.

## Spreadsheet studio

`excel_evaluator.html` reads spreadsheet headers in the browser and lets the user map columns for filenames, prompts, ground truths, and optional metadata. It supports image and video dataset modes and submits the selected mapping to the same processing endpoint. The backend resolves each media path and writes batch results to a runtime cache.

The spreadsheet must contain a column whose values identify the media files. Image/video directories are supplied separately through the interface. Ensure that the directory and filenames are valid from the backend process's environment.

## Dashboard

`dashboard.html` renders the completed session matrix. Depending on the workflow, it can show original media, augmented media, extracted video frames, baseline and processed VLM outputs, latency, BERTScore, BLEU, token counts, and export controls. CSV and Excel downloads are handled by backend routes using the session token.

## Backend contract

The pages depend on these routes:

- `GET /` returns `index.html`.
- `GET /excel_studio` returns `excel_evaluator.html`.
- `POST /process` accepts multipart form data and returns a `text/event-stream` response containing progress and rendered dashboard content.
- `GET /download_csv/{session_token}` downloads CSV results.
- `GET /download_metrics?session_token=...` downloads Excel results.

Available models, augmentations, and extraction labels are inserted into templates by the backend. Keep frontend option values aligned with the keys in `backend/config.py`.

## Static assets and paths

Shared static files are served under `/static`. Generated uploads and processed media are written to `/static/uploads`, while batch CSV caches are written to `/static/cache`. The backend mounts these paths from the repository-level `static/` directory.

Most page styling is embedded in the HTML files. [`static/styles.css`](../static/styles.css) is a shared static stylesheet available to the application, but page-specific styles may remain local to each interface where that matches the existing design.

## Frontend development notes

- Keep form field names synchronized with the parameters in `backend/main.py`.
- Preserve the event-stream parsing behavior when changing progress or dashboard rendering.
- Test both empty and multi-file selections, failed media paths, and expired session tokens.
- Avoid assuming that every augmentation or extraction implementation is enabled; the backend configuration is authoritative.
- The frontend is served by FastAPI, so open it through the running application rather than relying on direct file URLs for complete behavior.
