# Test Suite: VLM Robustness & Evaluation Framework

This directory contains the automated test suite for verifying the Vision-Language Model (VLM) evaluation system. It validates data processing pipelines, image augmentations, algorithmic keyframe extractors, and multimodal evaluation metrics.

---

## Directory Organization

```text
tests/
├── README.py               # Documentation of testing protocols and known limitations
├── Scripts/                # Individual test modules
│   ├── test_augmentation.py       # Visual filters, transforms, and neural backbones
│   ├── test_evaluation.py         # BLEU, BERTScore, CLIPScore, latency, and tokens
│   └── test_frames_extraction.py  # 20 keyframe sampling algorithms & mathematical checks
└── Report/                 # Generated standalone HTML test run summaries
    ├── report-aug.html     # Augmentation test report
    ├── report-eval.html    # Metrics and evaluation report
    ├── report-frames.html  # Keyframe extraction report
    └── final_report.html   # Consolidated test execution report
```

---

## What We Tested

### 1. Augmentation Engine (`test_augmentation.py`)
* **Input Flexibility**: Verified that image inputs can be provided as raw NumPy matrices (`np.ndarray`), disk paths (`str` / `pathlib.Path`), raw byte streams (`io.BytesIO`), or standard PIL images without conversion failures.
* **Classical Filters**: Validated 18 classical computer vision methods (CLAHE, GammaCorrection, RetinexSSR, ElasticDeformation, etc.) to ensure that pixel matrices are mathematically modified rather than passed through unaltered.
* **Defensive Fallbacks**: Verified that specifying `"none"` or unrecognized dummy filter names falls back safely to returning the original image.
* **Prompt Resilience**: Tested spatial methods (`ContextAwareZoom`, `QueryAwareBoundingBox`, `SemanticHazardIsolation`, `PerceptionMetricGrounding`) against empty strings (`""`), whitespace-only queries, and `None` to ensure cropping/bounding boxes do not collapse into 0x0 dimensions.
* **Deep Learning Backbones**: Verified that neural vision models (FastSAM, MobileSAM, DepthAnything, ZoeDepth, RT-DETR, etc.) reconstruct valid 8-bit unsigned images (`uint8`) with output dimensions >= 32x32, confirming that internal feature maps are upsampled out of bottleneck layers.

### 2. Evaluation Metrics (`test_evaluation.py`)
* **Timing & Tokens**: Verified non-negative duration calculations, symmetric latency difference mathematics, and exact whitespace/punctuation token counts.
* **BLEU Bounds**:
  * Exact matches on sentences >= 4 words evaluate to 1.0.
  * Orthogonal vocabularies (zero shared words) evaluate strictly to 0.0.
  * Monotonic score degradation holds as text is progressively truncated or substituted.
  * Verified short sentence exact matches (< 4 words) using adjusted n-gram weights.
* **BERTScore Hierarchy**: Tested that true semantic synonyms score strictly higher than unrelated sentences or explicit logical negations, while preventing tokenizer crashes on empty candidate strings.
* **CLIPScore Visual Grounding**: Validated that image-text alignment yields significantly higher scores than mismatched visual concepts, while missing files or empty strings fail safely to 0.0 instead of throwing unhandled FileNotFoundError crashes.
* **Pipeline Integration**: Verified the complete output contract of `evaluate_outputs()`, including metrics mappings, ground truth matching, and zero-score fallbacks when no media is provided.

### 3. Keyframe Extraction (`test_frames_extraction.py`)
* **Algorithmic Correctness (Laplacian)**: Built a synthetic 30-frame benchmark video (flat dark frames, high-variance checkerboards, moving color spheres) to mathematically prove the `laplacian` sampler selects the maximum edge variance frame (> 20.0) rather than flat surfaces.
* **Visual Diversity**: Verified that `histogram` and `ssim` extractors select distinct scenes (mean pixel divergence > 15.0) rather than adjacent duplicates when the budget is > 1.
* **20 Registered Extractors**: Executed all active extractors against frame budgets (K <= 3), verifying that frames saved to disk are non-empty and readable by image decoders.
* **Query Resilience**: Tested prompt-driven extractors (`bolt`, `qa_iframes`, `keyvideollm`, `smart_sampling`) against `None` and blank prompts.
* **Defensive Failures**: Confirmed invalid extractor names raise explicit ValueError exceptions.

---

## What Was Not Possible (Known Limitations & Edge Cases)

1. **Simultaneous Multi-Suite Execution (Windows Memory Access Violations)**:
   Running all test modules concurrently in a single Pytest session causes C-level access violations (`0xC0000005` in `torch.storage.__getitem__`). This is caused by PyTorch and Hugging Face's `transformers` multi-threaded tensor loading (`_materialize_copy` / `spawn_materialize`) conflicting with Python 3.12 under Windows. To avoid memory fragmentation, test modules are run individually.
2. **Full End-to-End API Multipart Streaming (`main.py`)**:
   Direct API endpoint testing via FastAPI's `TestClient` was omitted from automated regression due to relative static path resolution conflicts (`../static` vs `Software/static`) when launched from differing parent directory depths. Core testing is focused on backend logic modules (`augmentation`, `evaluation`, and `FramesExtraction`).
3. **Strict Unsmoothed BLEU-4 on Short Strings**:
   Standard un-smoothed BLEU-4 mathematically evaluates to 0.0 for any sentence shorter than 4 tokens because the 4-gram precision (p_4) is zero. To test short text exact matches, dynamic length-dependent n-gram weights or smoothing functions were required.

---

## How to Run Tests

Ensure the workspace root has `pythonpath = . backend backend/Augmentify` configured in `pytest.ini`.

### Run Augmentation Suite
```powershell
pytest tests/Scripts/test_augmentation.py --html=tests/Report/report-aug.html --self-contained-html -v
```

### Run Evaluation Suite
```powershell
pytest tests/Scripts/test_evaluation.py --html=tests/Report/report-eval.html --self-contained-html -v
```

### Run Frame Extraction Suite
```powershell
pytest tests/Scripts/test_frames_extraction.py --html=tests/Report/report-frames.html --self-contained-html -v
```

---

## Test Execution Reports

HTML reports generated from test executions are available in the `Report/` folder:

* [Augmentation Test Report](./Report/report-aug.html)
* [Evaluation Test Report](./Report/report-eval.html)
* [Frame Extraction Test Report](./Report/report-frames.html)
* [Consolidated Final Report](./Report/final_report.html)