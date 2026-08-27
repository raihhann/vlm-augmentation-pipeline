"""Compute and aggregate metrics for comparing baseline and augmented outputs.

The module lazily caches BERTScore and CLIP resources, then reports text
similarity, image-text grounding, latency differences, and token counts.
"""

import os
import time
from typing import Optional, Union, Tuple
import torch
import torch.nn.functional as F
from PIL import Image
from bert_score import BERTScorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import CLIPModel, CLIPProcessor

# ==========================================
# Global Caches for Lazy-Loaded Models
# ==========================================
_BERT_SCORER_INSTANCE: Optional[BERTScorer] = None
_CLIP_MODEL_INSTANCE: Optional[CLIPModel] = None
_CLIP_PROCESSOR_INSTANCE: Optional[CLIPProcessor] = None
_DEVICE: Optional[str] = None


def get_device() -> str:
    """Detects available computing device (CUDA or CPU)."""
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return _DEVICE


def get_bert_scorer() -> BERTScorer:
    """
    Returns the singleton instance of BERTScorer.
    Loads the PyTorch weights into memory ONCE on first call.
    """
    global _BERT_SCORER_INSTANCE
    if _BERT_SCORER_INSTANCE is None:
        print("Initializing BERTScorer instance (loading weights)...")
        _BERT_SCORER_INSTANCE = BERTScorer(
            lang="en",
            model_type="distilbert-base-uncased"
        )
        print("BERTScorer initialized successfully.")
    return _BERT_SCORER_INSTANCE


def get_clip_resources() -> Tuple[CLIPModel, CLIPProcessor]:
    """
    Returns the singleton instances of CLIPModel and CLIPProcessor.
    Loads CLIP weights into GPU/CPU memory ONCE on first call.
    """
    global _CLIP_MODEL_INSTANCE, _CLIP_PROCESSOR_INSTANCE
    if _CLIP_MODEL_INSTANCE is None or _CLIP_PROCESSOR_INSTANCE is None:
        model_name = "openai/clip-vit-base-patch32"
        device = get_device()
        print(f"Initializing CLIPModel & Processor ('{model_name}') on {device.upper()}...")
        
        _CLIP_PROCESSOR_INSTANCE = CLIPProcessor.from_pretrained(model_name)
        _CLIP_MODEL_INSTANCE = CLIPModel.from_pretrained(model_name).to(device)
        _CLIP_MODEL_INSTANCE.eval()
        print("CLIP initialized successfully.")

    return _CLIP_MODEL_INSTANCE, _CLIP_PROCESSOR_INSTANCE


# ==========================================
# Inference Time Measurement
# ==========================================
def measure_inference_time(start_time: float, end_time: float) -> float:
    """Computes total inference time in milliseconds."""
    return round((end_time - start_time), 2)


def compute_latency_difference(t1: float, t2: float) -> float:
    """Difference between two inference times (ms)."""
    return round(abs(t1 - t2), 2)


# ==========================================
# Text Similarity Metrics
# ==========================================
def compute_bert_scores(predictions: list[str], ground_truth: str) -> list[float]:
    """
    Calculates BERTScore F1 for a batch of predictions against a single reference
    using the cached BERTScorer singleton.
    """
    scorer = get_bert_scorer()
    references = [ground_truth] * len(predictions)
    
    # Calculate scores on pre-loaded model
    _, _, F1 = scorer.score(predictions, references)
    
    return [round(f, 4) for f in F1.tolist()]


def compute_bleu(prediction: str, ground_truth: str) -> float:
    """Calculates linguistic overlap (n-gram similarity)."""
    chencherry = SmoothingFunction()
    ref_tokens = [ground_truth.split()]
    cand_tokens = prediction.split()
    score_val = sentence_bleu(ref_tokens, cand_tokens, smoothing_function=chencherry.method1)
    return round(score_val, 4)


# ==========================================
# Visual Grounding Metric (CLIPScore)
# ==========================================
def compute_clip_score(
    image_input: Optional[Union[str, Image.Image]], 
    text: str, 
    w: float = 2.5
) -> float:
    """
    Computes reference-free CLIPScore: w * max(cos(c, v), 0).
    Accepts either an image file path (str) or a PIL Image object.
    """
    if not text or str(text).strip() == "" or str(text).lower() == "nan" or image_input is None:
        return 0.0

    try:
        # Load PIL image if a path string is provided
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                return 0.0
            pil_img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        else:
            return 0.0

        model, processor = get_clip_resources()
        device = get_device()

        # Tokenize text and preprocess image
        inputs = processor(
            text=[str(text)[:300]], 
            images=pil_img, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            
            # L2 Normalize feature vectors
            img_embeds = F.normalize(outputs.image_embeds, p=2, dim=-1)
            text_embeds = F.normalize(outputs.text_embeds, p=2, dim=-1)
            
            # Cosine similarity
            similarity = (img_embeds @ text_embeds.T).squeeze().item()
            
            # Official CLIPScore formula: w * max(cos(c, v), 0)
            clip_score = w * max(similarity, 0.0)
            
            return round(clip_score, 4)

    except Exception as e:
        print(f"[Warning] CLIPScore computation failed: {e}")
        return 0.0


# ==========================================
# Token Count
# ==========================================
def compute_token_count(text: str) -> int:
    """Basic whitespace token count."""
    return len(text.split())


# ==========================================
# Main Evaluation Pipeline
# ==========================================
def evaluate_outputs(
    original_output: str,
    augmented_output: str,
    inference_time_original: float,
    inference_time_augmented: float,
    ground_truth: str,
    image_input: Optional[Union[str, Image.Image]] = None
) -> dict:
    """
    Central function that aggregates all evaluation metrics (BERTScore, BLEU, CLIPScore, Latency, Token Count).
    Returns a structured dictionary consumed by downstream services/dashboards.
    """
    # 1. Text Similarity Pass (BERTScore)
    bert_scores = compute_bert_scores([original_output, augmented_output], ground_truth)
    original_bert_score, augmented_bert_score = bert_scores[0], bert_scores[1]

    # 2. Surface Overlap Pass (BLEU)
    original_bleu_score = compute_bleu(original_output, ground_truth)
    augmented_bleu_score = compute_bleu(augmented_output, ground_truth)

    # 3. Visual Grounding Pass (CLIPScore)
    original_clip_score = compute_clip_score(image_input, original_output, w=2.5)
    augmented_clip_score = compute_clip_score(image_input, augmented_output, w=2.5)

    # 4. Token & Latency Statistics
    token_count_original = compute_token_count(original_output)
    token_count_augmented = compute_token_count(augmented_output)

    latency_diff = compute_latency_difference(
        inference_time_original,
        inference_time_augmented
    )

    return {
        "original_bert_score": original_bert_score,
        "augmented_bert_score": augmented_bert_score,
        "original_bleu_score": original_bleu_score,
        "augmented_bleu_score": augmented_bleu_score,
        "original_clip_score": original_clip_score,
        "augmented_clip_score": augmented_clip_score,
        "token_count_original": token_count_original,
        "token_count_augmented": token_count_augmented,
        "latency_diff": latency_diff
    }