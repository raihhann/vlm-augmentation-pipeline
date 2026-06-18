# evaluation.py
"""
Evaluation module for comparing model outputs.

This file will:
- Measure inference time
- Compute text similarity metrics
- Compute token statistics
- Return structured evaluation results

For now: returns mock values.
Later: replace mock logic with real metrics.
"""

import time
import random
from bert_score import score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction



# ==========================================
# Inference Time Measurement
# ==========================================
def measure_inference_time(start_time: float, end_time: float) -> float:
    """
    Computes total inference time in milliseconds..
    """
    return round((end_time - start_time), 2)


# ==========================================
# Mock BERTScore
# ==========================================
def compute_bert_score(prediction: str, ground_truth: str) -> float:
    """Calculates semantic similarity using BERT embeddings."""
    print("Calculating BertScore")
    P, R, F1 = score([prediction], [ground_truth] ,lang="en", model_type="distilbert-base-uncased", verbose=False)
    print(f"Computed BERTScore - P: {P.item():.4f}, R: {R.item():.4f}, F1: {F1.item():.4f}")
    return F1.item()


# ==========================================
# Mock BLEU Score
# ==========================================
def compute_bleu(prediction: str, ground_truth: str) -> float:
    """Calculates linguistic overlap (n-gram similarity)."""
    chencherry = SmoothingFunction()
    # BLEU expects lists of tokens
    ref_tokens = [ground_truth.split()]
    cand_tokens = prediction.split()
    return sentence_bleu(ref_tokens, cand_tokens, smoothing_function=chencherry.method1)


# ==========================================
# Token Count
# ==========================================
def compute_token_count(text: str) -> int:
    """
    Basic token count (placeholder).
    Later: use tokenizer from selected model.
    """
    return len(text.split())


# ==========================================
# Latency Difference
# ==========================================
def compute_latency_difference(t1: float, t2: float) -> float:
    """
    Difference between two inference times (ms).
    """
    return round(abs(t1 - t2), 2)


# ==========================================
# Main Evaluation Pipeline
# ==========================================
def evaluate_outputs(original_output: str,
                     augmented_output: str,
                     inference_time_original: float,
                     inference_time_augmented: float,
                     ground_truth: str) -> dict:
    """
    Central function that aggregates all evaluation metrics.
    Returns a dictionary that dashboard will consume.
    """

    original_bert_score = compute_bert_score(original_output, ground_truth)
    augmented_bert_score = compute_bert_score(augmented_output, ground_truth)
    original_bleu_score = compute_bleu(original_output, ground_truth)
    augmented_bleu_score = compute_bleu(augmented_output, ground_truth)

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
        "token_count_original": token_count_original,
        "token_count_augmented": token_count_augmented,
        "latency_diff": latency_diff
    }