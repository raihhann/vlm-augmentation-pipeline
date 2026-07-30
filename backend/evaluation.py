# evaluation.py
"""
Evaluation module for comparing model outputs.

Measures:
- Latency (inference time in milliseconds)
- Semantic similarity (BERTScore F1)
- Surface-level n-gram overlap (BLEU)
- Token statistics
"""

import time
from typing import Optional
from bert_score import BERTScorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Global cache for the lazy-loaded model
_BERT_SCORER_INSTANCE: Optional[BERTScorer] = None


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
    ground_truth: str
) -> dict:
    """
    Central function that aggregates all evaluation metrics.
    Returns a structured dictionary consumed by downstream services/dashboards.
    """
    # Fast batch pass
    bert_scores = compute_bert_scores([original_output, augmented_output], ground_truth)
    original_bert_score, augmented_bert_score = bert_scores[0], bert_scores[1]
    # print(f"Original BERTScore: {original_bert_score}, Augmented BERTScore: {augmented_bert_score} with ground truth: {ground_truth} original: {original_output} augmented: {augmented_output}")
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