import numpy as np
import pytest
from PIL import Image

from backend.evaluation import (
    compute_bert_scores,
    compute_bleu,
    compute_clip_score,
    compute_latency_difference,
    compute_token_count,
    evaluate_outputs,
    measure_inference_time,
)


# --- Latency & Token Counter Tests ---

def test_inference_time_math():
    assert measure_inference_time(10.0, 15.33) == 5.33
    assert measure_inference_time(0.0, 0.0) == 0.0
    assert measure_inference_time(1.111, 2.222) == 1.11


def test_latency_difference():
    # Order shouldn't matter (always positive delta)
    assert compute_latency_difference(10.5, 4.2) == 6.3
    assert compute_latency_difference(4.2, 10.5) == 6.3
    assert compute_latency_difference(8.75, 8.75) == 0.0


@pytest.mark.parametrize("text, expected", [
    ("The robot detected a red obstacle near the doorway.", 9),
    ("", 0),
    ("    ", 0),
    ("\t\n  \r", 0),
    ("word", 1),
    ("multi-word   spaced \t\n tokens", 3),
    ("multi - word spaced", 4),
    ("Detected 3 items, moving.", 4),
])
def test_token_counting(text, expected):
    assert compute_token_count(text) == expected


# --- BLEU Score Tests ---

def test_bleu_identical_text():
    text = "a small green plant on the desk"
    assert compute_bleu(text, text) == 1.0


def test_bleu_no_overlap_gives_zero():
    cand = "blue ocean waves crashing down"
    ref = "red mechanical gripper arm"
    assert compute_bleu(cand, ref) == 0.0


def test_bleu_empty_inputs():
    assert compute_bleu("", "a small green plant") == 0.0
    assert compute_bleu("a small green plant", "") == 0.0
    assert compute_bleu("", "") == 0.0


def test_bleu_degradation():
    ref = "the industrial robot arm picked up the metal cylinder quickly"
    high = "the industrial robot arm grabbed the metal cylinder quickly"
    mid = "the industrial robot picked a cylinder"
    low = "the robot"

    score_high = compute_bleu(high, ref)
    score_mid = compute_bleu(mid, ref)
    score_low = compute_bleu(low, ref)

    assert 1.0 > score_high > score_mid
    assert score_mid >= score_low >= 0.0


@pytest.mark.parametrize("text", [
    "chair",
    "red chair",
    "a red table",
])
def test_bleu_short_sentences(text):
    # Short exact matches (< 4 tokens) should stay positive with smoothing/weighting
    score = compute_bleu(text, text)
    assert 0.0 < score <= 1.0


# --- BERTScore Tests ---

def test_bert_score_blank_inputs():
    scores = compute_bert_scores(["", "   "], "valid reference sentence")
    assert scores == [0.0, 0.0]


def test_bert_score_ranking():
    gt = "a red fire extinguisher mounted on the brick wall"
    synonym = "a crimson fire extinguisher attached to the wall"
    unrelated = "a blue plastic bottle lying on the floor"
    negated = "there is no red fire extinguisher on the wall"

    scores = compute_bert_scores([synonym, unrelated, negated], gt)
    score_syn, score_unrel, score_neg = scores

    assert all(0.0 <= s <= 1.0 for s in scores)
    assert score_syn > score_unrel
    assert score_syn > score_neg


# --- CLIPScore Tests ---

@pytest.mark.parametrize("bad_text", ["", "   ", "nan", "NaN", None])
def test_clip_score_empty_text(bad_text):
    blank = Image.fromarray(np.zeros((50, 50, 3), dtype=np.uint8))
    assert compute_clip_score(blank, bad_text) == 0.0


def test_clip_score_missing_image():
    # Pass None and an intentionally non-existent file path to ensure missing inputs fail gracefully with 0.0 instead of raising FileNotFoundError
    assert compute_clip_score(None, "a photo of a cat") == 0.0
    assert compute_clip_score("missing_file.jpg", "a photo of a cat") == 0.0


def test_clip_score_color_match(tmp_path):
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    arr[:, :] = [255, 0, 0]
    img = Image.fromarray(arr)

    img_path = str(tmp_path / "red.jpg")
    img.save(img_path)

    match = compute_clip_score(img_path, "a solid bright red color")
    mismatch = compute_clip_score(img_path, "a forest covered in snow")

    assert match > mismatch
    assert match > 0.15


# --- Full Evaluation Pipeline Tests ---

def test_evaluate_outputs_summary_dict():
    canvas = Image.fromarray(np.full((80, 80, 3), 100, dtype=np.uint8))

    res = evaluate_outputs(
        original_output="a red chair near table",
        augmented_output="an office desk with a red wooden chair",
        inference_time_original=2.5,
        inference_time_augmented=4.1,
        ground_truth="a red chair near table",
        image_input=canvas,
    )

    expected_keys = [
        "original_bert_score",
        "augmented_bert_score",
        "original_bleu_score",
        "augmented_bleu_score",
        "original_clip_score",
        "augmented_clip_score",
        "token_count_original",
        "token_count_augmented",
        "latency_diff",
        "FutureWork : semantic_score",
    ]
    for key in expected_keys:
        assert key in res

    # 5 words vs 8 words
    assert res["token_count_original"] == 5
    assert res["token_count_augmented"] == 8

    # |2.5 - 4.1|
    assert res["latency_diff"] == 1.6

    # Baseline is identical to ground truth
    assert res["original_bleu_score"] == 1.0
    assert 0.0 < res["augmented_bleu_score"] < 1.0

    assert res["original_clip_score"] > 0.0
    assert res["augmented_clip_score"] > 0.0


def test_evaluate_outputs_without_image():
    res = evaluate_outputs(
        original_output="the robotic arm moved",
        augmented_output="the robotic arm stopped",
        inference_time_original=1.0,
        inference_time_augmented=1.2,
        ground_truth="the robotic arm moved",
        image_input=None,
    )

    assert res["original_clip_score"] == 0.0
    assert res["augmented_clip_score"] == 0.0