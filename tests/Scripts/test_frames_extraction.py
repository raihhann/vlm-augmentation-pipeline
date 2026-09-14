import os
import cv2
import numpy as np
import pytest
from PIL import Image

from Augmentify.FramesExtraction import extract_frames, FRAME_EXTRACTORS


# Filter out experimental/unregistered modules
ALL_METHODS = [
    m for m in FRAME_EXTRACTORS.keys() 
    if m not in ["super_smart_sampling"]
]

# Methods requiring prompt handling
TEXT_DRIVEN_METHODS = [
    m for m in ["bolt", "qa_iframes", "keyvideollm", "smart_sampling"]
    if m in FRAME_EXTRACTORS
]


@pytest.fixture(scope="module")
def sample_video(tmp_path_factory):
    # Generates a synthetic 30-frame video with 3 clear scene changes:
    # 1. Dark flat frames (low edge variance)
    # 2. High-contrast checkerboard (high edge variance)
    # 3. Cyan background with a moving red ball
    out_dir = tmp_path_factory.mktemp("video_data")
    video_path = str(out_dir / "sample_action.mp4")

    w, h = 320, 240
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))

    # Part 1: Blank dark scene
    for _ in range(10):
        writer.write(np.full((h, w, 3), 15, dtype=np.uint8))

    # Part 2: Sharp checkerboard grid
    grid = np.full((h, w, 3), 15, dtype=np.uint8)
    grid[::15, :] = 240
    grid[:, ::15] = 240
    for _ in range(10):
        writer.write(grid)

    # Part 3: Colored motion scene
    for i in range(10):
        frame = np.full((h, w, 3), [20, 180, 220], dtype=np.uint8)
        cv2.circle(frame, (40 + (i * 20), 120), 25, (0, 0, 255), -1)
        writer.write(frame)

    writer.release()
    return video_path


# --- Mathematical Verification ---

def test_laplacian_prefers_sharp_frames(sample_video, tmp_path):
    # Laplacian filter should isolate the checkerboard over the flat frames
    save_folder = str(tmp_path / "laplacian_out")

    frames = extract_frames(
        video_path=sample_video,
        method="laplacian",
        max_frames=1,
        prompt="focus on sharp patterns",
        output_folder=save_folder,
    )

    assert len(frames) == 1
    gray = np.array(Image.open(frames[0]).convert("L"))
    edge_variance = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Flat frames have near-zero variance; checkerboards easily exceed 20.0
    assert edge_variance > 20.0


# --- Diversity Verification ---

@pytest.mark.parametrize("method", ["histogram", "ssim"])
def test_scene_change_diversity(sample_video, tmp_path, method):
    # Budget of 2 should extract visually distinct scenes rather than duplicates
    save_folder = str(tmp_path / f"diversity_{method}")

    frames = extract_frames(
        video_path=sample_video,
        method=method,
        max_frames=2,
        prompt="track distinct scene changes",
        output_folder=save_folder,
    )

    assert len(frames) >= 2

    frame_a = np.array(Image.open(frames[0])).astype(float)
    frame_b = np.array(Image.open(frames[1])).astype(float)

    # Mean absolute difference between distinct scenes should be noticeable
    pixel_delta = np.mean(np.abs(frame_a - frame_b))
    assert pixel_delta > 15.0


# --- Core Execution & Frame Budgets ---

@pytest.mark.parametrize("method", ALL_METHODS)
def test_all_registered_extractors(sample_video, tmp_path, method):
    save_folder = str(tmp_path / f"run_{method}")
    budget = 3

    frames = extract_frames(
        video_path=sample_video,
        method=method,
        max_frames=budget,
        prompt="red circle moving across screen",
        output_folder=save_folder,
    )

    assert isinstance(frames, list)
    # Never exceed requested budget
    assert 1 <= len(frames) <= budget

    for path in frames:
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

        with Image.open(path) as img:
            assert img.width > 0 and img.height > 0


# --- Prompt Boundary Resilience ---

@pytest.mark.parametrize("method", TEXT_DRIVEN_METHODS)
@pytest.mark.parametrize("prompt_val", [None, "", "   "])
def test_text_extractors_handle_empty_prompts(sample_video, tmp_path, method, prompt_val):
    # Empty or whitespace prompts must not trigger string formatting crashes
    save_folder = str(tmp_path / f"prompt_edge_{method}")

    frames = extract_frames(
        video_path=sample_video,
        method=method,
        max_frames=2,
        prompt=prompt_val,
        output_folder=save_folder,
    )

    assert isinstance(frames, list)
    assert 1 <= len(frames) <= 2


# --- Defensive Failure ---

def test_unknown_method_raises_value_error(sample_video, tmp_path):
    save_folder = str(tmp_path / "invalid_run")

    with pytest.raises(ValueError, match="is invalid"):
        extract_frames(
            video_path=sample_video,
            method="fake_extractor_name",
            max_frames=2,
            output_folder=save_folder,
        )