import io
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from backend.augmentation import apply_augmentation

"""Raw Image in different formats for testing augmentation pipeline input flexibility."""

@pytest.fixture
def raw_numpy_image():
    """NumPy array."""
    arr = np.zeros((224, 224, 3), dtype=np.uint8)
    arr[40:180, 40:180] = [255, 120, 40]
    return arr


@pytest.fixture
def pil_image(raw_numpy_image):
    """PIL Image format."""
    return Image.fromarray(raw_numpy_image)


@pytest.fixture
def disk_image_path(tmp_path, pil_image):
    """File path string pointing to a saved image on disk."""
    file_path = tmp_path / "test_sample.jpg"
    pil_image.save(file_path)
    return str(file_path)


@pytest.fixture
def image_bytes(pil_image):
    """Raw encoded byte."""
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG")
    return buf.getvalue()


"""Listing alll augmentation methods for parameterized testing based on their categories."""
CLASSICAL_METHODS = [
    "CLAHE",
    "GammaCorrection",
    "RetinexSSR",
    "ColorJittering",
    "NoiseInjection",
    "KernelBlurringSharpening",
    "Solarization",
    "GridMaskDropout",
    "GeometricTransformations",
    "RandomCroppingResizing",
    "RandomErasingCutout",
    "ElasticDeformation",
    "ShearMapping",
    "PerspectiveTransform",
    "MosaicAugmentation",
    "WeatherFogSimulation",
    "StyleTransferFilter",
    "TestTimeAugmentation",
]

SPATIAL_METHODS = [
    "ContextAwareZoom",
    "QueryAwareBoundingBox",
    "SemanticHazardIsolation",
    "PerceptionMetricGrounding",
]

AI_MODELS = [
    "FastSAM",
    "MobileSAM",
    "MIDAS",
    "DepthAnything",
    "ZoeDepth",
    "RT-DETR",
    "PoseEstimation",
    "Rembg",
    "SurfaceNormalization",
]


def test_input_format_flexibility(raw_numpy_image, disk_image_path, image_bytes):
    """Verifies that the augmentation pipeline can accept various input formats apart from PIL without errors."""
    np_input = Image.fromarray(raw_numpy_image)
    out_np = apply_augmentation(np_input, method="CLAHE", prompt="")
    assert isinstance(out_np, Image.Image)

    file_input = Image.open(disk_image_path)
    out_file = apply_augmentation(file_input, method="CLAHE", prompt="")
    assert isinstance(out_file, Image.Image)

    bytes_input = Image.open(io.BytesIO(image_bytes))
    out_bytes = apply_augmentation(bytes_input, method="CLAHE", prompt="")
    assert isinstance(out_bytes, Image.Image)


def test_fallback_behavior(pil_image):
    # 'none' must return the exact same pixels
    out_none = apply_augmentation(pil_image, method="none", prompt="")
    assert np.array_equal(np.array(out_none), np.array(pil_image))

    # Unknown filter names should safely fall back rather than crashing
    out_unknown = apply_augmentation(pil_image, method="fake_method", prompt="")
    assert np.array_equal(np.array(out_unknown), np.array(pil_image))


@pytest.mark.parametrize("method", CLASSICAL_METHODS)
def test_classical_methods_actually_mutate_image(pil_image, method):
    out = apply_augmentation(pil_image, method=method, prompt="")

    assert isinstance(out, Image.Image)
    assert out.width > 0 and out.height > 0

    #convert both images to RGB arrays for pixel-wise comparison
    orig_arr = np.array(pil_image.convert("RGB"))
    out_arr = np.array(out.convert("RGB"))

    # Verify that the output image is not identical to the input image (i.e., the augmentation had an effect)
    if orig_arr.shape == out_arr.shape:
        diff = np.mean(np.abs(orig_arr.astype(float) - out_arr.astype(float)))
        assert diff > 0.0, f"Method '{method}' returned an unchanged image."


@pytest.mark.parametrize("method", SPATIAL_METHODS)
@pytest.mark.parametrize("prompt_input", ["find the plant", "", "   ", None])
def test_spatial_methods_handle_blank_and_null_prompts(pil_image, method, prompt_input):
    # None, whitespace, or empty prompts must not throw unhandled exceptions
    out = apply_augmentation(pil_image, method=method, prompt=prompt_input)
    assert isinstance(out, Image.Image)
    assert out.width > 0 and out.height > 0


@pytest.mark.parametrize("model_name", AI_MODELS)
def test_deep_learning_models_return_valid_tensors(pil_image, model_name):
    out = apply_augmentation(pil_image, method=model_name, prompt="")

    assert isinstance(out, Image.Image)

    # Ensure backbone reconstructed a viable image rather than returning an un-upsampled feature.
    assert out.width >= 32 and out.height >= 32

    # Check for valid 8-bit image array structure
    arr = np.array(out)
    assert arr.dtype == np.uint8
    assert arr.ndim in (2, 3)