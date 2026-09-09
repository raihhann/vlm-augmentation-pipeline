"""Dispatch configured image augmentations and return processed PIL images.

The module provides the integration point used by the web service and currently
routes enabled methods to their corresponding Augmentify implementations.
"""

from sys import implementation

from PIL import Image
from Augmentify.augment.fastSAM import run_fastsam
from Augmentify.augment.midas import run_midas_depth
from Augmentify.augment.mobileSAM import run_mobilesam
from Augmentify.augment.yolo_world_annotate import annotate
from Augmentify.augment.ContextAware_Zoom import run_automated_zoom
from Augmentify.augment.detr import run_rtdetr
from Augmentify.augment.depth_anything import run_depth_anything
from Augmentify.augment.geometric_segmentation import run_geometric_segmentation
from Augmentify.augment.Illumination import run_gamma_correction, run_clahe, run_retinex_ssr
from Augmentify.augment.pose_estimation import run_pose_estimation
from Augmentify.augment.rembg import run_rembg
from Augmentify.augment.SailencyCrop import run_saliency_crop
from Augmentify.augment.surface_normalization import run_surface_normals
from Augmentify.augment.zoe_depth import run_zoe_depth
from Augmentify.augment.geometric_transformation import run_geometric_transformations
from Augmentify.augment.random_cropping_resizing import run_random_cropping_resizing
from Augmentify.augment.color_jittering import run_color_jittering
from Augmentify.augment.noise_injection import run_noise_injection
from Augmentify.augment.kernel_blurring_sharpening import run_kernel_blurring_sharpening
from Augmentify.augment.random_erasing_cutout import run_random_erasing_cutout
from Augmentify.augment.elastic_deformation import run_elastic_deformation
from Augmentify.augment.shear_mapping import run_shear_mapping
from Augmentify.augment.perspective_transform import run_perspective_transform
from Augmentify.augment.mosaic_augmentation import run_mosaic_augmentation
from Augmentify.augment.solarization import run_solarization
from Augmentify.augment.weather_fog_simulation import run_weather_fog_simulation
from Augmentify.augment.gridmask_dropout import run_gridmask_dropout
from Augmentify.augment.style_transfer_filter import run_style_transfer_filter
from Augmentify.augment.test_time_augmentation import run_test_time_augmentation
from Augmentify.augment.query_aware_bounding_box import run_query_aware_bounding_box
from Augmentify.augment.semantic_hazard_isolation import run_semantic_hazard_isolation
from Augmentify.augment.perception_metric_grounding import run_perception_metric_grounding
from Augmentify.augment.scene_graph_genration import run_scene_graph_generator
from Augmentify.FutureWork.augmentation_selector import run_adaptive_policy_selection

def apply_augmentation(image: Image.Image, method: str, prompt: str) -> Image.Image:
    print(f"Applying augmentation: {method} with prompt: {prompt}")
    """Apply a specified image augmentation or processing method.

    Routes the input PIL image to the corresponding backend implementation 
    based on the provided method name, utilizing the text prompt when required.

    Args:
        image (Image.Image): The input PIL image to be processed.
        method (str): The name of the augmentation or transformation technique to apply.
        prompt (str): A text prompt used by context-aware or query-driven methods.

    Returns:
        Image.Image: The processed PIL image resulting from the augmentation.
    """

    if method == "FastSAM":
        return run_fastsam(image)
    
    elif method == "MobileSAM":
        return run_mobilesam(image)
    
    elif method == "MIDAS":
        return run_midas_depth(image)

    elif method == "Yolo_World_Annotate":
        return annotate(image)

    elif method == "ContextAwareZoom":
        return run_automated_zoom(image, prompt)

    elif method == "RT-DETR":
        return run_rtdetr(image)

    elif method == "DepthAnything":
        return run_depth_anything(image)

    elif method == "ZoeDepth":
        return run_zoe_depth(image)

    elif method == "GeometricSegmentation":
        return run_geometric_segmentation(image)

    elif method == "GammaCorrection":
        return run_gamma_correction(image)

    elif method == "CLAHE":
        return run_clahe(image)

    elif method == "RetinexSSR":
        return run_retinex_ssr(image)

    elif method == "PoseEstimation":
        return run_pose_estimation(image)
    
    elif method == "Rembg":
        return run_rembg(image)
    
    elif method == "SaliencyCrop":
        return run_saliency_crop(image, prompt)

    elif method == "SurfaceNormalization":
        return run_surface_normals(image)

    elif method == "GeometricTransformations":
        return run_geometric_transformations(image)

    elif method == "RandomCroppingResizing":
        return run_random_cropping_resizing(image)

    elif method == "ColorJittering":
        return run_color_jittering(image)

    elif method == "NoiseInjection":
        return run_noise_injection(image)

    elif method == "KernelBlurringSharpening":
        return run_kernel_blurring_sharpening(image)

    elif method == "RandomErasingCutout":
        return run_random_erasing_cutout(image)

    elif method == "ElasticDeformation":
        return run_elastic_deformation(image)

    elif method == "ShearMapping":
        return run_shear_mapping(image)
    
    elif method == "PerspectiveTransform":
        return run_perspective_transform(image)
    
    elif method == "MosaicAugmentation":
        return run_mosaic_augmentation(image)
    
    elif method == "Solarization":
        return run_solarization(image)
    
    elif method == "WeatherFogSimulation":
        return run_weather_fog_simulation(image)
    
    elif method == "GridMaskDropout":
        return run_gridmask_dropout(image)
    
    elif method == "StyleTransferFilter":
        return run_style_transfer_filter(image)
    
    elif method == "TestTimeAugmentation":
        return run_test_time_augmentation(image)

    elif method == "QueryAwareBoundingBox":
        return run_query_aware_bounding_box(image, prompt=prompt)

    elif method == "SemanticHazardIsolation":
        return run_semantic_hazard_isolation(image)

    elif method == "PerceptionMetricGrounding":
        return run_perception_metric_grounding(image)

    elif method == "SceneGraphGeneration":
        return run_scene_graph_generator(image)

    elif method == "Future Work : Smart Augmentation Selector":
        return run_adaptive_policy_selection(image, prompt=prompt)

    elif method == "none":
        return image

    else:
        return image