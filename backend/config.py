"""Define models, image augmentations, and video extraction strategies exposed by the API.

The constants in this module are shared by the FastAPI views and frontend
templates to keep selectable processing options in one place.
"""

#: List of supported Vision-Language Models available for inference, could be extended to more just by adding their name here matching with ollama.
AVAILABLE_MODELS = [
    "qwen3-vl:4b",
    "llava-phi3",
    "mock"
]

#: Comprehensive pool of image augmentation and transformation techniques
AVAILABLE_AUGMENTATIONS = [
    "FastSAM",
    "MobileSAM",
    "MIDAS",
    "Yolo_World_Annotate",
    "ContextAwareZoom",
    "RT-DETR",
    "DepthAnything",
    "GeometricSegmentation",
    "GammaCorrection",
    "CLAHE",
    "RetinexSSR",
    "PoseEstimation",
    "Rembg",
    "SaliencyCrop",
    "SurfaceNormalization",
    "ZoeDepth",
    "GeometricTransformations",   
    "RandomCroppingResizing",     
    "ColorJittering",             
    "NoiseInjection",             
    "KernelBlurringSharpening",   
    "RandomErasingCutout",        
    "ElasticDeformation",
    "ShearMapping",
    "PerspectiveTransform",
    "MosaicAugmentation",
    "Solarization",
    "WeatherFogSimulation",
    "GridMaskDropout",
    "StyleTransferFilter",
    "TestTimeAugmentation",  
    "QueryAwareBoundingBox",
    "SemanticHazardIsolation",
    "PerceptionMetricGrounding",
    "SceneGraphGeneration",
    #FutureWork: Placeholder for future adaptive and query-aware extraction methods
    "Future Work : Smart Augmentation Selector"
]

# Keyframe extraction strategies mapped to human-readable UI labels
AVAILABLE_EXTRACTIONS = {
    # Category 1: Structural & Pixel-Level
    "iframes": "Codec Native I-Frames",
    "scenedetect": "PySceneDetect (Adaptive Cut)",
    "peak_signal": "Signal PeakUtils",
    "ssim": "SSIM Structural Similarity",
    "histogram": "HSV Histogram Dissimilarity",
    "laplacian": "Laplacian Sharpness Variance",
    "curvature": "Curvature Points (Ciocca-Schettini)",
    "bitwise_xor": "Bitwise-XOR Dissimilarity",
    "covariance": "Covariance Matrix Min-Eigenvalue",
    
    # Category 2: Motion & Kinematics
    "sparse_flow": "Sparse Optical Flow (Lucas-Kanade)",
    "dense_flow": "Dense Optical Flow Energy (Farneback)",
    
    # Category 3: Unsupervised Deep Latents
    "katna": "Katna Multi-Stage (LUV + K-Means)",
    "kmeans": "Latent Space K-Means Medoids",
    "maxinfo": "MaxInfo (SVD + MaxVol Geometric)",
    "tsdpc": "TSDPC Density Peaks Clustering",
    "delaunay": "Dynamic Delaunay Graph Clustering",
    
    # Category 4: Multimodal Query VLMs
    "keyvideollm": "KeyVideoLLM (CLIP Query Top-K)",
    "bolt": "BOLT (CDF Inverse Transform Sampling)",
    "qa_iframes": "QA-IF (Query-Aware Codec I-Frames + CLIPSeg)",
    "smart_sampling": "Smart Sampling (Prompt Grounded YOLO-World)",

    #FutureWork: Placeholder for future adaptive and query-aware extraction methods
    "super_smart_sampling": "Super Smart Sampling (Intelligent Query-Grounded Scoring)"


}


