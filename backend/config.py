# config.py
# Central place for configuration variables

AVAILABLE_MODELS = [
    "qwen3-vl:4b",
    "llava-phi3",
    "mock"
]

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
}