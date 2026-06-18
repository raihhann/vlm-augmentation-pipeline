# Augmentify/extract/video_extractor.py
# This module acts as the unified factory for video sampling methodologies

import os
from PIL import Image
from typing import List

# Import our modular, traceable video processing backends
from Augmentify.FramesExtraction.I_frames import run_intra_frame_sampler
from Augmentify.FramesExtraction.maxinfo import run_maxinfo_sampler
from Augmentify.FramesExtraction.optical_flow import run_optical_flow_gating_sampler
from Augmentify.FramesExtraction.trajectory_space import run_trajectory_sampler
from Augmentify.FramesExtraction.qframe import run_qframe_sampler
from Augmentify.FramesExtraction.bolt import run_bolt_sampler

def extract_video_keyframes(video_path: str, method: str, query: str, output_dir: str) -> List[str]:
    """
    Factory interface mirroring apply_augmentation.
    Accepts video tracking pathways, executes the chosen keyframe gate strategy (1-6),
    and returns a list of absolute file-system strings pointing to the extracted images.
    """
    print(f"🎬 Video Keyframe Extraction Method: {method} | Prompt: {query}")
    
    # Ensure a dedicated job directory exists to isolate user timeline states
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert numerical strings from HTML form selector tokens into explicit method routines
    if method == "1":
        # Method 1: Traditional Codec Baseline (Intra-coded / I-Frame extraction simulator)
        run_intra_frame_sampler(video_path, output_folder=output_dir)
        
    elif method == "2":
        # Method 2: Unsupervised Spatial-Diversity Feature Selection (MaxInfo Framework)
        # Slices 8 highly distinct topological latent nodes
        run_maxinfo_sampler(video_path, output_folder=output_dir, target_k=8)
        
    elif method == "3":
        # Method 3: Sparse Optical Flow Tracking and Feature Velocity Gating
        # Employs a fixed threshold envelope parameter at 4.5
        run_optical_flow_gating_sampler(video_path, output_folder=output_dir, motion_threshold=4.5)
        
    elif method == "4":
        # Method 4: Latent Trajectory Space Partitioning and Deep Feature Distance
        run_trajectory_sampler(video_path, output_folder=output_dir, distance_threshold=0.35)
        
    elif method == "5":
        # Method 5: Query-Aware Semantic Token Gating (Q-Frame Framework)
        run_qframe_sampler(video_path, target_query=query, output_folder=output_dir, confidence_threshold=0.85)
        
    elif method == "6":
        # Method 6: Inference-Time Query-Frame Prioritization (BOLT Framework)
        run_bolt_sampler(video_path, target_query=query, output_folder=output_dir, budget_k=5)
        
    else:
        # Fallback security routing: execute basic method 1
        run_intra_frame_sampler(video_path, output_folder=output_dir)

    # Collect and return all extracted image paths sorted sequentially by timeline position
    extracted_files = [
        os.path.join(output_dir, f) for f in os.listdir(output_dir) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    return sorted(extracted_files)