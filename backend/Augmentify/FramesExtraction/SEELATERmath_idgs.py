import os
import glob
import cv2
import numpy as np

import os
import cv2
import numpy as np

def run_mathematical_idgs_v2(video_path, output_folder="math_idgs_v2_frames", motion_threshold=2.0, complexity_drift=5.0):
    """
    IDGS Version 2: Tracking-Aware Mathematical Sampler.
    Differentiates between a camera tracking a single object vs. a camera 
    discovering entirely new visual scenery.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    last_gray = None
    last_spatial_var = 0.0
    saved_count = 0
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small_gray = cv2.resize(gray, (320, 240))

        if last_gray is None:
            last_gray = small_gray
            last_spatial_var = cv2.Laplacian(small_gray, cv2.CV_64F).var()
            # Save first frame as baseline anchor
            ts = frame_idx / fps
            cv2.imwrite(os.path.join(output_folder, f"math_v2_{ts:.1f}s.jpg"), frame)
            saved_count += 1
            continue

        # 1. Measure raw temporal movement
        pixel_delta = cv2.mean(cv2.absdiff(small_gray, last_gray))[0]

        # 2. Measure current structural complexity
        current_spatial_var = cv2.Laplacian(small_gray, cv2.CV_64F).var()
        
        # 3. THE FIX: Compute the Structural Variance Drift (Novelty Math)
        # If the camera follows the tractor, the overall complexity profile 
        # stays almost identical (|current - last| is very low).
        variance_drift = abs(current_spatial_var - last_spatial_var)

        # GATING LOGIC:
        # We only open the gate if there is motion AND the complexity profile changes 
        # (meaning new objects/scenery are entering the lens, not just tracking the old ones).
        if pixel_delta > motion_threshold and variance_drift > complexity_drift:
            saved_count += 1
            last_spatial_var = current_spatial_var
            
            ts = frame_idx / fps
            cv2.imwrite(os.path.join(output_folder, f"math_v2_{ts:.1f}s.jpg"), frame)
            print(f"🔓 [GATE OPEN] Sec {ts:.1f}s | New Information Discovered! Drift: {variance_drift:.2f}")
            
            last_gray = small_gray
        else:
            # Maintain structural history to follow smooth pans without opening the gate
            if pixel_delta > motion_threshold:
                last_gray = small_gray

    cap.release()
    print(f"🎉 V2 Complete! Reduced tractor video clutter down to high-density instances.")

# --- RUN IT ---
if __name__ == "__main__":
    run_mathematical_idgs_v2(
        video_path="../../TestVideos/tractor2.mp4", # Adjust to filter out blank/empty spaces
    )