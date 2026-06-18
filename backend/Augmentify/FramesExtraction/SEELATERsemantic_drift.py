import os
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

print("⏳ Loading Ultra-Lightweight Semantic Encoder...")
device = "cuda" if torch.cuda.is_available() else "cpu"
# We use a tiny, lightning-fast vision transformer optimized for edge feature extraction
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = AutoModel.from_pretrained("google/vit-base-patch16-224-in21k").to(device)
model.eval()
print(f"✅ Semantic Encoder ready on {device.upper()}\n")

def run_semantic_drift_sampler(video_path, output_folder="model_drift_frames3", drift_threshold=0.12):
    """
    Model-Based Semantic Trajectory Sampler.
    Tracks the cosine drift of deep feature embeddings. Blocks redundant frames 
    when a camera is merely panning or tracking the same semantic subject.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    last_saved_embed = None
    saved_count = 0
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Evaluate at a baseline to protect edge hardware (e.g., check 2 frames per second)
        if frame_idx % max(1, int(fps / 2)) != 0:
            continue

        # Convert to PIL for the transformer processor
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = processor(images=pil_img, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            # Grab the pooler output (a clean 768-dimensional semantic fingerprint of the frame)
            current_embed = outputs.pooler_output
            # Normalize the vector to unit length
            current_embed /= current_embed.norm(dim=-1, keepdim=True)

        if last_saved_embed is None:
            # Anchor frame: Always capture the first instance
            last_saved_embed = current_embed.clone()
            ts = frame_idx / fps
            cv2.imwrite(os.path.join(output_folder, f"semantic_anchor_{ts:.1f}s.jpg"), frame)
            saved_count += 1
            continue

        # Calculate the Semantic Distance (1.0 - Cosine Similarity)
        # This tells us if the MEANING of the scene has drifted, completely ignoring 
        # simple camera tracking shifts or uniform panning motions.
        cosine_similarity = torch.dot(current_embed[0], last_saved_embed[0]).item()
        semantic_drift = 1.0 - cosine_similarity

        # GATING DECISION:
        # If the tractor is just moving through the same field scenery, the semantic drift 
        # stays below the threshold, completely locking the gate and saving VLM tokens.
        if semantic_drift > drift_threshold:
            saved_count += 1
            ts = frame_idx / fps
            cv2.imwrite(os.path.join(output_folder, f"semantic_drift_{ts:.1f}s.jpg"), frame)
            print(f"🔓 [GATE OPEN] Sec {ts:.1f}s | Semantic Drift Spiked: {semantic_drift:.3f} -> New Context!")
            
            # Reset our anchor to this new structural landmark
            last_saved_embed = current_embed.clone()

    cap.release()
    print(f"\n🎉 Done! The model successfully locked out the tracking clutter.")

if __name__ == "__main__":
    run_semantic_drift_sampler("../../TestVideos/tractor.mp4", drift_threshold=0.10)