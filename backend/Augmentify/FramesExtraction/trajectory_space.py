import os
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

print("⏳ Loading Trajectory Encoder Backbone (CLIP)...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()
print(f"✅ Trajectory Analyzer ready on {device.upper()}\n")

def run_trajectory_sampler(video_path, output_folder="method4_trajectory_output", distance_threshold=0.35):
    """
    Method 4: Latent Trajectory Space Partitioning.
    Triggers selection when the cumulative displacement vector in latent space branches beyond threshold boundaries.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Extract and save initial node vector reference point
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return

    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor(images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        anchor_feat = model.get_image_features(**inputs)
        anchor_feat /= anchor_feat.norm(dim=-1, keepdim=True)
        anchor_feat = anchor_feat.cpu().numpy()[0]

    cv2.imwrite(os.path.join(output_folder, "trajectory_frame_0.0s.jpg"), frame)
    saved_count = 1
    frame_idx = 1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % max(1, int(fps / 2)) != 0:
            continue

        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = processor(images=pil_img, return_tensors="pt").to(device)
        
        with torch.no_grad():
            current_feat = model.get_image_features(**inputs)
            current_feat /= current_feat.norm(dim=-1, keepdim=True)
            current_feat = current_feat.cpu().numpy()[0]

        # Calculate cosine distance boundary across the latent trajectory path
        latent_distance = 1.0 - np.dot(anchor_feat, current_feat)

        if latent_distance > distance_threshold:
            saved_count += 1
            ts = frame_idx / fps
            cv2.imwrite(os.path.join(output_folder, f"trajectory_frame_{ts:.1f}s.jpg"), frame)
            print(f"🔓 [PATH BOUNDARY BREACHED] Sec {ts:.1f}s | Latent Drift Distance: {latent_distance:.3f}")
            anchor_feat = current_feat # Update local anchor node tracking coordinates

    cap.release()
    print(f"🎉 Done! Captured {saved_count} frames over the trajectory timeline.\n")

if __name__ == "__main__":
    run_trajectory_sampler("../../TestVideos/tractor3.mp4")