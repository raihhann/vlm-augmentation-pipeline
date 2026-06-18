import os
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

print("⏳ Loading BOLT Framework Backbone (CLIP)...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()
print(f"✅ BOLT Engine ready on {device.upper()}\n")

def run_bolt_sampler(video_path, target_query, output_folder="method6_bolt_output", budget_k=5):
    """
    Method 6: BOLT (Inference-Time Inverse Transform Prioritization).
    Treats global query-alignment profile outputs as a continuous probability density curve.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    fps = cap.get(cv2.CAP_PROP_FPS)

    text_inputs = processor(text=[target_query, "scenery background"], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    frame_buffer, timestamps, raw_scores = [], [], []
    frame_idx = 0

    print("🔄 Pass 1: Profiling whole video timeline weight distributions...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % max(1, int(fps / 2)) != 0:
            continue

        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image_inputs = processor(images=pil_img, return_tensors="pt").to(device)
        with torch.no_grad():
            image_features = model.get_image_features(**image_inputs)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = torch.matmul(image_features, text_features.T) * model.logit_scale.exp()
            prob = logits.softmax(dim=-1).cpu().numpy()[0][0]

        frame_buffer.append(frame)
        timestamps.append(frame_idx / fps)
        raw_scores.append(prob)
    cap.release()

    print("🧮 Pass 2: Executing Inverse Transform Sampling on the probability density...")
    raw_scores = np.array(raw_scores)
    weights = (raw_scores - np.min(raw_scores)) / (np.max(raw_scores) - np.min(raw_scores) + 1e-8) + 0.05
    cdf = np.cumsum(weights)
    cdf /= cdf[-1]

    u_points = np.linspace(0.01, 0.99, budget_k)
    selected_indices = []
    for u in u_points:
        idx = np.searchsorted(cdf, u)
        if idx not in selected_indices and idx < len(frame_buffer):
            selected_indices.append(idx)

    for step, idx in enumerate(selected_indices, 1):
        ts = timestamps[idx]
        cv2.imwrite(os.path.join(output_folder, f"bolt_frame_{ts:.1f}s.jpg"), frame_buffer[idx])
        print(f"🔒 [SAVED BY PRIORITIZATION] Frame at Sec {ts:.1f}s | Profile Density: {raw_scores[idx]:.3f}")

if __name__ == "__main__":
    run_bolt_sampler(
        video_path="../../TestVideos/tractor3.mp4",
        target_query="a blue farm tractor working in a dirt field",
        budget_k=5
    )