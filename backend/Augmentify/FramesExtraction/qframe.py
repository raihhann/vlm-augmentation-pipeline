import os
import cv2
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

print("⏳ Loading Query-Aware VLM Alignment Engine (CLIP)...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()
print(f"✅ Q-Frame Multi-Modal Engine ready on {device.upper()}\n")

def run_qframe_sampler(video_path, target_query, output_folder="method5_qframe_output", confidence_threshold=0.85):
    """
    Method 5: Q-Frame (Query-Aware Semantic Token Gating).
    Extracts frames adaptively based on localized cross-modal alignment scores.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    fps = cap.get(cv2.CAP_PROP_FPS)

    text_inputs = processor(text=[target_query, "generic empty scenery background"], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    saved_count, frame_idx = 0, 0
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
            logits_per_image = torch.matmul(image_features, text_features.T) * model.logit_scale.exp()
            query_confidence = logits_per_image.softmax(dim=-1).cpu().numpy()[0][0]

        if query_confidence >= confidence_threshold:
            saved_count += 1
            ts = frame_idx / fps
            cv2.imwrite(os.path.join(output_folder, f"frame_{ts:.1f}s.jpg"), frame)
            print(f"🔓 [Q-FRAME GATE OPEN] Sec {ts:.1f}s | Query Match Confidence: {query_confidence:.3f}")

    cap.release()
    print(f"🎉 Done! Filtered query-conditioned semantic states down to {saved_count} frames.\n")

if __name__ == "__main__":
    run_qframe_sampler(
        video_path="../../TestVideos/tractor3.mp4",
        target_query="a blue farm tractor working in a dirt field"
    )