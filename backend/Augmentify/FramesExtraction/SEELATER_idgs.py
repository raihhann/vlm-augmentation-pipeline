import os
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

print("⏳ Loading CLIP Encoder for Tier 2...")

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
).to(device)

processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)

print(f"✅ Ready on {device.upper()}\n")


def run_optimized_idgs(
    video_path,
    target_prompt,
    output_folder="optimized_idgs_frames",
    pixel_threshold=15.0,
    gate_threshold=0.03,
):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30

    last_gray = None
    last_embed = None
    saved_count = 0
    frame_idx = 0

    print("🧠 Encoding text prompt...")

    # Precompute text embeddings once
    text_inputs = processor(
        text=[target_prompt, "background"],
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
        text_features /= text_features.norm(
            dim=-1,
            keepdim=True
        )

    print("🎬 Processing video...\n")

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        frame_idx += 1

        # ==================================================
        # TIER 1: FAST PIXEL CHANGE FILTER
        # ==================================================

        small_gray = cv2.resize(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            (160, 120)
        )

        # First frame becomes baseline
        if last_gray is None:
            last_gray = small_gray

            cv2.imwrite(
                os.path.join(output_folder, "frame_0.jpg"),
                frame
            )

            print("📌 Saved baseline frame")
            continue

        pixel_delta = cv2.mean(
            cv2.absdiff(small_gray, last_gray)
        )[0]
        print(f"Frame {frame_idx}: Pixel Delta = {pixel_delta:.2f}")
        last_gray = small_gray

        # Skip nearly identical frames
        if pixel_delta < pixel_threshold:
            continue

        # ==================================================
        # TIER 2: CLIP SEMANTIC FILTER
        # ==================================================

        pil_img = Image.fromarray(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )

        inputs = processor(
            images=pil_img,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():

            image_features = model.get_image_features(**inputs)

            image_features /= image_features.norm(
                dim=-1,
                keepdim=True
            )

            logits = (
                torch.matmul(
                    image_features,
                    text_features.T
                )
                * model.logit_scale.exp()
            )

            probabilities = logits.softmax(dim=-1)

            # Probability that image matches target prompt
            p_current = probabilities.cpu().numpy()[0][0]

        # ==================================================
        # Compute semantic novelty
        # ==================================================

        if last_embed is None:
            delta_v = 1.0
        else:
            delta_v = (
                1.0
                - torch.dot(
                    image_features[0],
                    last_embed[0]
                ).item()
            )

        g_score = delta_v * p_current

        # ==================================================
        # Save only important frames
        # ==================================================

        if g_score >= gate_threshold:

            saved_count += 1

            last_embed = image_features.clone()

            timestamp = frame_idx / fps

            filename = os.path.join(
                output_folder,
                f"idgs_{timestamp:.1f}s.jpg"
            )

            cv2.imwrite(filename, frame)

            print(
                f"🔓 [GATE] "
                f"Time={timestamp:.1f}s | "
                f"PixelΔ={pixel_delta:.1f} | "
                f"SemanticΔ={delta_v:.3f} | "
                f"PromptProb={p_current:.3f} | "
                f"Score={g_score:.3f}"
            )

    cap.release()

    print("\n===================================")
    print("✅ Processing complete")
    print(f"📁 Saved frames: {saved_count}")
    print(f"📂 Output folder: {output_folder}")
    print("===================================")


# ==========================================================
# RUN EXAMPLE
# ==========================================================

if __name__ == "__main__":

    VIDEO_PATH = r"../../TestVideos/tractor2.mp4"

    run_optimized_idgs(
        video_path=VIDEO_PATH,
        target_prompt="tractor",
        output_folder="optimized_idgs_frames2",
        pixel_threshold=4,
        gate_threshold=0.03
    )