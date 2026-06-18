import os
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

print("⏳ Loading MaxInfo Embedding Backbone (CLIP ViT-B/32)...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()
print(f"✅ MaxInfo Encoder ready on {device.upper()}\n")

def native_rect_maxvol(A, max_steps=100, tol=1.05):
    """
    Lightweight implementation of the Rectangular Maximum Volume (MaxVol) algorithm.
    Finds a subset of rows that maximizes the submatrix volume det(A_sub * A_sub^T).
    """
    n, idx_dim = A.shape
    if n <= idx_dim:
        return np.arange(n)
    
    # Gaussian elimination with partial pivoting to find initial full-rank rows
    B = A.copy()
    pivots = []
    remaining_rows = list(range(n))
    
    for i in range(idx_dim):
        max_row_idx = np.argmax(np.abs(B[remaining_rows, i]))
        pivot_row = remaining_rows[max_row_idx]
        pivots.append(pivot_row)
        remaining_rows.remove(pivot_row)
        
        # Eliminate column components
        pivot_val = B[pivot_row, i]
        if np.abs(pivot_val) > 1e-10:
            for r in remaining_rows:
                factor = B[r, i] / pivot_val
                B[r, i:] -= factor * B[pivot_row, i:]

    pivots = np.array(pivots)
    
    # Iterative volume maximization expansion steps
    for step in range(max_steps):
        A_sub = A[pivots]
        # Compute coefficients: C = A * inv(A_sub)
        try:
            C = np.dot(A, np.linalg.pinv(A_sub))
        except np.linalg.LinAlgError:
            break
            
        # Find index of the absolute maximum element in the coefficient matrix
        max_idx = np.unravel_index(np.argmax(np.abs(C), axis=None), C.shape)
        row_idx, col_idx = max_idx
        
        # Check if swapping column/row breaches volume tolerance boundary
        if np.abs(C[row_idx, col_idx]) <= tol:
            break
            
        # Swap row indexes to expand volume parameters
        pivots[col_idx] = row_idx
        
    return sorted(pivots)

def run_maxinfo_sampler(video_path, output_folder="method2_maxinfo_output", target_k=8):
    """
    Method 2: MaxInfo Framework (SVD + Rectangular MaxVol Submatrix Search).
    Extracts high-diversity visual landmarks completely unsupervised.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"🎬 Running Method 2 (MaxInfo) on: {os.path.basename(video_path)}")
    print(f"📊 Target Subspace Budget: Up to {target_k} Diverse Frames")

    frame_buffer = []
    timestamps = []
    embeddings = []
    frame_idx = 0

    # Pass 1: Gather uniform candidate pool (Baseline: Extract 1 frame per second)
    print("🔄 Step 1: Scanning video to build dense candidate footprint...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if frame_idx % max(1, int(fps)) != 0:
            continue

        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = processor(images=pil_img, return_tensors="pt").to(device)

        with torch.no_grad():
            # Extract image features and preserve standard normalized CLS representation
            img_feat = model.get_image_features(**inputs)
            img_feat /= img_feat.norm(dim=-1, keepdim=True)
            img_feat = img_feat.cpu().numpy()[0]

        frame_buffer.append(frame)
        timestamps.append(frame_idx / fps)
        embeddings.append(img_feat)

    cap.release()

    if not embeddings:
        print("❌ Error: No candidate frames extracted.")
        return

    # Construct complete video feature global matrix Q
    Q = np.array(embeddings) # shape: (n_frames, 512)
    n_samples = Q.shape[0]
    
    print(f"🧮 Step 2: Running Truncated SVD over Global Matrix shape {Q.shape}...")
    # Perform Singular Value Decomposition
    U, S_vals, Vt = np.linalg.svd(Q, full_matrices=False)
    
    # Isolate principal visual directions down to target frame dimensions budget
    s_dim = min(target_k, n_samples)
    Q_s = U[:, :s_dim] # shape: (n_frames, s_dim)

    print("📐 Step 3: Executing Rectangular MaxVol index search...")
    # MaxVol optimization to select the rows that expand differential entropy limits
    selected_indices = native_rect_maxvol(Q_s, tol=1.05)
    
    # Cap selection exactly to match target token limit protocols if exceeded
    if len(selected_indices) > target_k:
        selected_indices = selected_indices[:target_k]

    saved_count = 0
    for idx in selected_indices:
        saved_count += 1
        ts = timestamps[idx]
        frame_to_save = frame_buffer[idx]
        out_path = os.path.join(output_folder, f"maxinfo_frame_{ts:.1f}s.jpg")
        cv2.imwrite(out_path, frame_to_save)
        print(f"🔒 [SAVED BY MAXVOL] Selected Index: {idx} | Timeline Spot: {ts:.1f}s")

    print(f"🎉 Done! Extracted {saved_count} maximally informative structural keyframes.\n")

if __name__ == "__main__":
    run_maxinfo_sampler(
        video_path="../../TestVideos/tractor3.mp4", 
        output_folder="method2_maxinfo_output",
        target_k=5
    )