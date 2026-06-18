import os
import cv2
import numpy as np

def run_optical_flow_gating_sampler(video_path, output_folder="method3_optical_flow_output", motion_threshold=4.5):
    """
    Method 3: Sparse Optical Flow Tracking and Feature Velocity Gating.
    Tracks geometric keypoints using Lucas-Kanade optical flow to calculate
    the absolute magnitude of motion vectors between consecutive frames.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"🎬 Running Sparse Optical Flow on: {os.path.basename(video_path)}")
    print(f"📊 Total Frames: {total_frames} | Motion Threshold (tau_motion): {motion_threshold}")

    # Shi-Tomasi corner detection parameters (to find trackable features with eigenvalue constraints)
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    
    # Lucas-Kanade optical flow parameters minimizing residual neighborhood error
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    ret, old_frame = cap.read()
    if not ret:
        print("❌ Error: Video file empty or unreadable.")
        cap.release()
        return

    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    
    # Always save the absolute first frame as anchor reference node
    saved_count = 1
    frame_idx = 1
    cv2.imwrite(os.path.join(output_folder, "frame_0.0s.jpg"), old_frame)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # If we lose our keypoints, find a fresh batch to track from the preceding frame
        if p0 is None or len(p0) < 10:
            p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
            if p0 is None:
                old_gray = frame_gray.copy()
                continue

        # Math Crux: Calculate optical flow displacement vectors between consecutive frames
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        if p1 is not None and len(p1) > 0:
            # Filter and isolate valid corresponding tracking points
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            if len(good_new) > 0:
                # Calculate the Euclidean distance magnitude each tracking dot traveled
                # Vector math matching your thesis: sqrt((x2 - x1)^2 + (y2 - y1)^2)
                motion_vectors = np.sqrt(np.sum((good_new - good_old) ** 2, axis=1))
                mean_motion_magnitude = np.mean(motion_vectors)
            else:
                mean_motion_magnitude = 0.0
        else:
            mean_motion_magnitude = 0.0

        # GATING DECISION:
        # If the mean spatial motion vector magnitude spikes past our static threshold parameter,
        # the view has shifted significantly enough to warrant a keyframe token capture.
        if mean_motion_magnitude > motion_threshold:
            saved_count += 1
            ts = frame_idx / fps
            out_path = os.path.join(output_folder, f"frame_{ts:.1f}s.jpg")
            cv2.imwrite(out_path, frame)
            print(f"🔓 [GATE OPEN] Sec {ts:.1f}s | Mean Flow Magnitude: {mean_motion_magnitude:.3f}")
            
            # Re-extract clean features immediately from the newly saved node to refresh tracking
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
        else:
            # If the motion envelope is steady, pass coordinates forward sequentially 
            # to maintain continuous trajectory tracking across frames
            p0 = good_new.reshape(-1, 1, 2) if p1 is not None and len(good_new) > 0 else None

        old_gray = frame_gray.copy()

    cap.release()
    print(f"🎉 Done! Compressed {total_frames} frames down to {saved_count} optical flow tracked frames.\n")

if __name__ == "__main__":
    # Standard engineering execution path to evaluate on your tractor clip
    video_clip_path = "../../TestVideos/tractor3.mp4"
    
    # Run the sampler with your defined motion limit threshold boundary
    run_optical_flow_gating_sampler(
        video_path=video_clip_path,
        output_folder="method3_optical_flow_output",
        motion_threshold=4.5
    )