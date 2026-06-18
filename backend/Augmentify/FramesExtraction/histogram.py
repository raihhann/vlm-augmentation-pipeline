import os
import cv2
import numpy as np


def run_histogram_dissimilarity_sampler(
        video_path,
        output_folder="histogram_output",
        threshold=0.35):
    """
    Method 2:
    Histogram Dissimilarity and Color-Space Intersection

    Pipeline:
        Video
          ↓
        HSV Histogram
          ↓
        Compare with Last Saved Keyframe
          ↓
        Bhattacharyya Distance
          ↓
        Threshold Gate
          ↓
        Save Keyframe

    A frame is retained only when its histogram distance
    exceeds the predefined threshold relative to the
    previously retained keyframe.
    """

    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("\n" + "=" * 60)
    print("METHOD 2: HISTOGRAM DISSIMILARITY")
    print("=" * 60)
    print(f"🎬 Video: {os.path.basename(video_path)}")
    print(f"📊 Total Frames: {total_frames}")
    print(f"🎯 Threshold: {threshold}")

    anchor_hist = None
    saved_count = 0
    frame_idx = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_idx += 1

        # --------------------------------------------------
        # Convert to HSV
        # --------------------------------------------------

        hsv = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2HSV
        )

        # --------------------------------------------------
        # 3D Histogram
        # H = 8 bins
        # S = 8 bins
        # V = 8 bins
        # Total = 512 features
        # --------------------------------------------------

        hist = cv2.calcHist(
            [hsv],
            [0, 1, 2],
            None,
            [8, 8, 8],
            [0, 180, 0, 256, 0, 256]
        )

        # Normalize histogram
        cv2.normalize(
            hist,
            hist,
            alpha=0,
            beta=1,
            norm_type=cv2.NORM_MINMAX
        )

        # --------------------------------------------------
        # First frame becomes anchor
        # --------------------------------------------------

        if anchor_hist is None:

            timestamp = frame_idx / fps

            filename = os.path.join(
                output_folder,
                f"frame_{timestamp:.2f}s.jpg"
            )

            cv2.imwrite(filename, frame)

            anchor_hist = hist
            saved_count += 1

            print(
                f"📌 Anchor Frame "
                f"@ {timestamp:.2f}s"
            )

            continue

        # --------------------------------------------------
        # Histogram Distance
        # --------------------------------------------------

        distance = cv2.compareHist(
            anchor_hist,
            hist,
            cv2.HISTCMP_BHATTACHARYYA
        )

        # --------------------------------------------------
        # Gate Decision
        # --------------------------------------------------

        if distance > threshold:

            timestamp = frame_idx / fps

            filename = os.path.join(
                output_folder,
                f"frame_{timestamp:.2f}s.jpg"
            )

            cv2.imwrite(filename, frame)

            print(
                f"🔓 Keyframe "
                f"@ {timestamp:.2f}s "
                f"| Distance = {distance:.3f}"
            )

            saved_count += 1

            # Update anchor
            anchor_hist = hist

    cap.release()

    compression_ratio = total_frames / max(saved_count, 1)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Original Frames : {total_frames}")
    print(f"Keyframes Saved : {saved_count}")
    print(f"Compression     : {compression_ratio:.1f}x")
    print("=" * 60 + "\n")


if __name__ == "__main__":

    run_histogram_dissimilarity_sampler(
        "../../TestVideos/room.mp4",
        threshold=0.35
    )