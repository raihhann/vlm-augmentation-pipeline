import os
import subprocess


def run_intra_frame_sampler(
    video_path,
    output_folder="intra_frame_sampler"
):
    """
    Traditional Codec-Level Baseline (I-Frame Extraction)

    Extracts ONLY native codec I-frames from a compressed video
    stream using FFmpeg. No semantic, motion, or query analysis
    is performed.

    Parameters
    ----------
    video_path : str
        Path to input video.

    output_folder : str
        Folder where extracted I-frames are stored.
    """

    # --------------------------------------------------
    # Create output folder
    # --------------------------------------------------

    os.makedirs(output_folder, exist_ok=True)

    print("\n" + "=" * 60)
    print("METHOD 1: TRADITIONAL I-FRAME BASELINE")
    print("=" * 60)
    print(f"Input Video : {video_path}")
    print(f"Output Path : {output_folder}")

    # --------------------------------------------------
    # Output naming pattern
    # --------------------------------------------------

    output_pattern = os.path.join(
        output_folder,
        "iframe_%04d.jpg"
    )

    # --------------------------------------------------
    # FFmpeg command
    # --------------------------------------------------

    command = [
        "ffmpeg",
        "-i", video_path,

        # Keep ONLY codec I-frames
        "-vf", "select='eq(pict_type,I)'",

        # Avoid duplicate frame insertion
        "-vsync", "vfr",

        # High-quality JPEG
        "-q:v", "2",

        output_pattern,

        # Overwrite existing images
        "-y"
    ]

    try:

        subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        extracted_frames = len([
            f for f in os.listdir(output_folder)
            if f.lower().endswith(".jpg")
        ])

        print(
            f"\n✓ Successfully extracted "
            f"{extracted_frames} native I-frames"
        )

    except FileNotFoundError:

        print(
            "\n❌ FFmpeg not found.\n"
            "Please install FFmpeg and ensure it is "
            "available in your system PATH."
        )

    except subprocess.CalledProcessError as e:

        print("\n❌ FFmpeg execution failed.")

        try:
            print(
                e.stderr.decode(
                    errors="ignore"
                )
            )
        except Exception:
            print(e)

    print("=" * 60 + "\n")


if __name__ == "__main__":

    VIDEO_PATH = "../../TestVideos/tractor.mp4"

    run_intra_frame_sampler(
        video_path=VIDEO_PATH,
        output_folder="intra_frame_sampler"
    )