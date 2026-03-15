#!/usr/bin/env python3
"""
MOVA inference script - edit the config below and run with: python efrat_run.py
"""

import subprocess
import sys

# =============================================================================
# CONFIG - Edit these to control the inference
# =============================================================================

PROMPT = """A cheerful animated puppy pilot sitting in a small pink helicopter flying in a bright blue sky with soft white clouds. The character looks directly at the camera and happily talks to the viewer. She is smiling, excited, and expressive, like a children's cartoon character.

She tells the audience about the wonderful day she had — how happy she is and how she ate lots of delicious candies and sweets today. She laughs and says it was amazing and fun. Then she explains that because she ate so many sweets, she is now going to brush her teeth to keep them clean and healthy.

The helicopter gently hovers in the sky while she talks. The tone is joyful, playful, and energetic, like a kids' TV show. Bright colors, soft lighting, cute 3D animation style, friendly facial expressions, cinematic animated lighting, smooth character animation, high-quality children's animation style.

Camera: medium close-up, character speaking directly to camera inside the helicopter cockpit.
Mood: happy, playful, wholesome children's content.
Style: colorful 3D animated cartoon, soft clouds, bright daylight sky."""

REF_PATH = "inputs/sky_in_heli.jpg"
OUTPUT_DIR = "outputs"
OUTPUT_BASE_NAME = "puppy_pilot"  # Filename auto-built: {base}_{length}s_{WxH}_{steps}steps.mp4

# Video length: 4, 5, 6, 7, or 8 seconds (sets NUM_FRAMES automatically)
VIDEO_LENGTH = 6  # seconds

# Video settings (HEIGHT/WIDTH must be divisible by 16)
HEIGHT = 1280 # 640 848
WIDTH = 720 # 352 480
FPS = 24.0

# Frame counts for each length (num_frames - 1 must be divisible by 4)
FRAMES_BY_LENGTH = {4: 97, 5: 121, 6: 145, 7: 169, 8: 193}

# Inference settings
NUM_INFERENCE_STEPS = 20
CKPT_PATH = "checkpoints/MOVA-720p"
ATTN_TYPE = "torch_flash"
OFFLOAD = "cpu"  # "none", "cpu", or "group"

# =============================================================================
# Run
# =============================================================================

def main():
    num_frames = FRAMES_BY_LENGTH.get(VIDEO_LENGTH)
    if num_frames is None:
        print(f"Error: VIDEO_LENGTH must be 4, 5, 6, 7, or 8 (got {VIDEO_LENGTH})")
        sys.exit(1)

    output_name = f"{OUTPUT_BASE_NAME}_{VIDEO_LENGTH}s_{WIDTH}x{HEIGHT}_{NUM_INFERENCE_STEPS}steps.mp4"
    output_path = f"{OUTPUT_DIR}/{output_name}"
    print(f"Output: {output_path}")

    cmd = [
        "torchrun", "--nproc_per_node=1", "scripts/inference_single.py",
        "--ckpt_path", CKPT_PATH,
        "--prompt", PROMPT,
        "--ref_path", REF_PATH,
        "--output_path", output_path,
        "--height", str(HEIGHT),
        "--width", str(WIDTH),
        "--num_frames", str(num_frames),
        "--fps", str(FPS),
        "--num_inference_steps", str(NUM_INFERENCE_STEPS),
        "--attn_type", ATTN_TYPE,
        "--offload", OFFLOAD,
    ]
    print("Running:", " ".join(cmd[:6]), "...")
    sys.exit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
