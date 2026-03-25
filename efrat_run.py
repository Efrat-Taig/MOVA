#!/usr/bin/env python3
"""
MOVA inference script - edit the config below and run with: python efrat_run.py

H100 / driver CUDA 12.x: install torch, torchvision, and torchaudio from the *same*
CUDA 12 wheel index so extensions agree (avoids libcudart.so.13 vs cu126 mismatch):

  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

Or run: bash scripts/install_pytorch_cuda126.sh
"""

import os
import subprocess
import sys

# =============================================================================
# CONFIG - Edit these to control the inference
# =============================================================================

PROMPT =  """
High-quality 3D animation in the style of Paw Patrol, vibrant colors, clean CGI, cinematic lighting, bright and cheerful atmosphere, smooth character animation, Nick Jr aesthetic, 4k render, detailed textures, expressive facial expressions.

Skye from Paw Patrol, the cute female cockapoo rescue puppy pilot, wearing her pink pilot goggles and pink vest, flying her small pink helicopter high in a bright blue sky with soft fluffy clouds. The helicopter gently hovers in the air while she sits in the cockpit.

Skye looks directly into the camera and talks to the viewer like a cheerful kids TV host. She speaks clearly in FIRST PERSON about her own day. She is happy, excited, and very expressive.

Dialogue (first person, speaking directly to camera):
"Hi everyone! I had such a happy day today!"
"I ate soooo many candies and sweets!"
"It was so yummy and so much fun!"
"But now I need to brush my teeth really well so they stay clean and healthy!"

She laughs, smiles, and talks energetically while sitting in the helicopter.

Camera: medium close-up inside the helicopter cockpit, Skye facing the camera.
Motion: helicopter gently hovering in the sky with soft movement.
Mood: joyful, playful, energetic children's show.
Environment: bright daylight sky, soft clouds.

Style: colorful 3D animated cartoon, Paw Patrol style animation.
"""

REF_PATH = "inputs/sky_in_heli.jpg"
OUTPUT_DIR = "outputs"
OUTPUT_BASE_NAME = "puppy_pilot"  # Filename auto-built: {base}_{length}s_{WxH}_{steps}steps.mp4

# Video length: 4, 5, 6, 7, or 8 seconds (sets NUM_FRAMES automatically)
VIDEO_LENGTH = 6  # seconds

# Resolution: "360p", "480p", or "720p" (WIDTH x HEIGHT, portrait; must be divisible by 16)
RESOLUTION = "720p"  # 360p=480x848, 480p=640x1120, 720p=720x1280
RESOLUTIONS = {"360p": (480, 848), "480p": (640, 1120), "720p": (720, 1280)}

# Negative prompt - what to avoid in the generation
NEGATIVE_PROMPT = (
    "blurry, overexposed, static, low quality, distorted, ugly, "
    "worst quality, JPEG artifacts, extra fingers, deformed"
)

# Video settings (overridden by RESOLUTION above)
HEIGHT = 1280
WIDTH = 720
FPS = 24.0

# Frame counts for each length (num_frames - 1 must be divisible by 4)
FRAMES_BY_LENGTH = {
    4: 97, 5: 121, 6: 145, 7: 169, 8: 193,
    10: 241, 15: 361, 20: 481,
}

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

    width, height = RESOLUTIONS.get(RESOLUTION, (WIDTH, HEIGHT))
    if RESOLUTION not in RESOLUTIONS:
        print(f"Warning: RESOLUTION '{RESOLUTION}' unknown, using {width}x{height}")

    output_name = f"{OUTPUT_BASE_NAME}_{VIDEO_LENGTH}s_{width}x{height}_{NUM_INFERENCE_STEPS}steps.mp4"
    output_path = f"{OUTPUT_DIR}/{output_name}"
    print(f"Output: {output_path}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Single-GPU / some cloud images: avoid NCCL P2P/IB edge cases (see MOVA H100 notes).
    env = os.environ.copy()
    env.setdefault("NCCL_P2P_DISABLE", "1")
    env.setdefault("NCCL_IB_DISABLE", "1")

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=1",
        "scripts/inference_single.py",
        "--ckpt_path", CKPT_PATH,
        "--prompt", PROMPT,
        "--negative_prompt", NEGATIVE_PROMPT,
        "--ref_path", REF_PATH,
        "--output_path", output_path,
        "--height", str(height),
        "--width", str(width),
        "--num_frames", str(num_frames),
        "--fps", str(FPS),
        "--num_inference_steps", str(NUM_INFERENCE_STEPS),
        "--attn_type", ATTN_TYPE,
        "--offload", OFFLOAD,
    ]
    print("Running:", " ".join(cmd[:8]), "...")
    sys.exit(subprocess.run(cmd, env=env, cwd=os.path.dirname(os.path.abspath(__file__))).returncode)


if __name__ == "__main__":
    main()
