#!/usr/bin/env python3
"""
MOVA experiment runner - runs multiple configs to find the best video.
Start with quick experiments (360p, 15 steps), then run full experiments.

SETUP (run once per new terminal/session):
    conda activate mova
    pip install yunchang   # or: python run_experiments.py --setup

RUN:
    python run_experiments.py
"""

import subprocess
import sys
from pathlib import Path

# --setup: install yunchang and exit (run once per new env)
if "--setup" in sys.argv:
    print("Installing yunchang...")
    subprocess.run([sys.executable, "-m", "pip", "install", "yunchang"], check=True)
    print("Setup complete. Run: python run_experiments.py")
    sys.exit(0)

# Fail fast if yunchang is missing (inference_single.py needs it)
try:
    import yunchang.kernels  # noqa: F401
except ModuleNotFoundError:
    print(__doc__)
    print("Run: pip install yunchang  (or: python run_experiments.py --setup)")
    sys.exit(1)

# Import config from efrat_run
from efrat_run import (
    REF_PATH,
    OUTPUT_DIR,
    OUTPUT_BASE_NAME,
    RESOLUTIONS,
    FRAMES_BY_LENGTH,
    CKPT_PATH,
    ATTN_TYPE,
    OFFLOAD,
)

# Suffix added to all prompts
PROMPT_SUFFIX = " static camera composition maintained for the entire clip"

# Prompt type 1 (from efrat_run.py)
PROMPT_TYPE_1 = """
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
""" + PROMPT_SUFFIX

# Prompt type 2
PROMPT_TYPE_2 = """
High-quality 3D animation in the style of Paw Patrol, vibrant colors, clean CGI, cinematic lighting, bright cheerful atmosphere, smooth character animation, Nick Jr aesthetic, 4k render, detailed textures, expressive facial expressions.

Skye from Paw Patrol, the female cockapoo rescue puppy pilot wearing pink pilot goggles and a pink vest, sitting inside her small pink helicopter. The helicopter is hovering in a bright blue sky with soft fluffy clouds.

IMPORTANT: The entire video is ONE continuous shot. The camera angle does NOT change. No cuts, no scene transitions, no jump cuts, no camera switches.

The camera stays fixed in a stable medium close-up inside the helicopter cockpit. Skye remains centered in the frame and faces the camera the whole time. The helicopter gently hovers but the framing stays consistent.

Skye looks directly into the camera and talks to the viewer like a cheerful kids TV host. She speaks in FIRST PERSON about her own day. She is happy, excited and expressive.

Dialogue (first person, speaking to camera):
"Hi everyone! I had such a happy day today!"
"I ate soooo many candies and sweets!"
"It was so yummy and so much fun!"
"But now I need to brush my teeth really well so they stay clean and healthy!"

Her mouth movements match the speech. She smiles, laughs and talks energetically.

Motion: only subtle animation — slight helicopter hover, small head movement, blinking, natural mouth movement.
Camera: locked camera, stable framing, no zoom, no rotation, no angle change.
Shot: single continuous shot for the entire clip.
Mood: joyful, playful children's show.
Environment: bright daylight sky with soft clouds.
""" + PROMPT_SUFFIX

# Prompt type 3
PROMPT_TYPE_3 = """
Stable single-shot 3D animated scene of Skye from Paw Patrol inside her pink rescue helicopter cockpit. High-quality children's TV animation, Paw Patrol CGI style, vibrant colors, clean rendering, cinematic but simple lighting, smooth animation, detailed textures.

Scene setup:
Skye, the female cockapoo rescue puppy wearing pink pilot goggles and a pink vest, sits in the pilot seat of her pink helicopter. The helicopter is calmly hovering in a bright blue sky with soft clouds visible through the cockpit window.

Camera and framing:
Locked camera.
Single continuous shot.
Medium close-up framing from inside the cockpit.
Skye centered in frame and facing forward.
No camera movement, no cuts, no transitions, no angle change.

Action:
Skye speaks directly to the viewer like a cheerful children's show host. She talks in first person about her day. Her mouth moves naturally while speaking, with small head movements, blinking, and expressive facial animation.

Dialogue:
"Hi friends! Guess what? I had such a fun day today!"
"I ate lots and lots of yummy candy!"
"It was super sweet and really fun!"
"But now I need to brush my teeth really well!"

Motion:
Very gentle helicopter hover motion only.
Subtle character animation: blinking, small head movement, natural speaking mouth motion.

Mood:
Bright, friendly, playful children's show atmosphere.
""" + PROMPT_SUFFIX

PROMPTS = {1: PROMPT_TYPE_1, 2: PROMPT_TYPE_2, 3: PROMPT_TYPE_3}

# Negative prompt type 1 (from efrat_run.py)
NEGATIVE_PROMPT_TYPE_1 = (
    "blurry, overexposed, static, low quality, distorted, ugly, "
    "worst quality, JPEG artifacts, extra fingers, deformed"
)

# Negative prompt type 2
NEGATIVE_PROMPT_TYPE_2 = (
    "scene change, shot change, jump cut, camera switch, camera movement, "
    "zoom, pan, rotation, changing angle, changing perspective, "
    "multiple scenes, montage, cinematic cuts, dynamic camera, "
    "blurry, low quality, compression artifacts, distorted face, "
    "deformed body, extra limbs, extra fingers"
)

NEGATIVE_PROMPTS = {1: NEGATIVE_PROMPT_TYPE_1, 2: NEGATIVE_PROMPT_TYPE_2}

# =============================================================================
# EXPERIMENT PHASES - Edit to enable/disable phases
# =============================================================================

# Phase 1: Quick smoke test - low res, short video (15 steps min for decent motion)
QUICK_EXPERIMENTS = [
    {"resolution": "360p", "steps": 15, "video_length": 4, "seed": 42},
    {"resolution": "360p", "steps": 15, "video_length": 4, "seed": 0},
]

# Phase 2: Seed sweep at medium quality
SEED_EXPERIMENTS = [
    {"resolution": "480p", "steps": 20, "video_length": 6, "seed": s}
    for s in [42, 123, 456, 789, 999]
]

# Phase 3: Steps comparison
STEPS_EXPERIMENTS = [
    {"resolution": "480p", "steps": 10, "video_length": 6, "seed": 42},
    {"resolution": "480p", "steps": 20, "video_length": 6, "seed": 42},
    {"resolution": "480p", "steps": 30, "video_length": 6, "seed": 42},
    {"resolution": "480p", "steps": 40, "video_length": 6, "seed": 42},
    {"resolution": "480p", "steps": 50, "video_length": 6, "seed": 42},
]

# Phase 4: Full quality
FULL_EXPERIMENTS = [
    {"resolution": "360p", "steps": 50, "video_length": 10, "seed": 42},
    {"resolution": "360p", "steps": 50, "video_length": 15, "seed": 42},
    {"resolution": "480p", "steps": 50, "video_length": 20, "seed": 42},
    {"resolution": "720p", "steps": 50, "video_length": 20, "seed": 42},
    {"resolution": "720p", "steps": 50, "video_length": 20, "seed": 42},
]

# Phase 5: Prompt experiments - compare prompt types at 720p, 50 steps, 4s
PROMPT_EXPERIMENTS = [
    {"resolution": "720p", "steps": 50, "video_length": 8, "seed": 42, "prompt_type": 1},
    {"resolution": "720p", "steps": 50, "video_length": 8, "seed": 42, "prompt_type": 2},
    {"resolution": "720p", "steps": 50, "video_length": 8, "seed": 42, "prompt_type": 3},
    {"resolution": "720p", "steps": 50, "video_length": 8, "seed": 0, "prompt_type": 1},
    {"resolution": "720p", "steps": 50, "video_length": 8, "seed": 0, "prompt_type": 2},
    {"resolution": "720p", "steps": 50, "video_length": 8, "seed": 0, "prompt_type": 3},
]


def run_experiment(exp: dict) -> bool:
    """Run a single experiment. Returns True on success."""
    resolution = exp["resolution"]
    steps = exp["steps"]
    video_length = exp["video_length"]
    seed = exp.get("seed", 42)
    prompt_type = exp.get("prompt_type", 1)

    prompt = PROMPTS[prompt_type]
    neg_prompt_type = 1 if prompt_type == 1 else 2
    negative_prompt = NEGATIVE_PROMPTS[neg_prompt_type]

    width, height = RESOLUTIONS[resolution]
    num_frames = FRAMES_BY_LENGTH[video_length]

    output_name = f"{OUTPUT_BASE_NAME}_{video_length}s_{width}x{height}_{steps}steps_seed{seed}.mp4"
    if "prompt_type" in exp:
        output_name = f"{OUTPUT_BASE_NAME}_pt{prompt_type}_{video_length}s_{width}x{height}_{steps}steps_seed{seed}.mp4"
    output_path = str(Path(OUTPUT_DIR) / output_name)

    cmd = [
        "torchrun", "--nproc_per_node=1", "scripts/inference_single.py",
        "--ckpt_path", CKPT_PATH,
        "--prompt", prompt,
        "--negative_prompt", negative_prompt,
        "--ref_path", REF_PATH,
        "--output_path", output_path,
        "--height", str(height),
        "--width", str(width),
        "--num_frames", str(num_frames),
        "--fps", "24.0",
        "--num_inference_steps", str(steps),
        "--seed", str(seed),
        "--attn_type", ATTN_TYPE,
        "--offload", OFFLOAD,
    ]

    print(f"\n{'='*60}")
    prompt_info = f" | prompt_type={prompt_type}" if "prompt_type" in exp else ""
    print(f"Running: {resolution} | {steps} steps | {video_length}s | seed={seed}{prompt_info}")
    print(f"Output: {output_path}")
    print("="*60)

    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    experiments = FULL_EXPERIMENTS  # Phase 4 only

    # experiments = (
#     QUICK_EXPERIMENTS
#     + SEED_EXPERIMENTS
#     + STEPS_EXPERIMENTS
#     + FULL_EXPERIMENTS
# )

    phase_name = "Phase 4: Full quality"
    print(f"\nMOVA Experiments: {len(experiments)} runs ({phase_name})")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    for i, exp in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] ", end="")
        success = run_experiment(exp)
        if not success:
            print(f"\nExperiment failed. Stopping.")
            sys.exit(1)

    print(f"\n{'='*60}")
    print(f"All {len(experiments)} experiments completed successfully!")
    print(f"Outputs in: {OUTPUT_DIR}/")
    print("="*60)


if __name__ == "__main__":
    main()
