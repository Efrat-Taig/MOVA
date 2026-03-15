#!/usr/bin/env python3
"""
MOVA experiment runner - runs multiple configs to find the best video.
Start with quick experiments (360p, 10 steps), then run full experiments.
"""

import subprocess
import sys
from pathlib import Path

# Import config from efrat_run
from efrat_run import (
    PROMPT,
    REF_PATH,
    OUTPUT_DIR,
    OUTPUT_BASE_NAME,
    RESOLUTIONS,
    NEGATIVE_PROMPT,
    FRAMES_BY_LENGTH,
    CKPT_PATH,
    ATTN_TYPE,
    OFFLOAD,
)

# =============================================================================
# EXPERIMENT PHASES - Edit to enable/disable phases
# =============================================================================

# Phase 1: Quick smoke test - low res, few steps, short video (verify it runs)
QUICK_EXPERIMENTS = [
    {"resolution": "360p", "steps": 10, "video_length": 4, "seed": 42},
    {"resolution": "360p", "steps": 10, "video_length": 4, "seed": 123},
    {"resolution": "360p", "steps": 10, "video_length": 4, "seed": 456},
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
]

# Phase 4: Full quality
FULL_EXPERIMENTS = [
    {"resolution": "720p", "steps": 30, "video_length": 6, "seed": 42},
]


def run_experiment(exp: dict) -> bool:
    """Run a single experiment. Returns True on success."""
    resolution = exp["resolution"]
    steps = exp["steps"]
    video_length = exp["video_length"]
    seed = exp.get("seed", 42)

    width, height = RESOLUTIONS[resolution]
    num_frames = FRAMES_BY_LENGTH[video_length]

    output_name = f"{OUTPUT_BASE_NAME}_{video_length}s_{width}x{height}_{steps}steps_seed{seed}.mp4"
    output_path = str(Path(OUTPUT_DIR) / output_name)

    cmd = [
        "torchrun", "--nproc_per_node=1", "scripts/inference_single.py",
        "--ckpt_path", CKPT_PATH,
        "--prompt", PROMPT,
        "--negative_prompt", NEGATIVE_PROMPT,
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
    print(f"Running: {resolution} | {steps} steps | {video_length}s | seed={seed}")
    print(f"Output: {output_path}")
    print("="*60)

    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    experiments = (
        QUICK_EXPERIMENTS
        + SEED_EXPERIMENTS
        + STEPS_EXPERIMENTS
        + FULL_EXPERIMENTS
    )

    print(f"\nMOVA Experiments: {len(experiments)} runs (fully automated)")
    print("Phases: Quick → Seed sweep → Steps → Full quality\n")

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
