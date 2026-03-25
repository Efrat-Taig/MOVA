#!/usr/bin/env python3
"""
Run every benchmark under inputs/BM_v1/benchmark_v1 and write one MP4 per folder
into v1_10_sec_W640_H349/, using each config.json (prompts, dimensions, duration, etc.).

Veo subfolders only override prompts; reference image and timing are taken from
the sibling scene folder with the same name under benchmark_v1/.

Requires: conda env with MOVA + yunchang (same as run_experiments.py).

Usage:
    python run_benchmark_v1.py
    python run_benchmark_v1.py --dry-run
    python run_benchmark_v1.py --steps 50
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Repo root (directory containing this script)
REPO_ROOT = Path(__file__).resolve().parent
BENCHMARK_ROOT = REPO_ROOT / "inputs" / "BM_v1" / "benchmark_v1"
OUTPUT_ROOT = REPO_ROOT / "v1_10_sec_W640_H349"

DEFAULT_NEGATIVE = (
    "blurry, overexposed, static, low quality, distorted, ugly, "
    "worst quality, JPEG artifacts, extra fingers, deformed"
)


def snap_multiple_of_16(x: int) -> int:
    x = int(x)
    return max(16, (x // 16) * 16)


def discover_config_paths(root: Path) -> list[Path]:
    return sorted(root.rglob("config.json"))


def is_under_veo(config_path: Path, root: Path) -> bool:
    try:
        rel = config_path.relative_to(root)
    except ValueError:
        return False
    return len(rel.parts) > 1 and rel.parts[0] == "veo"


def merge_veo_with_parent(config_path: Path, root: Path, raw: dict) -> dict:
    """Veo configs only carry prompt variants; inherit the rest from the main scene."""
    if not is_under_veo(config_path, root):
        return raw
    scene_name = config_path.parent.name
    parent_cfg = root / scene_name / "config.json"
    if not parent_cfg.exists():
        print(f"Warning: veo config without parent {parent_cfg}, using veo JSON only")
        return raw
    base = json.loads(parent_cfg.read_text(encoding="utf-8"))
    base.update(raw)
    return base


def resolve_ref_path(config_path: Path, root: Path, cfg: dict) -> Path | None:
    folder = config_path.parent
    images = cfg.get("images") or []
    for entry in images:
        p = entry.get("path")
        if p:
            candidate = (folder / p).resolve()
            if candidate.is_file():
                return candidate
    for name in ("start_frame.png", "start_frame.jpg", "ref.jpg", "ref.png"):
        candidate = (folder / name).resolve()
        if candidate.is_file():
            return candidate
    if is_under_veo(config_path, root):
        scene_name = config_path.parent.name
        for name in ("start_frame.png", "start_frame.jpg"):
            candidate = (root / scene_name / name).resolve()
            if candidate.is_file():
                return candidate
    return None


def build_prompt(cfg: dict) -> str:
    parts: list[str] = []
    gp = cfg.get("global_prompt")
    if gp and str(gp).strip():
        parts.append(str(gp).strip())
    for sp in cfg.get("specific_prompts") or []:
        if sp and str(sp).strip():
            parts.append(str(sp).strip())
    if not parts:
        raise ValueError("No global_prompt or specific_prompts in config")
    return "\n\n".join(parts)


def num_frames_for_duration(duration: float, fps: float, fallback_map: dict) -> int:
    d = int(round(duration))
    if d in fallback_map:
        return fallback_map[d]
    # Match efrat_run.FRAMES_BY_LENGTH pattern: duration * fps + 1 at 24 fps
    n = int(round(duration * fps)) + 1
    if (n - 1) % 4 != 0:
        n = ((n - 1) // 4) * 4 + 1
    return max(5, n)


def run_one(
    cfg: dict,
    config_path: Path,
    ref_path: Path,
    out_mp4: Path,
    ckpt_path: str,
    attn_type: str,
    offload: str,
    num_inference_steps: int,
    frames_by_length: dict,
    env: dict,
) -> bool:
    fps = float(cfg.get("fps") or 24.0)
    duration = float(cfg.get("duration") or 10)
    seed = int(cfg.get("seed") if cfg.get("seed") is not None else 42)
    neg = (cfg.get("negative_prompt") or "").strip() or DEFAULT_NEGATIVE
    dim = cfg.get("dimensions") or {}
    width = snap_multiple_of_16(dim.get("width") or 1280)
    height = snap_multiple_of_16(dim.get("height") or 720)

    num_frames = num_frames_for_duration(duration, fps, frames_by_length)
    prompt = build_prompt(cfg)

    cmd = [
        "torchrun",
        "--nproc_per_node=1",
        str(REPO_ROOT / "scripts" / "inference_single.py"),
        "--ckpt_path",
        ckpt_path,
        "--prompt",
        prompt,
        "--negative_prompt",
        neg,
        "--ref_path",
        str(ref_path),
        "--output_path",
        str(out_mp4),
        "--height",
        str(height),
        "--width",
        str(width),
        "--num_frames",
        str(num_frames),
        "--fps",
        str(fps),
        "--num_inference_steps",
        str(num_inference_steps),
        "--seed",
        str(seed),
        "--attn_type",
        attn_type,
        "--offload",
        offload,
    ]

    print(f"\n{'=' * 60}")
    print(f"Config: {config_path.relative_to(REPO_ROOT)}")
    print(f"Output: {out_mp4.relative_to(REPO_ROOT)}")
    print(f"{width}x{height} | {num_frames} frames | {fps} fps | seed={seed} | steps={num_inference_steps}")
    print("=" * 60)

    result = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env)
    return result.returncode == 0


def _preflight_cuda() -> None:
    try:
        import torch
    except ImportError:
        print(
            "PyTorch is not installed in this Python environment. "
            "Activate your MOVA conda env (e.g. `conda activate mova`) and install dependencies.",
            file=sys.stderr,
        )
        sys.exit(1)
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        cuda_build = getattr(torch.version, "cuda", None)
        mismatch = ""
        if cuda_build is not None:
            major = int(str(cuda_build).split(".")[0])
            if major >= 13:
                mismatch = (
                    "  • If `nvidia-smi` works but PyTorch still shows no GPU, the driver likely only supports "
                    "CUDA 12.x while this PyTorch build targets CUDA 13.x. Reinstall matching wheels:\n"
                    "      bash scripts/repair_cuda_pytorch_driver_mismatch.sh\n"
                )
        print(
            "CUDA GPU not visible to PyTorch — inference cannot run.\n"
            "  • Run `nvidia-smi`. If it errors, install NVIDIA kernel modules for your *current* kernel, then reboot.\n"
            "  • With sudo: `sudo apt-get update && sudo apt-get install -y linux-modules-nvidia-570-server-open-$(uname -r)` then reboot.\n"
            "  • Without VM sudo: attach `scripts/gcp_startup_nvidia_kernel_modules.sh` as a GCE startup script (runs as root), "
            "then reset the instance — see comments at the top of that file.\n"
            "  • Use the conda env with CUDA PyTorch: `python -c \"import torch; print(torch.version.cuda)\"` should not be None.\n"
            f"{mismatch}",
            file=sys.stderr,
        )
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BM_v1 benchmarks into v1_10_sec_W640_H349/")
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=BENCHMARK_ROOT,
        help="Root folder containing benchmark scene directories",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Output directory (mirrors relative paths under benchmark root)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Inference steps (default: NUM_INFERENCE_STEPS from efrat_run)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned runs only")
    args = parser.parse_args()

    if not args.dry_run:
        _preflight_cuda()

    benchmark_root = args.benchmark_root.resolve()
    output_root = args.output_root.resolve()

    if not benchmark_root.is_dir():
        print(f"Benchmark root not found: {benchmark_root}", file=sys.stderr)
        sys.exit(1)

    try:
        from efrat_run import (
            ATTN_TYPE,
            CKPT_PATH,
            FRAMES_BY_LENGTH,
            NUM_INFERENCE_STEPS,
            OFFLOAD,
        )
    except ImportError as e:
        print("Import efrat_run failed:", e, file=sys.stderr)
        sys.exit(1)

    steps = args.steps if args.steps is not None else NUM_INFERENCE_STEPS
    ckpt_path = CKPT_PATH
    attn_type = ATTN_TYPE
    offload = OFFLOAD

    env = os.environ.copy()
    env.setdefault("NCCL_P2P_DISABLE", "1")
    env.setdefault("NCCL_IB_DISABLE", "1")

    configs = discover_config_paths(benchmark_root)
    if not configs:
        print(f"No config.json under {benchmark_root}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(configs)} benchmark config(s)")
    output_root.mkdir(parents=True, exist_ok=True)

    for i, config_path in enumerate(configs, 1):
        rel_parent = config_path.parent.relative_to(benchmark_root)
        out_dir = output_root / rel_parent
        out_mp4 = out_dir / "output.mp4"

        raw = json.loads(config_path.read_text(encoding="utf-8"))
        cfg = merge_veo_with_parent(config_path, benchmark_root, raw)
        ref = resolve_ref_path(config_path, benchmark_root, cfg)
        if ref is None:
            print(f"\n[{i}/{len(configs)}] SKIP (no reference image): {config_path}", file=sys.stderr)
            continue

        try:
            build_prompt(cfg)
        except ValueError as e:
            print(f"\n[{i}/{len(configs)}] SKIP ({e}): {config_path}", file=sys.stderr)
            continue

        if args.dry_run:
            print(f"[{i}/{len(configs)}] would run -> {out_mp4}")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[{i}/{len(configs)}] ", end="")
        ok = run_one(
            cfg,
            config_path,
            ref,
            out_mp4,
            ckpt_path,
            attn_type,
            offload,
            steps,
            FRAMES_BY_LENGTH,
            env,
        )
        if not ok:
            print(f"\nFailed: {config_path}", file=sys.stderr)
            sys.exit(1)

    if args.dry_run:
        print("\nDry run complete; no jobs executed.")
    else:
        print(f"\n{'=' * 60}")
        print(f"All {len(configs)} benchmark(s) finished.")
        print(f"Videos under: {output_root}/")
        print("=" * 60)


if __name__ == "__main__":
    main()
