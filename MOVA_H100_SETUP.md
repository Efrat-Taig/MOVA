# MOVA-720p: Verified Setup Guide (NVIDIA H100)

> **Verification status:** Cross-referenced against `scripts/inference_single.py` and working terminal commands on this machine. All parameters match the codebase.

**Automated setup:** Run `./setup_mova_h100.sh` for a one-shot installation. Use `--skip-clone` if the repo exists, `--skip-download` to skip model fetch, and `--run-test` to run inference after setup.

## 1. Environment (Python 3.13)

```bash
conda create -n mova python=3.13 -y
conda activate mova
pip install torch torchvision torchaudio
```

## 2. Repo & Deps

```bash
git clone https://github.com/OpenMOSS/MOVA.git
cd MOVA
pip install -e .
pip install diffusers transformers accelerate decord opencv-python bitsandbytes yunchang moviepy av huggingface_hub hf_transfer
```

## 3. Backend Check (Single GPU / H100)

The inference script uses `dist.init_process_group(backend="nccl")` for distributed init. **Current upstream already uses `nccl`**; no patch is needed.

If you have an older clone that still uses `backend="gloo"`, run:

```bash
sed -i 's/backend="gloo"/backend="nccl"/g' scripts/inference_single.py
```

To verify: `grep -n 'backend=' scripts/inference_single.py` should show `backend="nccl"`.

## 4. High-Speed Download

```bash
mkdir -p checkpoints/MOVA-720p
export HF_HUB_ENABLE_HF_TRANSFER=1
huggingface-cli download OpenMOSS-Team/MOVA-720p --local-dir checkpoints/MOVA-720p --local-dir-use-symlinks False
```

## 5. Verified Inference Command

All arguments below are defined in `scripts/inference_single.py` and have been tested:

```bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
torchrun --nproc_per_node=1 scripts/inference_single.py \
    --ckpt_path checkpoints/MOVA-720p \
    --prompt "A cinematic shot of a rainy neon city at night, heavy thunder sounds and splashing water" \
    --ref_path assets/single_person.jpg \
    --output_path ./outputs/mova_test.mp4 \
    --height 720 --width 1280 --attn_type torch_flash --offload cpu
```

### Argument Reference

| Argument      | Required | Default   | Notes                                      |
|---------------|----------|-----------|--------------------------------------------|
| `--ckpt_path` | Yes      | —         | Path to MOVA-720p checkpoint directory      |
| `--prompt`    | Yes      | —         | Text description for video + audio          |
| `--ref_path`  | Yes      | —         | First-frame reference image (e.g. `assets/single_person.jpg`) |
| `--output_path` | No    | `./data/samples/output.mp4` | Output video path       |
| `--height`    | No       | 720       | Video height                               |
| `--width`     | No       | 1280      | Video width                                |
| `--attn_type` | No       | `fa`      | `torch_flash` recommended for H100          |
| `--offload`   | No       | `none`    | `cpu` reduces VRAM, uses more host RAM      |

## 6. Troubleshooting

- **NCCL errors on H100:** Ensure `NCCL_P2P_DISABLE=1` and `NCCL_IB_DISABLE=1` are set before `torchrun`.
- **Missing reference image:** Use `assets/single_person.jpg` or `assets/verse_bench.jpg` from the repo, or provide your own image path.
- **Out of VRAM:** Use `--offload cpu` or `--offload group` (see main README for benchmarks).
