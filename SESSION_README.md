# From login to running MOVA (`efrat_run.py`)

Use this checklist whenever you open a new terminal or a new machine.

---

## Every new terminal (environment already set up)

1. **Activate Conda** (use the same env name you created, e.g. `mova`):

   ```bash
   conda activate mova
   ```

2. **Go to the repo**:

   ```bash
   cd /path/to/MOVA
   ```

   Example: `cd ~/MOVA`

3. **Optional sanity check** (GPU + PyTorch):

   ```bash
   nvidia-smi
   python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.__version__)"
   ```

   You want `CUDA: True` and a version tag like `+cu126` (CUDA 12.x wheels on a CUDA 12.x driver).

4. **Reference image** must exist for the path in `efrat_run.py` (default: `inputs/sky_in_heli.jpg`). Either add that file or edit `REF_PATH` in `efrat_run.py`.

5. **Checkpoint** folder must exist: `checkpoints/MOVA-720p` (see “First time on this machine” below if missing).

6. **Run**:

   ```bash
   python efrat_run.py
   ```

   Output video path is printed first (under `outputs/`). Inference can take a long time at 720p with CPU offload.

---

## First time on this machine

### Automated (recommended)

From the cloned repo (after installing [Miniconda](https://docs.conda.io/en/latest/miniconda.html) if needed):

```bash
cd /path/to/MOVA
bash efrat_bootstrap_new_machin --download-model
```

`--download-model` pulls **MOVA-720p** from Hugging Face (large download). Omit the flag if you will copy checkpoints in yourself.

Then put your reference image at `inputs/sky_in_heli.jpg` (or edit `REF_PATH` in `efrat_run.py`).

### Manual steps (if you prefer)

Do these once (or after wiping the environment).

#### 1. Conda + Python 3.13

```bash
conda create -n mova python=3.13 -y
conda activate mova
```

(Or: `conda env create -f environment.yml` then `conda activate mova`.)

#### 2. PyTorch aligned with your NVIDIA driver (important)

On many cloud GPUs the driver reports **CUDA 12.x**. Install **torch, torchvision, and torchaudio** from the **same** CUDA 12 wheel index **before** `pip install -e .`, so you avoid errors like `libcudart.so.13` or “no GPUs found” with a mismatched PyTorch build:

```bash
bash scripts/install_pytorch_cuda126.sh
```

#### 3. Install MOVA in editable mode

```bash
cd /path/to/MOVA
pip install -e .
```

Helper packages (optional but common):

```bash
pip install moviepy av huggingface_hub hf_transfer
```

#### 4. Download weights (~large disk usage)

```bash
mkdir -p checkpoints/MOVA-720p
pip install huggingface_hub
export HF_HUB_ENABLE_HF_TRANSFER=1   # optional, faster download
huggingface-cli download OpenMOSS-Team/MOVA-720p \
  --local-dir checkpoints/MOVA-720p \
  --local-dir-use-symlinks False
```

#### 5. Single-GPU script (`scripts/inference_single.py`)

For one GPU, the repo expects NCCL and CUDA to work. `efrat_run.py` sets `NCCL_P2P_DISABLE=1` and `NCCL_IB_DISABLE=1` for the child process to reduce flaky NCCL behavior on some setups.

---

## If something fails

| Symptom | What to try |
|--------|-------------|
| `conda activate mova` → env not found | Create the env (see “First time”) or run `conda info --envs` and use the correct name. |
| `libcudart.so.13` or torchaudio load error | Re-run `bash scripts/install_pytorch_cuda126.sh` so **torchaudio** matches **torch**. |
| `CUDA: False` with a GPU present | PyTorch build does not match the driver; reinstall with `scripts/install_pytorch_cuda126.sh` (or a PyTorch build that matches your driver; see [pytorch.org](https://pytorch.org)). |
| `ProcessGroupNCCL` / no GPUs | Usually same as above: fix PyTorch + driver, then retry. |
| `FileNotFoundError` for `ref_path` | Create the image path or change `REF_PATH` in `efrat_run.py`. |
| Missing checkpoint | Download `MOVA-720p` into `checkpoints/MOVA-720p` (step 4 above). |

---

## Official upstream docs

Project home and full instructions: [OpenMOSS/MOVA](https://github.com/OpenMOSS/MOVA).
