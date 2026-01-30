# MOVA Ascend NPU Support

## Overview

`MOVA` ([**Mo**ss **V**ideo & **A**udio](https://mosi.cn/)) is the **first open-source video-audio unified generation model** natively supporting Huawei Ascend NPU, delivering high-performance, synchronized video-audio generation with native LoRA fine-tuning and inference acceleration on Ascend hardware. This document offers comprehensive guidance to deploy, train, and optimize `MOVA` on Ascend NPUs.


## Table of Contents
- [Overview](#overview)
- [Hardware Requirements](#hardware-requirements)
- [Software Environment](#software-environment)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)
- [Architecture Overview](#architecture-overview)
- [Support](#support)
- [Acknowledgments](#acknowledgments)
- [Changelog](#changelog)

---

## Hardware Requirements

### Supported NPU Models

- **Ascend 910B** series (recommended)

### Recommended Configuration

- 8x Ascend 910B NPUs (for multi-NPU training)
- ≥ 64 GB NPU memory per card 
- 512 GB system memory 
- 2 TB+ storage space for models and datasets
- x86_64  (recommended) /ARM64 architecture

---

## Software Environment

### Operating System

- Ubuntu 24.04 LTS (recommended)

### Software Stack

- **CANN (Compute Architecture for Neural Networks)**: 8.3.RC2 or higher
- **PyTorch**: 2.8.0+ with NPU support
- **torch_npu**: Compatible with PyTorch version
- **Python**: 3.11+


## Installation

1. **pull the ready-to-use `Ascend 910B` docker image**

To run `MOVA` on Ascend NPU, you need to have the Ascend software stack installed. Now for the user-friendliness , no need to manually install any Ascend software stack, simply pull the ready-to-use container image for instant deployment.

```bash
docker pull swr.cn-north-4.myhuaweicloud.com/ascend-sact/ascend-910b-ubuntu:v2.8
```

or pull the latest version `LingJing-灵镜` docker image. for more details, please refer to the [Ascend SACT](https://gitcode.com/Ascend-SACT/ascend-docker) documentation.

```bash
docker pull swr.cn-north-4.myhuaweicloud.com/ascend-sact/ascend-910b-ubuntu:latest
```

then start the ascend docker image as follows:

```bash
docker run -it --rm --name ascend-910b-mova \
    --privileged \
    --network host \
    --device /dev/davinci_manager \
    --device /dev/davinci0 \
    --device /dev/davinci1 \
    --device /dev/davinci2 \
    --device /dev/davinci3 \
    --device /dev/davinci4 \
    --device /dev/davinci5 \
    --device /dev/davinci6 \
    --device /dev/davinci7 \
    --device /dev/hisi_hdc \
    --device /dev/devmm_svm \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
    -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /data:/data \
    swr.cn-north-4.myhuaweicloud.com/ascend-sact/ascend-910b-ubuntu:v2.8 \
    bash
```
Replace `/data` with your own mount data path.

2. **pull the MOVA source code and checkout to the `feat/npu` branch**

```bash
cd $PERSONAL_WORKSPACE

git clone -b feat/npu https://github.com/OpenMOSS/MOVA.git && cd MOVA
```
replace `$PERSONAL_WORKSPACE` with your own workspace path.

3. **download the pre-trained model weights both 360p and 720p from the [huggingface release page](https://huggingface.co/collections/OpenMOSS-Team/mova).**

```bash
## 360p Resolution Model
hf download OpenMOSS-Team/MOVA-360p --local-dir ${PERSONAL_WORKSPACE}/MOVA/ckpts/MOVA-360p

## 720p Resolution Model
hf download OpenMOSS-Team/MOVA-720p --local-dir ${PERSONAL_WORKSPACE}/MOVA/ckpts/MOVA-720p
```

## Quick Start

### Video Generation (1 NPUs)
modify the `scripts/diffusers_example_1_360p.sh` to set the `ckpt_path` to the path of the pre-trained model weights.

Then run the video generation script with default configuration. The video generation script will run on 1 NPU.
```bash
cd ${PERSONAL_WORKSPACE}/MOVA/ && bash scripts/diffusers_example_1_360p.sh
```
The output video will be saved in the `${PERSONAL_WORKSPACE}/MOVA/data/samples/` directory.

**Inference Parameters** , you can modify the inference parameters in the `scripts/diffusers_example_1.sh`.

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--height` | Video height | 352 |
| `--width` | Video width | 640 |
| `--num_frames` | Number of frames | 193 |
| `--num_inference_steps` | Denoising steps | 25 |
| `--seed` | Random seed | 42 |

---

### Video Generation (multi NPUs and support 360p & 720p) 

Take 4 NPUs inference case as example, modify the `scripts/diffusers_example_4_360p.sh` to set the `ckpt_path` to the path of the pre-trained model weights.

Then run the video generation script with default configuration. The video generation script will run on 4 NPUs.

```bash
cd ${PERSONAL_WORKSPACE}/MOVA/ && bash scripts/diffusers_example_4_360p.sh
```
The output video will be saved in the `${PERSONAL_WORKSPACE}/MOVA/data/samples/` directory.

### LoRA Fine-tuning (8 NPUs)

1. prepare your lora datasets as following instruction or simply use the huggingface dataset public provided.


Create a metadata file (JSONL format) for your training data, each data need two fields: `video_path` and `caption`.
```jsonl
[
  {
    "video_path": "/data/datasets/09266f072b4545bf03cbfb05700be1d6_0011_589.671_597.721.mp4",
    "caption": "A teal‑haired character wearing a white shirt with red trim, a green plaid skirt and a blue hair clip glides leftward and exits the frame"
  },
  {
    "video_path": "/data/datasets/09266f072b4545bf03cbfb05700be1d6_0006_72.770_80.820.mp4",
    "caption": "The video opens on a light‑pink backdrop dotted with darker pink polka‑spots, where a blonde‑haired girl in a black‑and‑red school uniform clutches a bag and stands beside a pink‑haired girl wearing a white shirt, gray skirt and a red bow."
  }]
```
2. edit the training configuration file `configs/training/mova_train_accelerate_8.py` and start lora training
- to set the `from_pretrained` to the path of the pre-trained model weights
- to set the `data_root` to the path of the lora datasets
- to set the `metadata_file` to the path of the metadata file

```bash
# Start training

cd ${PERSONAL_WORKSPACE}/MOVA/ && bash scripts/training_scripts/example/accelerate_train_8.sh
```
the fine-tuned model weights will be saved in the `${PERSONAL_WORKSPACE}/MOVA/checkpoints/` directory.

3. monitoring training process
Training logs are automatically saved to `${PERSONAL_WORKSPACE}/MOVA/output/` directory:

```bash
# View training logs
cd ${PERSONAL_WORKSPACE}/MOVA/ && tail -f output/accelerate_train_8.log

# Monitor NPU usage
npu-smi info
```

### LoRA Inference

[TODO]

### SGLang Inference
[TODO]

### Performance Reference On Ascend

We provide inference benchmarks for generating an `8-second` videos under different conditions. `ascend910B` with `64G` NPU-VRAM and `376TFlops` (bfloat16)

| Mode                                      | NPUs           | Peak NPU VRAM (per NPU) |  Step Time (s)  |           
| ----------------------------------------- | -------------- | -----------------------:| ---------------:|
|  inference case1 (360p & single NPU)      | 1 × ascend910B |     ≈46GB               |  37.0           | 
|  inference case2 (360p & 4*NPUs)          | 4 × ascend910B |     ≈42GB               |  12.3           | 
|  inference case3 (360p & 8*NPUs)          | 8 × ascend910B |     ≈42GB               |  7.6            |
|  inference case4 (720p & 8*NPUs)          | 8 × ascend910B |     ≈47GB               |  60.6           |  
| Accelerate + FSDP LoRA (360p & 8*NPUs)    | 8 × ascend910B |     ≈39GB               |  34.1           | 
|         LoRA inference                    | 1 × ascend910B |     ≈TODO               |  TODO           |
|         SGLang inference                    | 1 × ascend910B |     ≈TODO               |  TODO           |
---

## Performance Optimization

### NPU and GPU environment autodetect

The project includes automatic environment detection and setup:

```bash
# Source environment preparation script
source scripts/env/env_prepare.sh
```

This script automatically:
- Detects NPU vs GPU hardware
- Sets appropriate device visibility (`ASCEND_RT_VISIBLE_DEVICES` or `CUDA_VISIBLE_DEVICES`)
- Configures HCCL parameters for NPU
- Sets memory allocation policies

---

### NPU-Specific Optimizations

The codebase includes automatic NPU optimizations:

1. **Long Context Attention**: Uses NPU-optimized [USPAttention(ulysses and ring attention kernels)](https://github.com/feifeibear/long-context-attention) for memory-efficient attention
2. **Fast GELU**: Uses `torch_npu.fast_gelu` for faster activation
3. **RMS Norm**: Uses `torch_npu.npu_rms_norm` for optimized normalization
4. **Rotary Pos Embedding**: Uses `torch_npu.npu_rotary_mul` for position encoding
5. **Memory Management**: Configured with 99% memory fraction

### Memory Configuration

```python
# Set in adapter.py (automatically applied)
torch_npu.npu.set_per_process_memory_fraction(0.99)
torch_npu.npu.config.allow_internal_format = False
```

### HCCL Configuration

```bash
# Set in env_prepare.sh
export HCCL_DETERMINISTIC=TRUE
export HCCL_HOST_SOCKET_PORT_RANGE='auto'
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
```

### Performance Tips

1. **Use BF16**: Always use bfloat16 for better performance and memory efficiency
2. **Batch Size**: Adjust based on available NPU memory
3. **Gradient Accumulation**: Increase for effective larger batch sizes. and now batch size is set to `1` by default.
4. **Data Loading**: Use multiple workers for efficient data loading

---

## Troubleshooting

### Common Issues

#### 1. NPU Not Detected

**Problem**: `torch_npu` import fails or NPU not available. Maybe the Driver is not correctly mounted into the container.

**Solution**:
```bash
# Check CANN installation
npu-smi info

# Verify torch_npu installation
python -c "import torch_npu; print(torch_npu.__version__)"
```

#### 2. Out of Memory

**Problem**: Training fails with OOM error

**Solutions**:
- Reduce batch size
- Enable gradient checkpointing
- Enable mixed precision training
- Enable params offloading to CPU
- Load params from CPU to NPU when needed
- Reduce LoRA rank

#### 3. ARM64 Architecture not fully validated

**Problem**: Training fails on ARM64 architecture

Due to the lack of pip package of `torchcodec==0.7.0` (arm64) in the `ascend-910b-ubuntu:v2.8`, users need to install `torchcodec` manually by source.

### 4. others

1. warning logs `Failed to load CPU gemm_4bit_forward from kernels-community: No module named 'kernels'. Please make sure you already 'pip install kernels' and the kernels >= 0.11.1` do not effect the result, due to the package of `bitsandbytes`. It will be removed or to support low precision training on NPU later.

---

## Architecture Overview

### NPU Adapter Layer

The project uses a unified adapter layer ([`mova/utils/adapter.py`](mova/utils/adapter.py)) that automatically detects and adapts to NPU or GPU:

- **Device Detection**: Automatic NPU/GPU detection
- **Backend Selection**: Chooses appropriate compute backend
- **Kernel Optimization**: Uses NPU-optimized kernels when available
- **Memory Format**: Handles different memory formats

### Distributed Training

- **Backend**: HCCL for NPU, NCCL for GPU
- **Strategy**: FSDP for memory-efficient distributed training
- **Context Parallel**: Ring attention and ulysses for long sequences

### Model Components

- **Video DiT**: Video diffusion transformer
- **Audio DiT**: Audio diffusion transformer
- **Dual Tower Bridge**: Cross-modal attention bridge
- **LoRA Layers**: Parameter-efficient fine-tuning adapters

---

## Support

For issues and questions:

1. Check the [Troubleshooting](#troubleshooting) section
2. Search existing GitHub issues
3. Create a new issue with detailed information
4. Include NPU model, CANN version, and error logs

---

## Acknowledgments
- Original `MOVA authors`
- [Huawei Ascend](https://gitcode.com/Ascend) team for NPU support
- [PyTorch NPU](https://gitcode.com/Ascend/pytorch) & & [Ascend-SACT](https://gitcode.com/Ascend-SACT) community

---

## Changelog

### Version 0.1.0 (Current)

- Initial NPU support
- LoRA fine-tuning on NPU
- Multi-NPU distributed training
- FSDP integration
- Context Parallel support
- NPU-optimized kernels
