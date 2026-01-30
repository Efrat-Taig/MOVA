# MOVA Ascend NPU 支持

## 概述

`MOVA` ([**Mo**ss **V**ideo & **A**udio](https://mosi.cn/)) 是**首个原生支持华为 Ascend NPU 的开源视频音频统一生成模型**，在 Ascend 硬件上提供高性能、同步的视频音频生成，并支持原生 LoRA 微调和推理加速。本文档提供了在 Ascend NPU 上部署、训练和优化 `MOVA` 的全面指导。


## 目录
- [概述](#概述)
- [硬件要求](#硬件要求)
- [软件环境](#软件环境)
- [安装](#安装)
- [快速开始](#快速开始)
- [性能优化](#性能优化)
- [故障排除](#故障排除)
- [架构概述](#架构概述)
- [支持](#支持)
- [致谢](#致谢)
- [更新日志](#更新日志)

---

## 硬件要求

### 支持的 NPU 型号

- **Ascend 910B** 系列（推荐）

### 推荐配置

- 8x Ascend 910B NPU（用于多 NPU 训练）
- 每卡 ≥ 64 GB NPU 内存
- 512 GB 系统内存
- 2 TB+ 存储空间用于模型和数据集
- x86_64（推荐）/ARM64 架构

---

## 软件环境

### 操作系统

- Ubuntu 24.04 LTS（推荐）

### 软件栈

- **CANN (Compute Architecture for Neural Networks)**: 8.3.RC2 或更高版本
- **PyTorch**: 2.8.0+ 支持 NPU
- **torch_npu**: 与 PyTorch 版本兼容
- **Python**: 3.11+


## 安装

1. **拉取即用型 `Ascend 910B` docker 镜像**

要在 Ascend NPU 上运行 `MOVA`，您需要安装 Ascend 软件栈。现在为了用户友好性，无需手动安装任何 Ascend 软件栈，只需拉取即用型容器镜像即可即时部署。

```bash
docker pull swr.cn-north-4.myhuaweicloud.com/ascend-sact/ascend-910b-ubuntu:v2.8
```

或拉取最新版本 `LingJing-灵镜` docker 镜像。更多详情请参考 [Ascend SACT](https://gitcode.com/Ascend-SACT/ascend-docker) 文档。

```bash
docker pull swr.cn-north-4.myhuaweicloud.com/ascend-sact/ascend-910b-ubuntu:latest
```

然后按如下方式启动 ascend docker 镜像：

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
将 `/data` 替换为您自己的挂载数据路径。

2. **拉取 MOVA 源代码并切换到 `feat/npu` 分支**

```bash
cd $PERSONAL_WORKSPACE

git clone -b feat/npu https://github.com/OpenMOSS/MOVA.git && cd MOVA
```
将 `$PERSONAL_WORKSPACE` 替换为您自己的工作空间路径。

3. **从 [huggingface 发布页面](https://huggingface.co/collections/OpenMOSS-Team/mova) 下载 360p 和 720p 预训练模型权重。**

```bash
## 360p 分辨率模型
hf download OpenMOSS-Team/MOVA-360p --local-dir ${PERSONAL_WORKSPACE}/MOVA/ckpts/MOVA-360p

## 720p 分辨率模型
hf download OpenMOSS-Team/MOVA-720p --local-dir ${PERSONAL_WORKSPACE}/MOVA/ckpts/MOVA-720p
```

## 快速开始

### 视频生成（1 NPU）
修改 `scripts/diffusers_example_1_360p.sh` 以将 `ckpt_path` 设置为预训练模型权重的路径。

然后使用默认配置运行视频生成脚本。视频生成脚本将在 1 个 NPU 上运行。
```bash
cd ${PERSONAL_WORKSPACE}/MOVA/ && bash scripts/diffusers_example_1_360p.sh
```
输出视频将保存在 `${PERSONAL_WORKSPACE}/MOVA/data/samples/` 目录中。

**推理参数**，您可以在 `scripts/diffusers_example_1.sh` 中修改推理参数。

| 参数 | 描述 | 默认值 |
|-----------|-------------|---------|
| `--height` | 视频高度 | 352 |
| `--width` | 视频宽度 | 640 |
| `--num_frames` | 帧数 | 193 |
| `--num_inference_steps` | 去噪步数 | 25 |
| `--seed` | 随机种子 | 42 |

---

### 视频生成（多 NPU 并支持 360p 和 720p）

以 4 NPU 推理为例，修改 `scripts/diffusers_example_4_360p.sh` 以将 `ckpt_path` 设置为预训练模型权重的路径。

然后使用默认配置运行视频生成脚本。视频生成脚本将在 4 个 NPU 上运行。

```bash
cd ${PERSONAL_WORKSPACE}/MOVA/ && bash scripts/diffusers_example_4_360p.sh
```
输出视频将保存在 `${PERSONAL_WORKSPACE}/MOVA/data/samples/` 目录中。

### LoRA 微调（8 NPU）

1. 按照以下说明准备您的 lora 数据集，或直接使用 huggingface 提供的公共数据集。

为训练数据创建元数据文件（JSONL 格式），每个数据需要两个字段：`video_path` 和 `caption`。
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
2. 编辑训练配置文件 `configs/training/mova_train_accelerate_8.py` 并启动 lora 训练
- 将 `from_pretrained` 设置为预训练模型权重的路径
- 将 `data_root` 设置为 lora 数据集的路径
- 将 `metadata_file` 设置为元数据文件的路径

```bash
# 启动训练

cd ${PERSONAL_WORKSPACE}/MOVA/ && bash scripts/training_scripts/example/accelerate_train_8.sh
```
微调后的模型权重将保存在 `${PERSONAL_WORKSPACE}/MOVA/checkpoints/` 目录中。

3. 监控训练过程
训练日志自动保存到 `${PERSONAL_WORKSPACE}/MOVA/output/` 目录：

```bash
# 查看训练日志
cd ${PERSONAL_WORKSPACE}/MOVA/ && tail -f output/accelerate_train_8.log

# 监控 NPU 使用情况
npu-smi info
```

### LoRA 推理

[待完成]

### SGLang 推理
[待完成]

### Ascend 性能参考

我们提供了在不同条件下生成 `8 秒` 视频的推理基准。`ascend910B` 具有 `64G` NPU-VRAM 和 `376TFlops`（bfloat16）

| 模式                                      | NPU 数量           | 峰值 NPU VRAM（每 NPU） |  步骤时间 (s)  |           
| ----------------------------------------- | -------------- | -----------------------:| ---------------:|
|  推理用例1 (360p & 单 NPU)      | 1 × ascend910B |     ≈46GB               |  37.0           | 
|  推理用例2 (360p & 4*NPU)          | 4 × ascend910B |     ≈42GB               |  12.3           | 
|  推理用例3 (360p & 8*NPU)          | 8 × ascend910B |     ≈42GB               |  7.6            |
|  推理用例4 (720p & 8*NPU)          | 8 × ascend910B |     ≈47GB               |  60.6           |  
| Accelerate + FSDP LoRA (360p & 8*NPU)    | 8 × ascend910B |     ≈39GB               |  34.1           | 
|         LoRA 推理                    | 1 × ascend910B |     ≈待完成               |  待完成           |
|         SGLang 推理                    | 1 × ascend910B |     ≈待完成               |  待完成           |
---

## 性能优化

### NPU 和 GPU 环境自动检测

项目包括自动环境检测和设置：

```bash
# 源环境准备脚本
source scripts/env/env_prepare.sh
```

此脚本自动：
- 检测 NPU 与 GPU 硬件
- 设置适当的设备可见性（`ASCEND_RT_VISIBLE_DEVICES` 或 `CUDA_VISIBLE_DEVICES`）
- 为 NPU 配置 HCCL 参数
- 设置内存分配策略

---

### NPU 特定优化

代码库包括自动 NPU 优化：

1. **长上下文注意力**：使用 NPU 优化的 [USPAttention(ulysses 和 ring attention 内核)](https://github.com/feifeibear/long-context-attention) 进行内存高效注意力计算
2. **快速 GELU**：使用 `torch_npu.fast_gelu` 进行更快的激活
3. **RMS 归一化**：使用 `torch_npu.npu_rms_norm` 进行优化归一化
4. **旋转位置嵌入**：使用 `torch_npu.npu_rotary_mul` 进行位置编码
5. **内存管理**：配置为 99% 内存比例

### 内存配置

```python
# 在 adapter.py 中设置（自动应用）
torch_npu.npu.set_per_process_memory_fraction(0.99)
torch_npu.npu.config.allow_internal_format = False
```

### HCCL 配置

```bash
# 在 env_prepare.sh 中设置
export HCCL_DETERMINISTIC=TRUE
export HCCL_HOST_SOCKET_PORT_RANGE='auto'
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
```

### 性能提示

1. **使用 BF16**：始终使用 bfloat16 以获得更好的性能和内存效率
2. **批次大小**：根据可用的 NPU 内存进行调整
3. **梯度累积**：增加以获得有效的更大批次大小。目前批次大小默认设置为 `1`。
4. **数据加载**：使用多个工作进程进行高效数据加载

---

## 故障排除

### 常见问题

#### 1. 未检测到 NPU

**问题**：`torch_npu` 导入失败或 NPU 不可用。可能是驱动程序未正确挂载到容器中。

**解决方案**：
```bash
# 检查 CANN 安装
npu-smi info

# 验证 torch_npu 安装
python -c "import torch_npu; print(torch_npu.__version__)"
```

#### 2. 内存不足

**问题**：训练因 OOM 错误而失败

**解决方案**：
- 减少批次大小
- 启用梯度检查点
- 启用混合精度训练
- 启用参数卸载到 CPU
- 需要时将参数从 CPU 加载到 NPU
- 减少 LoRA 秩

#### 3. ARM64 架构未完全验证

**问题**：在 ARM64 架构上训练失败

由于 `ascend-910b-ubuntu:v2.8` 中缺少 `torchcodec==0.7.0` (arm64) 的 pip 包，用户需要通过源代码手动安装 `torchcodec`。

### 4. 其他

1. 警告日志 `Failed to load CPU gemm_4bit_forward from kernels-community: No module named 'kernels'. Please make sure you already 'pip install kernels' and the kernels >= 0.11.1` 不影响结果，由于 `bitsandbytes` 包的原因。它将被移除或稍后支持 NPU 上的低精度训练。

---

## 架构概述

### NPU 适配层

项目使用统一的适配层（[`mova/utils/adapter.py`](mova/utils/adapter.py)），自动检测并适配 NPU 或 GPU：

- **设备检测**：自动 NPU/GPU 检测
- **后端选择**：选择适当的计算后端
- **内核优化**：在可用时使用 NPU 优化的内核
- **内存格式**：处理不同的内存格式

### 分布式训练

- **后端**：NPU 使用 HCCL，GPU 使用 NCCL
- **策略**：FSDP 用于内存高效的分布式训练
- **上下文并行**：Ring attention 和 ulysses 用于长序列

### 模型组件

- **视频 DiT**：视频扩散 Transformer
- **音频 DiT**：音频扩散 Transformer
- **双塔桥接**：跨模态注意力桥接
- **LoRA 层**：参数高效的微调适配器

---

## 支持

如有问题和疑问：

1. 查看 [故障排除](#故障排除) 部分
2. 搜索现有的 GitHub 问题
3. 创建包含详细信息的新问题
4. 包括 NPU 型号、CANN 版本和错误日志

---

## 致谢
- 原始 `MOVA 作者`
- [华为 Ascend](https://gitcode.com/Ascend) 团队提供 NPU 支持
- [PyTorch NPU](https://gitcode.com/Ascend/pytorch) 和 [Ascend-SACT](https://gitcode.com/Ascend-SACT) 社区

---

## 更新日志

### 版本 0.1.0（当前）

- 初始 NPU 支持
- NPU 上的 LoRA 微调
- 多 NPU 分布式训练
- FSDP 集成
- 上下文并行支持
- NPU 优化的内核
