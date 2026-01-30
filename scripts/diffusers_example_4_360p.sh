#!/bin/bash
# ============================================================
# example diffusers inference 8 GPU or NPU
# ============================================================

MOVA_HOME=$(cd "$(dirname "$0")/../" && pwd) && cd ${MOVA_HOME}
source scripts/env/env_prepare.sh $(basename "$0" .sh)


CP_SIZE=4
torchrun \
    --nproc_per_node=$CP_SIZE \
    scripts/inference_single.py \
    --ckpt_path /path/to/MOVA-360p \
    --cp_size $CP_SIZE \
    --height 352 \
    --width 640 \
    --prompt "A man in a blue blazer and glasses speaks in a formal indoor setting, framed by wooden furniture and a filled bookshelf. Quiet room acoustics underscore his measured tone as he delivers his remarks. At one point, he says, \"I would also say that this election in Germany wasn’t surprising.\"" \
    --ref_path "./assets/single_person.jpg" \
    --output_path "./data/samples/single_person.mp4" \
    --offload "cpu" \
    --seed 42 \
    --num_inference_steps 25