#!/bin/bash
# ============================================================
# example accelerate train 8 GPU or NPU
# ============================================================

MOSSVG_HOME=$(cd "$(dirname "$0")/../../../" && pwd) && cd ${MOSSVG_HOME}
source scripts/env/env_prepare.sh $(basename "$0" .sh)

# Training parameters
CONFIG="configs/training/mova_train_accelerate_8.py"
ACCELERATE_CONFIG="configs/training/accelerate/fsdp_8.yaml"

set -x
accelerate launch \
        --config_file ${ACCELERATE_CONFIG} \
        scripts/training_scripts/accelerate_train.py ${CONFIG}
set +x