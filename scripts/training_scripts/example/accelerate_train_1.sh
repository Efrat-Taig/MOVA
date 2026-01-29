#!/bin/bash
# ============================================================
# example accelerate train 1 GPU
# ============================================================

MOSSVG_HOME=$(cd "$(dirname "$0")/../../../" && pwd) && cd ${MOSSVG_HOME}
source scripts/env/env_prepare.sh $(basename "$0" .sh)


# Training parameters
CONFIG="configs/training/mova_train_accelerate_1.py"

python scripts/training_scripts/accelerate_train.py ${CONFIG}