# 获取当前脚本所在路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 获取第一个参数
SCRIPT_NAME="$1"

MOVA_HOME="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "============ 🐲 🐲 🐲 🐲 🐲 🐲 🐲 🐲 🐲  MOVA_HOME: ${MOVA_HOME} 🐲 🐲 🐲 🐲 🐲 🐲 🐲 🐲 🐲 ================"
echo ""

export PYTHONPATH="$(pwd):$PYTHONPATH"
if which npu-smi > /dev/null 2>&1; then
     export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
     export HCCL_DETERMINISTIC=TRUE
     ## 在RingAttention计算时候，单机多卡时候容易出现端口冲突
     export HCCL_HOST_SOCKET_PORT_RANGE='auto'
     export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
 else
     export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
     export PYTORCH_ALLOC_CONF=expandable_segments:True
 fi

 ### huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...To disable this warning, you can either:  - Avoid using `tokenizers` before the fork if possible  - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
export TOKENIZERS_PARALLELISM=false

OUTPUT_PATH=$MOVA_HOME/output && mkdir -p $OUTPUT_PATH
LOG_FILE="$OUTPUT_PATH/${SCRIPT_NAME}.log"
exec > >(tee -a "$LOG_FILE")
exec 2>&1