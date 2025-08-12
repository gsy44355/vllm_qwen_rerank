#!/bin/bash

# Docker 入口脚本
set -e

echo "=================================================="
echo "vLLM Rerank 服务启动"
echo "=================================================="

# 显示环境变量
echo "环境变量:"
echo "  HOST: ${HOST:-0.0.0.0}"
echo "  PORT: ${PORT:-8000}"
echo "  MODEL_PATH: ${MODEL_PATH:-Qwen/Qwen3-Reranker-4B}"
echo "  MODEL_SIZE: ${MODEL_SIZE:-4B}"
echo "  GPU_MEMORY_UTILIZATION: ${GPU_MEMORY_UTILIZATION:-0.8}"
echo "  MAX_MODEL_LEN: ${MAX_MODEL_LEN:-10000}"
echo "  LOG_LEVEL: ${LOG_LEVEL:-INFO}"
echo "  WORKERS: ${WORKERS:-1}"
echo ""

# 构建启动命令
CMD_ARGS=(
    "python" "start_service.py"
    "--host" "${HOST:-0.0.0.0}"
    "--port" "${PORT:-8000}"
    "--model-path" "${MODEL_PATH:-Qwen/Qwen3-Reranker-4B}"
    "--model-size" "${MODEL_SIZE:-4B}"
    "--gpu-memory-utilization" "${GPU_MEMORY_UTILIZATION:-0.8}"
    "--max-model-len" "${MAX_MODEL_LEN:-10000}"
    "--log-level" "${LOG_LEVEL:-INFO}"
    "--workers" "${WORKERS:-1}"
)

# 如果有额外的命令行参数，添加到命令中
if [ $# -gt 0 ]; then
    echo "添加额外参数: $@"
    CMD_ARGS+=("$@")
fi

echo "启动命令: ${CMD_ARGS[@]}"
echo "=================================================="

# 执行启动命令
exec "${CMD_ARGS[@]}"
