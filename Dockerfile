# 使用官方 Python 3.10 镜像（slim 版，体积小）
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量（这些在 docker-compose.yml 可以覆盖）
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    DEBIAN_FRONTEND=noninteractive \
    MODEL_PATH=Qwen/Qwen3-Reranker-4B \
    MODEL_SIZE=4B \
    GPU_MEMORY_UTILIZATION=0.8 \
    MAX_MODEL_LEN=10000

# 安装系统依赖和 GPU 运行必备工具
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 复制 requirements.txt
COPY requirements.txt .

# 安装 Python 依赖（如果需要 GPU，可在 requirements.txt 里写 torch==x.y.z+cu118）
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY rerank_service.py .
COPY start_service.py .
COPY docker-entrypoint.sh .
COPY start_gunicorn.py .

# 创建非 root 用户，并确保日志目录可写
RUN useradd --create-home --shell /bin/bash app && \
    mkdir -p /app/logs && \
    chown -R app:app /app && \
    chmod +x /app/docker-entrypoint.sh
USER app

# 暴露端口
EXPOSE 8000

# 启动命令 - 使用入口脚本
ENTRYPOINT ["/app/docker-entrypoint.sh"]
