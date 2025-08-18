# Gunicorn 配置文件 - 支持 CUDA MPS
# 这是一个基础配置文件，可以通过命令行参数覆盖

# 绑定地址和端口
bind = "0.0.0.0:8000"

# 工作进程数 - 支持多进程
workers = 2

# 工作进程类型
worker_class = "uvicorn.workers.UvicornWorker"

# 超时设置
timeout = 300
keepalive = 5
graceful_timeout = 60

# 预加载应用 - 对于 CUDA MPS 建议禁用预加载
preload_app = False

# 临时目录
worker_tmp_dir = "/dev/shm"

# 日志级别
loglevel = "info"

# 最大请求数（禁用以避免频繁重启）
max_requests_jitter = 0

# 应用模块
app_name = "rerank_service:app"

# CUDA MPS 相关配置
# 确保每个工作进程都能正确初始化 CUDA 上下文
worker_connections = 1000
max_requests = 0  # 禁用请求限制，避免频繁重启
