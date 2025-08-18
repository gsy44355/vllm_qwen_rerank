# Gunicorn 配置文件
# 这是一个基础配置文件，可以通过命令行参数覆盖

# 绑定地址和端口
bind = "0.0.0.0:8000"

# 工作进程数
workers = 2

# 工作进程类型
worker_class = "uvicorn.workers.UvicornWorker"

# 超时设置
timeout = 300
keepalive = 5
graceful_timeout = 60

# 预加载应用
preload_app = True

# 临时目录
worker_tmp_dir = "/dev/shm"

# 日志级别
loglevel = "info"

# 最大请求数（禁用以避免频繁重启）
max_requests_jitter = 0

# 应用模块
app_name = "rerank_service:app"
