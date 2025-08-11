# Docker 部署指南

本文档详细说明如何使用Docker部署vLLM Rerank服务。

## 前置要求

### 1. 安装Docker

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install docker.io docker-compose

# CentOS/RHEL
sudo yum install docker docker-compose

# macOS
brew install docker docker-compose

# Windows
# 下载并安装 Docker Desktop
```

### 2. 安装NVIDIA Docker（可选，用于GPU支持）

```bash
# 安装NVIDIA Docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

## 快速开始

### 1. 一键部署

```bash
# 给脚本执行权限
chmod +x docker-deploy.sh

# 一键部署（包含构建、启动、测试）
./docker-deploy.sh deploy
```

### 2. 分步部署

```bash
# 1. 构建镜像
./docker-deploy.sh build

# 2. 启动服务
./docker-deploy.sh start

# 3. 查看状态
./docker-deploy.sh status

# 4. 测试服务
./docker-deploy.sh test
```

## 使用docker-compose

### 基本操作

```bash
# 构建并启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down

# 重启服务
docker-compose restart

# 查看状态
docker-compose ps
```

### 自定义配置

编辑 `docker-compose.yml` 文件来自定义配置：

```yaml
version: '3.8'

services:
  rerank-service:
    build: .
    container_name: vllm-rerank-service
    ports:
      - "8000:8000"  # 修改端口映射
    environment:
      - PYTHONUNBUFFERED=1
      - CUDA_VISIBLE_DEVICES=0  # 指定GPU设备
    volumes:
      - model_cache:/root/.cache/huggingface
      - ./logs:/app/logs
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
```

## 管理命令

### 查看服务状态

```bash
# 查看容器状态
./docker-deploy.sh status

# 查看资源使用情况
docker stats vllm-rerank-service
```

### 查看日志

```bash
# 实时查看日志
./docker-deploy.sh logs

# 查看最近100行日志
docker-compose logs --tail=100

# 查看错误日志
docker-compose logs | grep ERROR
```

### 重启服务

```bash
# 重启服务
./docker-deploy.sh restart

# 或者使用docker-compose
docker-compose restart
```

### 停止服务

```bash
# 停止服务
./docker-deploy.sh stop

# 或者使用docker-compose
docker-compose down
```

### 清理资源

```bash
# 清理所有容器、镜像和数据
./docker-deploy.sh cleanup
```

## 故障排除

### 1. 构建失败

```bash
# 清理缓存重新构建
docker system prune -f
docker build --no-cache -t vllm-rerank-service .
```

### 2. 启动失败

```bash
# 查看详细错误信息
docker-compose logs

# 检查端口占用
netstat -tulpn | grep 8000

# 检查GPU可用性
nvidia-smi
```

### 3. GPU相关问题

```bash
# 测试NVIDIA Docker
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# 检查GPU驱动
nvidia-smi

# 检查Docker GPU支持
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

### 4. 内存不足

```bash
# 查看容器资源使用情况
docker stats vllm-rerank-service

# 调整GPU内存使用率（在rerank_service.py中修改）
# gpu_memory_utilization=0.6  # 降低到60%
```

## 生产环境部署

### 1. 使用外部数据库

```yaml
# 在docker-compose.yml中添加数据库服务
services:
  postgres:
    image: postgres:13
    environment:
      POSTGRES_DB: rerank
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data

  rerank-service:
    # ... 其他配置
    depends_on:
      - postgres
    environment:
      - DATABASE_URL=postgresql://user:password@postgres:5432/rerank
```

### 2. 使用反向代理

```yaml
# 添加Nginx反向代理
services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - rerank-service
```

### 3. 监控和日志

```yaml
# 添加监控服务
services:
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
```

## 性能优化

### 1. 模型缓存

```bash
# 挂载模型缓存目录
volumes:
  - /path/to/model/cache:/root/.cache/huggingface
```

### 2. GPU优化

```yaml
# 在docker-compose.yml中优化GPU配置
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
    limits:
      memory: 8G  # 限制内存使用
```

### 3. 网络优化

```yaml
# 使用host网络模式（仅Linux）
network_mode: host
```

## 安全考虑

### 1. 非root用户

Dockerfile中已经配置了非root用户运行：

```dockerfile
RUN useradd --create-home --shell /bin/bash app
USER app
```

### 2. 网络安全

```yaml
# 限制网络访问
networks:
  default:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

### 3. 资源限制

```yaml
# 限制资源使用
deploy:
  resources:
    limits:
      cpus: '2.0'
      memory: 8G
    reservations:
      cpus: '1.0'
      memory: 4G
```

## 常见问题

### Q: 容器启动后立即退出怎么办？

A: 检查日志查看错误信息：
```bash
docker-compose logs
```

### Q: GPU不可用怎么办？

A: 检查NVIDIA Docker安装：
```bash
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

### Q: 模型下载很慢怎么办？

A: 使用国内镜像源或手动下载模型到缓存目录。

### Q: 内存不足怎么办？

A: 降低GPU内存使用率或增加系统内存。

## 更多信息

- [Docker官方文档](https://docs.docker.com/)
- [Docker Compose文档](https://docs.docker.com/compose/)
- [NVIDIA Docker文档](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- [vLLM文档](https://docs.vllm.ai/)
