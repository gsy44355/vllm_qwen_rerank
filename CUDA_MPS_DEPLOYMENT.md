# CUDA MPS 多进程部署指南

本文档说明如何使用 CUDA MPS (Multi-Process Service) 来支持 vLLM Rerank 服务的多进程部署，解决 CUDA 上下文冲突问题。

## 背景

在使用 Docker Compose 部署 vLLM 服务时，如果使用 Gunicorn 多进程模式，会出现 CUDA 上下文冲突的问题。这是因为多个进程同时尝试初始化 CUDA 上下文导致的。

CUDA MPS 提供了一个解决方案，它允许多个进程共享同一个 GPU 上下文，从而避免冲突。

## 系统要求

- NVIDIA GPU 驱动 >= 470.0
- CUDA 工具包 >= 11.0
- Docker 和 Docker Compose
- 支持 CUDA 的容器运行时

## 部署步骤

### 1. 在宿主机上启动 CUDA MPS 服务

首先，在宿主机上启动 CUDA MPS 服务：

```bash
# 给脚本添加执行权限
chmod +x start_cuda_mps.sh

# 启动 MPS 服务
./start_cuda_mps.sh start

# 检查状态
./start_cuda_mps.sh status
```

### 2. 使用 Docker Compose 部署

使用修改后的 `docker-compose.yml` 配置文件：

```bash
# 构建并启动服务
docker-compose up --build

# 或者在后台运行
docker-compose up -d --build
```

### 3. 验证部署

检查服务是否正常运行：

```bash
# 检查容器状态
docker-compose ps

# 查看日志
docker-compose logs -f rerank-service

# 测试健康检查
curl http://localhost:8888/health
```

## 配置说明

### Docker Compose 配置

主要的配置变更包括：

1. **CUDA MPS 环境变量**：
   ```yaml
   CUDA_MPS_PIPE_DIRECTORY: /tmp/nvidia-mps
   CUDA_MPS_LOG_DIRECTORY: /tmp/nvidia-log
   CUDA_MPS_ENABLE: 1
   ```

2. **vLLM 多进程配置**：
   ```yaml
   VLLM_USE_MPS: 1
   VLLM_DISABLE_CUSTOM_ALL_REDUCE: 1
   ```

3. **目录挂载**：
   ```yaml
   volumes:
     - /tmp/nvidia-mps:/tmp/nvidia-mps
     - /tmp/nvidia-log:/tmp/nvidia-log
   ```

4. **多进程支持**：
   ```yaml
   WORKERS: 2  # 启用多进程
   ```

### Gunicorn 配置

`gun.py` 的主要变更：

1. **禁用预加载**：`preload_app = False`
   - 每个工作进程独立加载模型
   - 避免 CUDA 上下文冲突
   - 支持 CUDA MPS 多进程模式

2. **禁用请求限制**：`max_requests = 0`
   - 避免频繁重启工作进程
   - 保持模型在内存中

3. **增加工作连接数**：`worker_connections = 1000`
   - 提高并发处理能力

### 应用配置

`rerank_service.py` 的主要变更：

1. **CUDA MPS 检测和配置**
   - 自动检测 MPS 环境变量
   - 配置 vLLM 引擎参数

2. **禁用自定义 all-reduce**：`disable_custom_all_reduce=mps_enabled`
   - 在 MPS 模式下禁用自定义通信
   - 使用 MPS 提供的通信机制

3. **模型加载时机**
   - 在 FastAPI 的 `lifespan` 事件中加载
   - 每个工作进程独立初始化模型
   - 服务启动时加载，服务关闭时清理

## 模型加载机制详解

### 预加载 vs 非预加载模式

#### 预加载模式 (`preload_app = True`)
```
主进程启动
    ↓
加载模型到主进程内存
    ↓
fork() 创建多个工作进程
    ↓
所有工作进程共享同一份模型（copy-on-write）
    ↓
每个工作进程处理请求
```

**优点：**
- 模型只加载一次，节省内存
- 启动速度快

**缺点：**
- CUDA 上下文冲突（多个进程共享 GPU 上下文）
- 模型更新需要重启所有进程

#### 非预加载模式 (`preload_app = False`)
```
主进程启动
    ↓
fork() 创建多个工作进程
    ↓
每个工作进程独立执行 lifespan 事件
    ↓
每个工作进程独立加载模型
    ↓
每个工作进程处理请求
```

**优点：**
- 避免 CUDA 上下文冲突
- 每个进程有独立的 GPU 上下文
- 支持 CUDA MPS

**缺点：**
- 每个进程都加载模型，占用更多内存
- 启动时间较长

### 模型加载时机

在 FastAPI 应用中，模型通过 `lifespan` 事件管理器加载：

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 服务启动时执行
    await initialize_model(config)  # 模型在这里加载
    logger.info("服务启动完成")

    yield  # 服务运行期间

    # 服务关闭时执行
    if engine:
        await engine.close()
```

- **启动阶段**：每个工作进程启动时都会执行 `lifespan` 事件
- **运行阶段**：模型保持在内存中，处理请求
- **关闭阶段**：清理模型资源，释放内存

## 性能优化建议

### 1. 工作进程数配置

根据 GPU 内存大小调整工作进程数：

- **0.6B 模型**：2-4 个进程
- **4B 模型**：1-2 个进程  
- **8B 模型**：1 个进程

### 2. GPU 内存利用率

根据模型大小调整 `GPU_MEMORY_UTILIZATION`：

- **0.6B 模型**：0.7-0.8
- **4B 模型**：0.6-0.7
- **8B 模型**：0.5-0.6

### 3. 批处理配置

调整批处理参数以优化性能：

```yaml
MAX_NUM_BATCHED_TOKENS: 16384
MAX_MODEL_LEN: 32000
```

## 故障排除

### 1. MPS 服务启动失败

```bash
# 检查 NVIDIA 驱动
nvidia-smi

# 检查 CUDA 工具包
nvidia-cuda-mps-control --version

# 重新启动 MPS 服务
./start_cuda_mps.sh restart
```

### 2. 容器启动失败

```bash
# 查看详细日志
docker-compose logs rerank-service

# 检查 MPS 目录权限
ls -la /tmp/nvidia-mps
ls -la /tmp/nvidia-log
```

### 3. CUDA 上下文错误

如果仍然出现 CUDA 上下文错误：

1. 确保 MPS 服务正在运行
2. 检查环境变量是否正确设置
3. 尝试减少工作进程数
4. 重启容器和 MPS 服务

### 4. 性能问题

如果性能不理想：

1. 检查 GPU 利用率：`nvidia-smi`
2. 调整批处理参数
3. 监控内存使用情况
4. 考虑使用单进程模式

## 监控和维护

### 1. 监控 MPS 状态

```bash
# 查看 MPS 状态
./start_cuda_mps.sh status

# 查看 GPU 使用情况
nvidia-smi
```

### 2. 日志监控

```bash
# 查看应用日志
docker-compose logs -f rerank-service

# 查看 MPS 日志
tail -f /tmp/nvidia-log/mps.log
```

### 3. 定期维护

```bash
# 重启 MPS 服务（建议定期执行）
./start_cuda_mps.sh restart

# 清理日志文件
docker-compose exec rerank-service find /app/logs -name "*.log" -mtime +7 -delete
```

## 安全注意事项

1. **目录权限**：确保 MPS 目录有适当的权限设置
2. **资源限制**：监控 GPU 内存使用，避免过度分配
3. **网络安全**：在生产环境中配置适当的防火墙规则
4. **日志管理**：定期清理日志文件，避免磁盘空间不足

## 回退方案

如果 CUDA MPS 配置出现问题，可以快速回退到单进程模式：

1. 修改 `docker-compose.yml`：
   ```yaml
   WORKERS: 1
   CUDA_MPS_ENABLE: 0
   ```

2. 重新部署：
   ```bash
   docker-compose down
   docker-compose up --build
   ```

## 总结

使用 CUDA MPS 可以有效地解决 vLLM 多进程部署中的 CUDA 上下文冲突问题，提高服务的并发处理能力。通过正确的配置和监控，可以实现稳定高效的多进程 GPU 推理服务。
