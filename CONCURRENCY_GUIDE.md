# vLLM Rerank 服务并发配置指南

## 概述

本服务支持多种并发优化方式，可以根据不同的使用场景和硬件配置进行调优。

## 并发配置选项

### 1. FastAPI Workers（进程级并行）

```bash
# 启动2个FastAPI工作进程
python start_service.py --workers 2

# 使用环境变量
WORKERS=2 python start_service.py
```

**适用场景**：
- 多核CPU环境
- 需要处理大量HTTP请求
- 单GPU但需要进程级隔离

### 2. vLLM内部并发优化

```bash
# 高并发配置
python start_service.py \
    --max-num-seqs 1024 \
    --max-num-batched-tokens 32768 \
    --gpu-memory-utilization 0.9

# 低资源配置
python start_service.py \
    --max-num-seqs 128 \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.6
```

### 3. 张量并行（多GPU）

```bash
# 使用2个GPU进行张量并行
python start_service.py --tensor-parallel-size 2

# 检查可用GPU数量
python -c "import torch; print(f'GPU数量: {torch.cuda.device_count()}')"
```

### 4. 线程池优化

```bash
# 增加线程池工作线程数
python start_service.py --max-workers 8
```

## 推荐配置

### 高并发生产环境

```bash
python start_service.py \
    --workers 2 \
    --max-num-seqs 1024 \
    --max-num-batched-tokens 32768 \
    --gpu-memory-utilization 0.9 \
    --max-workers 8 \
    --model-size 4B
```

### 资源受限环境

```bash
python start_service.py \
    --workers 1 \
    --max-num-seqs 256 \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.6 \
    --max-workers 2 \
    --model-size 0.6B
```

### 高性能环境（多GPU）

```bash
python start_service.py \
    --workers 4 \
    --tensor-parallel-size 2 \
    --max-num-seqs 2048 \
    --max-num-batched-tokens 65536 \
    --gpu-memory-utilization 0.95 \
    --max-workers 16 \
    --model-size 8B
```

## 性能监控

### 健康检查

```bash
curl http://localhost:8000/health
```

返回的配置信息包括：
- `max_num_seqs`: 当前并发序列数
- `tensor_parallel_size`: 张量并行大小
- `max_workers`: 线程池工作线程数

### 性能测试

```bash
# 使用官方示例进行性能测试
python official_example.py

# 批量测试
python example_usage.py
```

## 注意事项

### GPU内存管理

1. **显存使用率**：
   - 0.6-0.7：保守配置，适合稳定运行
   - 0.8-0.9：平衡配置，推荐生产环境
   - 0.9+：激进配置，需要监控OOM风险

2. **并发序列数**：
   - 每个序列占用一定显存
   - 建议根据GPU显存大小调整
   - 24GB显存可支持512-1024序列

### 模型大小影响

| 模型大小 | 显存需求 | 推荐并发序列数 | 适用场景 |
|---------|---------|---------------|---------|
| 0.6B    | 2-4GB   | 256-512       | 资源受限 |
| 4B      | 8-12GB  | 512-1024      | 生产环境 |
| 8B      | 16-24GB | 1024-2048     | 高性能   |

### 最佳实践

1. **渐进式调优**：
   - 从保守配置开始
   - 逐步增加并发参数
   - 监控GPU使用率和响应时间

2. **监控指标**：
   - GPU显存使用率
   - 请求响应时间
   - 并发请求数
   - 错误率

3. **故障处理**：
   - 如果出现OOM，降低`gpu_memory_utilization`
   - 如果响应慢，增加`max_num_seqs`
   - 如果CPU瓶颈，增加`workers`

## 环境变量配置

```bash
# 生产环境配置
export WORKERS=2
export MAX_NUM_SEQS=1024
export MAX_NUM_BATCHED_TOKENS=32768
export GPU_MEMORY_UTILIZATION=0.9
export MAX_WORKERS=8
export TENSOR_PARALLEL_SIZE=1

# 启动服务
python start_service.py
```

## Docker部署

```bash
# 高并发Docker配置
docker run -d \
  -p 8000:8000 \
  --gpus all \
  -e WORKERS=2 \
  -e MAX_NUM_SEQS=1024 \
  -e GPU_MEMORY_UTILIZATION=0.9 \
  -e MAX_WORKERS=8 \
  vllm-rerank:latest
```
