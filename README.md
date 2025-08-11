# vLLM Rerank 服务

基于vLLM和Qwen3-Reranker-4B模型的文档重排序服务，提供RESTful API接口。

## 功能特性

- 基于vLLM的高性能推理
- **异步优化**：使用线程池避免阻塞事件循环
- 支持单次和批量文档重排序
- RESTful API接口
- 健康检查功能
- 自动模型初始化

## 安装依赖

```bash
pip install -r requirements.txt
```

## 启动服务

### 方式一：直接运行

```bash
python rerank_service.py
```

### 方式二：使用启动脚本

```bash
# 使用默认配置
python start_service.py

# 指定模型配置
python start_service.py \
    --model-path /path/to/local/model \
    --model-size 4B \
    --gpu-memory-utilization 0.6 \
    --max-model-len 8000
```

### 方式三：Docker部署（推荐）

#### 快速部署

```bash
# 给部署脚本执行权限
chmod +x docker-deploy.sh

# 一键部署（使用默认配置）
./docker-deploy.sh deploy

# 使用本地模型部署
MODEL_PATH=/models/qwen-rerank-4b MODEL_SIZE=4B ./docker-deploy.sh deploy

# 降低GPU内存使用率
GPU_MEMORY_UTILIZATION=0.6 ./docker-deploy.sh deploy
```

#### 分步部署

```bash
# 构建镜像
./docker-deploy.sh build

# 启动服务
./docker-deploy.sh start

# 查看状态
./docker-deploy.sh status

# 查看日志
./docker-deploy.sh logs
```

#### 使用docker-compose

```bash
# 构建并启动（使用默认配置）
docker-compose up -d

# 使用环境变量配置
MODEL_PATH=/models/qwen-rerank-4b MODEL_SIZE=4B docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

服务将在 `http://localhost:8000` 启动。

## 模型配置

### 支持的模型

- **0.6B**: Qwen/Qwen3-Reranker-0.6B - 适合资源受限环境
- **4B**: Qwen/Qwen3-Reranker-4B - 平衡性能和资源消耗（默认）
- **8B**: Qwen/Qwen3-Reranker-8B - 最高性能

### 下载模型

```bash
# 下载指定大小的模型
python download_models.py --model-size 4B --output-dir ./models

# 下载所有支持的模型
python download_models.py --model-size 0.6B --output-dir ./models
python download_models.py --model-size 4B --output-dir ./models
python download_models.py --model-size 8B --output-dir ./models
```

### 配置参数

| 参数 | 说明 | 默认值 | 范围 |
|------|------|--------|------|
| MODEL_PATH | 模型路径 | Qwen/Qwen3-Reranker-4B | 本地路径或HuggingFace模型名 |
| MODEL_SIZE | 模型大小 | 4B | 0.6B, 4B, 8B |
| GPU_MEMORY_UTILIZATION | GPU内存使用率 | 0.8 | 0.1-1.0 |
| MAX_MODEL_LEN | 最大模型长度 | 10000 | 1000-32000 |

## API接口

### 1. 健康检查

**GET** `/health`

检查服务状态和模型加载情况。

**响应示例：**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### 2. 文档重排序

**POST** `/rerank`

对文档列表进行重排序。

**请求参数：**
```json
{
  "query": "查询文本",
  "documents": ["文档1", "文档2", "文档3"],
  "instruction": "判断文档是否满足查询要求。答案只能是'yes'或'no'。",
  "max_length": 8192
}
```

**响应示例：**
```json
{
  "scores": [0.85, 0.72, 0.45],
  "ranked_documents": [
    {
      "document": "文档1",
      "score": 0.85,
      "rank": 1
    },
    {
      "document": "文档2", 
      "score": 0.72,
      "rank": 2
    },
    {
      "document": "文档3",
      "score": 0.45,
      "rank": 3
    }
  ]
}
```

### 3. 批量重排序

**POST** `/batch_rerank`

批量处理多个重排序请求。

### 4. 重新加载模型

**POST** `/reload_model`

重新加载模型配置。

**请求参数：**
```json
{
  "model_path": "/path/to/model",
  "model_size": "4B",
  "gpu_memory_utilization": 0.8,
  "max_model_len": 10000
}
```

**请求参数：**
```json
[
  {
    "query": "查询1",
    "documents": ["文档1", "文档2"],
    "instruction": "指令1"
  },
  {
    "query": "查询2", 
    "documents": ["文档3", "文档4"],
    "instruction": "指令2"
  }
]
```

## 测试服务

### 功能测试

运行测试脚本验证服务功能：

```bash
python test_service.py
```

### 性能测试

运行性能测试脚本评估并发性能：

```bash
python performance_test.py
```

性能测试会模拟多个并发请求，测试服务的响应时间和吞吐量。

## 使用示例

### Python客户端示例

```python
import requests

# 单次重排序
data = {
    "query": "暴力",
    "documents": ["杀人", "放火", "亲吻", "睡觉"],
    "instruction": "给出一个词，判断候选列表的词和给出词语最相近的"
}

response = requests.post("http://localhost:8000/rerank", json=data)
result = response.json()

print("重排序结果:")
for doc in result['ranked_documents']:
    print(f"排名 {doc['rank']}: {doc['document']} (分数: {doc['score']:.4f})")
```

### cURL示例

```bash
curl -X POST "http://localhost:8000/rerank" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "暴力",
       "documents": ["杀人", "放火", "亲吻", "睡觉"],
       "instruction": "给出一个词，判断候选列表的词和给出词语最相近的"
     }'
```

## 配置说明

### 模型配置

- **模型**: 可通过环境变量配置
- **tensor_parallel_size**: 1 (不使用GPU并行)
- **max_model_len**: 可通过环境变量配置
- **gpu_memory_utilization**: 可通过环境变量配置

### 服务配置

- **主机**: 0.0.0.0
- **端口**: 8000
- **日志级别**: info

## 注意事项

1. 首次启动时会下载模型，需要较长时间
2. 确保有足够的GPU内存运行模型
3. 建议在生产环境中使用Docker部署
4. Docker部署需要安装Docker和Docker Compose
5. 如需GPU支持，请安装NVIDIA Docker运行时

## 故障排除

### 常见问题

1. **模型下载失败**
   - 检查网络连接
   - 确保有足够的磁盘空间

2. **GPU内存不足**
   - 降低 `gpu_memory_utilization` 参数
   - 使用更小的模型

3. **服务启动失败**
   - 检查端口是否被占用
   - 查看日志输出

4. **Docker部署问题**
   - 确保Docker和Docker Compose已安装
   - 检查GPU驱动和NVIDIA Docker运行时
   - 查看容器日志：`docker-compose logs`

## 许可证

本项目基于MIT许可证开源。
