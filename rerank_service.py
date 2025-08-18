# Requires vllm>=0.10.0
import logging
import os
from typing import Dict, Optional, List, Any
import json
import math
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer
from vllm.engine.async_llm_engine import AsyncLLMEngine, AsyncEngineArgs
from vllm import SamplingParams
from vllm.inputs.data import TokensPrompt
import asyncio
from contextlib import asynccontextmanager
import uuid
import traceback
import gc
from concurrent.futures import ThreadPoolExecutor
import time
import hashlib
from collections import deque


# 配置日志  
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


## 删除原先的批量聚合队列与后台任务，避免跨请求共享 instruction


# 请求和响应模型
class RerankRequest(BaseModel):
    query: str
    documents: List[str]
    instruction: str = "判断文档是否满足查询要求。答案只能是'yes'或'no'。"
    max_length: int = 8192

class RerankResponse(BaseModel):
    scores: List[float]
    ranked_documents: List[Dict[str, Any]]

class ModelConfig(BaseModel):
    model_path: str
    model_size: str = "4B"  # 0.6B, 4B, 8B
    gpu_memory_utilization: float = 0.85  # 提高显存利用率
    max_model_len: int = 10000
    max_num_batched_tokens: int = 8192  # 批处理token数量限制

# 全局变量
tokenizer = None
engine = None
suffix_tokens = None
true_token = None
false_token = None
sampling_params = None
model_config = None
executor = None

def format_instruction(instruction, query, doc):
    """格式化指令"""
    text = [
        {"role": "system", "content": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."},
        {"role": "user", "content": f"<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {doc}"}
    ]
    return text

def process_inputs(pairs, instruction, max_length, suffix_tokens):
    """处理输入数据 - 优化版本"""
    messages = [format_instruction(instruction, query, doc) for query, doc in pairs]
    messages = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, enable_thinking=False
    )
    messages = [ele[:max_length] + suffix_tokens for ele in messages]
    messages = [TokensPrompt(prompt_token_ids=ele) for ele in messages]
    return messages

def generate_stable_request_id_from_tokens(tokens: List[int]) -> str:
    """基于 tokens 生成稳定的 request_id，用于KV缓存。
    为避免过长字符串，截取后缀一定长度再hash，保证稳定性且高概率唯一。
    """
    # 仅使用末尾最多 512 个 token 参与hash，既稳定又避免过长
    tail_tokens = tokens[-512:] if len(tokens) > 512 else tokens
    content = ",".join(str(t) for t in tail_tokens)
    return f"rerank_{hashlib.md5(content.encode()).hexdigest()[:16]}"

async def compute_logits_batch(engine, messages, sampling_params, true_token, false_token):
    """并发提交多条请求：每条使用 prompt 与 request_id；由引擎内部自动做批处理。"""
    try:
        async def compute_one(msg) -> float:
            # 生成稳定 request_id（基于 tokens）
            token_ids = getattr(msg, "prompt_token_ids", None)
            if token_ids is None:
                rid = f"rerank_{hashlib.md5(str(msg).encode()).hexdigest()[:16]}"
            else:
                rid = generate_stable_request_id_from_tokens(token_ids)

            async for output in engine.generate(
                prompt=msg,
                sampling_params=sampling_params,
                request_id=rid,
            ):
                final_logits = output.outputs[0].logprobs[-1]

                true_logit = float(final_logits[true_token].logprob) if true_token in final_logits else -10.0
                false_logit = float(final_logits[false_token].logprob) if false_token in final_logits else -10.0

                logits_tensor = torch.tensor([false_logit, true_logit], dtype=torch.float32)
                log_softmax_scores = torch.nn.functional.log_softmax(logits_tensor, dim=0)
                return log_softmax_scores[1].exp().item()

            return 0.0

        tasks = [asyncio.create_task(compute_one(msg)) for msg in messages]
        scores: List[float] = await asyncio.gather(*tasks)
        return scores

    except Exception as e:
        logger.error(f"计算 logits 时出错: {e}")
        raise

async def compute_logits_async(engine, messages, sampling_params, true_token, false_token, instruction, query):
    """单条处理版本 - 用于小批量或单个请求，使用稳定的request_id"""
    try:
        scores = []
        
        for i, message in enumerate(messages):
            request_id = generate_stable_request_id(instruction, query, i)
            async for output in engine.generate(
                prompt=message,
                sampling_params=sampling_params,
                request_id=request_id
            ):
                final_logits = output.outputs[0].logprobs[-1]
                
                # 正确提取 logprob 值
                if true_token in final_logits:
                    true_logit = final_logits[true_token].logprob
                else:
                    true_logit = -10.0
                    
                if false_token in final_logits:
                    false_logit = final_logits[false_token].logprob
                else:
                    false_logit = -10.0
                
                # 确保是 float 类型
                true_logit = float(true_logit)
                false_logit = float(false_logit)
                
                # 使用与官方一致的计算方式
                logits_tensor = torch.tensor([false_logit, true_logit], dtype=torch.float32)
                log_softmax_scores = torch.nn.functional.log_softmax(logits_tensor, dim=0)
                score = log_softmax_scores[1].exp().item()  # true 的概率
                
                scores.append(score)
                break
                
        return scores
        
    except Exception as e:
        logger.error(f"计算 logits 时出错: {e}")
        raise

async def initialize_model(config: ModelConfig = None):
    global tokenizer, engine, suffix_tokens, true_token, false_token, sampling_params, model_config, executor

    if config is None:
        config = ModelConfig(
            model_path='Qwen/Qwen3-Reranker-4B',
            model_size='4B',
            gpu_memory_utilization=0.85,
            max_model_len=10000,
            max_num_batched_tokens=16384  # ✅ 提高批处理token限制
        )

    model_config = config

    # 检查 CUDA MPS 支持
    mps_enabled = os.getenv('CUDA_MPS_ENABLE', '0') == '1'
    if mps_enabled:
        logger.info("CUDA MPS 已启用，将使用多进程安全模式")
        # 设置 CUDA MPS 相关环境变量
        os.environ['CUDA_MPS_ENABLE'] = '1'
        os.environ['VLLM_USE_MPS'] = '1'
        os.environ['VLLM_DISABLE_CUSTOM_ALL_REDUCE'] = '1'
    else:
        logger.info("使用标准 CUDA 模式")

    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    engine_args = AsyncEngineArgs(
        model=config.model_path,
        max_model_len=config.max_model_len,
        enable_prefix_caching=True,
        gpu_memory_utilization=config.gpu_memory_utilization,
        trust_remote_code=True,
        tensor_parallel_size=1,
        max_num_batched_tokens=config.max_num_batched_tokens,
        max_num_seqs=512,  # ✅ 提高并行序列数
        # CUDA MPS 相关配置
        disable_custom_all_reduce=mps_enabled,  # 在 MPS 模式下禁用自定义 all-reduce
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

    true_token = tokenizer("yes", add_special_tokens=False).input_ids[0]
    false_token = tokenizer("no", add_special_tokens=False).input_ids[0]

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=1,
        logprobs=20,
        allowed_token_ids=[true_token, false_token],
    )

    executor = ThreadPoolExecutor(max_workers=4)
    logger.info("模型初始化完成")

# 应用生命周期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = os.getenv('MODEL_PATH', 'Qwen/Qwen3-Reranker-4B')
    model_size = os.getenv('MODEL_SIZE', '4B')
    gpu_memory_utilization = float(os.getenv('GPU_MEMORY_UTILIZATION', '0.85'))
    max_model_len = int(os.getenv('MAX_MODEL_LEN', '10000'))
    max_num_batched_tokens = int(os.getenv('MAX_NUM_BATCHED_TOKENS', '16384'))

    config = ModelConfig(
        model_path=model_path,
        model_size=model_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens
    )

    await initialize_model(config)
    logger.info("服务启动完成")

    yield

    if engine:
        await engine.close()
    if executor:
        executor.shutdown(wait=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    logger.info("服务关闭完成")

# FastAPI应用
app = FastAPI(
    title="Rerank Service (Optimized)", 
    description="优化的单GPU文档重排序服务",
    lifespan=lifespan
)

@app.get("/")
async def root():
    """根路径"""
    return {"message": "Rerank Service is running (Optimized)"}

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy", 
        "model_loaded": engine is not None,
        "model_config": {
            "model_path": model_config.model_path if model_config else None,
            "model_size": model_config.model_size if model_config else None,
            "gpu_memory_utilization": model_config.gpu_memory_utilization if model_config else None,
            "max_model_len": model_config.max_model_len if model_config else None,
            "max_num_batched_tokens": model_config.max_num_batched_tokens if model_config else None
        } if model_config else None
    }

@app.post("/rerank", response_model=RerankResponse)
async def rerank_documents(request: RerankRequest):
    try:
        if not request.documents:
            raise HTTPException(status_code=400, detail="文档列表不能为空")

        # 即时根据该请求的 instruction 处理，不与其他请求合并
        pairs = [(request.query, doc) for doc in request.documents]
        inputs = process_inputs(
            pairs,
            request.instruction,
            request.max_length - len(suffix_tokens),
            suffix_tokens
        )
        scores = await compute_logits_batch(engine, inputs, sampling_params, true_token, false_token)

        # 携带原始索引并排序
        scored_with_index = [
            (idx, doc, score) for idx, (doc, score) in enumerate(zip(request.documents, scores))
        ]
        scored_with_index.sort(key=lambda x: x[2], reverse=True)
        ranked_docs = [
            {"document": doc, "score": score, "rank": rank_idx + 1, "index": idx}
            for rank_idx, (idx, doc, score) in enumerate(scored_with_index)
        ]

        return RerankResponse(scores=scores, ranked_documents=ranked_docs)

    except Exception as e:
        logger.error(f"重排序过程中发生错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")


@app.post("/reload_model")
async def reload_model(config: ModelConfig):
    """重新加载模型 - 优化版本"""
    try:
        logger.info("开始重新加载模型...")
        
        # 关闭旧引擎和线程池
        if engine:
            await engine.close()
        if executor:
            executor.shutdown(wait=True)
        
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # 初始化新模型
        await initialize_model(config)
        logger.info("模型重新加载完成")
        return {"message": "模型重新加载成功", "config": config.dict()}
    except Exception as e:
        logger.error(f"重新加载模型时发生错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"重新加载模型失败: {str(e)}")

@app.post("/clear_cache")
async def clear_cache():
    """清理GPU缓存"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        return {"message": "GPU缓存清理完成"}
    except Exception as e:
        logger.error(f"清理缓存时发生错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"清理缓存失败: {str(e)}")

if __name__ == "__main__":
    # 启动服务
    uvicorn.run(
        "rerank_service:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
