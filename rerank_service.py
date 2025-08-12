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


# 配置日志  
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def generate_stable_request_id(instruction, query, doc_index):
    """生成基于内容的稳定request_id，用于KV缓存"""
    content = f"{instruction}:{query}:{doc_index}"
    return f"rerank_{hashlib.md5(content.encode()).hexdigest()[:16]}"

async def compute_logits_batch(engine, messages, sampling_params, true_token, false_token, instruction, query):
    """优化的批处理版本 - 使用稳定的request_id启用KV缓存"""
    try:
        scores = []
        
        # 为每个消息生成稳定的request_id
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
    """异步初始化模型和tokenizer - 优化版本"""
    global tokenizer, engine, suffix_tokens, true_token, false_token, sampling_params, model_config, executor
    
    # 使用默认配置或传入的配置
    if config is None:
        config = ModelConfig(
            model_path='Qwen/Qwen3-Reranker-4B',
            model_size='4B',
            gpu_memory_utilization=0.85,  # 提高显存利用率
            max_model_len=10000,
            max_num_batched_tokens=8192
        )
    
    model_config = config
    
    logger.info(f"正在初始化模型: {config.model_path}")
    logger.info(f"模型大小: {config.model_size}")
    logger.info(f"GPU内存使用率: {config.gpu_memory_utilization}")
    logger.info(f"最大模型长度: {config.max_model_len}")
    logger.info(f"批处理token数量限制: {config.max_num_batched_tokens}")
    
    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    
    # 初始化 vLLM 异步引擎 - 优化配置，重点启用KV缓存
    engine_args = AsyncEngineArgs(
        model=config.model_path,
        max_model_len=config.max_model_len,
        enable_prefix_caching=True,  # 启用前缀缓存
        gpu_memory_utilization=config.gpu_memory_utilization,
        trust_remote_code=True,
        tensor_parallel_size=1,  # 单GPU
        max_num_batched_tokens=config.max_num_batched_tokens,  # 批处理优化
        max_num_seqs=256,  # 增加并发序列数
        enable_chunked_prefill=True,  # 启用分块预填充
        max_num_blocks_per_seq=256,  # 每个序列的最大block数
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    
    # 设置后缀tokens
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
    
    # 设置true/false tokens
    true_token = tokenizer("yes", add_special_tokens=False).input_ids[0]
    false_token = tokenizer("no", add_special_tokens=False).input_ids[0]
    
    # 设置采样参数
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=1,
        logprobs=20,
        allowed_token_ids=[true_token, false_token],
    )
    
    # 初始化线程池执行器
    executor = ThreadPoolExecutor(max_workers=4)
    
    logger.info("模型初始化完成")

# 应用生命周期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    model_path = os.getenv('MODEL_PATH', 'Qwen/Qwen3-Reranker-4B')
    model_size = os.getenv('MODEL_SIZE', '4B')
    gpu_memory_utilization = float(os.getenv('GPU_MEMORY_UTILIZATION', '0.85'))
    max_model_len = int(os.getenv('MAX_MODEL_LEN', '10000'))
    max_num_batched_tokens = int(os.getenv('MAX_NUM_BATCHED_TOKENS', '8192'))
    
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
    
    # 关闭时清理
    if engine:
        await engine.close()
    if executor:
        executor.shutdown(wait=True)
    # 清理GPU缓存
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
    """重排序文档 - 优化版本"""
    try:
        if not request.documents:
            raise HTTPException(status_code=400, detail="文档列表不能为空")
        
        # 创建查询-文档对
        pairs = [(request.query, doc) for doc in request.documents]
        
        # 处理输入
        inputs = process_inputs(
            pairs, 
            request.instruction, 
            request.max_length - len(suffix_tokens), 
            suffix_tokens
        )
        
        # 根据文档数量选择处理方式
        if len(inputs) > 10:  # 大批量使用批处理
            scores = await compute_logits_batch(engine, inputs, sampling_params, true_token, false_token, request.instruction, request.query)
        else:  # 小批量使用单条处理
            scores = await compute_logits_async(engine, inputs, sampling_params, true_token, false_token, request.instruction, request.query)
        
        # 创建排序后的文档列表
        ranked_docs = [
            {"document": doc, "score": score, "rank": i + 1}
            for i, (doc, score) in enumerate(sorted(zip(request.documents, scores), key=lambda x: x[1], reverse=True))
        ]
        
        return RerankResponse(scores=scores, ranked_documents=ranked_docs)
        
    except Exception as e:
        logger.error(f"重排序过程中发生错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")

@app.post("/batch_rerank")
async def batch_rerank(batch_requests: List[RerankRequest]):
    """批量重排序 - 优化版本，使用真正的批处理"""
    try:
        if not batch_requests:
            raise HTTPException(status_code=400, detail="批量请求不能为空")
        
        # 合并所有请求的文档
        all_pairs = []
        all_documents = []
        all_queries = []
        all_instructions = []
        
        for request in batch_requests:
            if not request.documents:
                continue
            pairs = [(request.query, doc) for doc in request.documents]
            all_pairs.extend(pairs)
            all_documents.extend(request.documents)
            all_queries.extend([request.query] * len(request.documents))
            all_instructions.extend([request.instruction] * len(request.documents))
        
        if not all_pairs:
            raise HTTPException(status_code=400, detail="没有有效的文档")
        
        # 使用第一个请求的max_length作为统一长度
        max_length = batch_requests[0].max_length
        
        # 处理输入 - 批量处理
        inputs = process_inputs(
            all_pairs, 
            all_instructions[0],  # 使用第一个指令
            max_length - len(suffix_tokens), 
            suffix_tokens
        )
        
        # 批量计算scores - 使用KV缓存
        scores = await compute_logits_batch(engine, inputs, sampling_params, true_token, false_token, all_instructions[0], all_queries[0])
        
        # 按原始请求分组结果
        results = []
        start_idx = 0
        for request in batch_requests:
            if not request.documents:
                results.append({"scores": [], "ranked_documents": []})
                continue
                
            end_idx = start_idx + len(request.documents)
            request_scores = scores[start_idx:end_idx]
            
            # 创建排序后的文档列表
            ranked_docs = [
                {"document": doc, "score": score, "rank": i + 1}
                for i, (doc, score) in enumerate(sorted(zip(request.documents, request_scores), key=lambda x: x[1], reverse=True))
            ]
            
            results.append({
                "scores": request_scores,
                "ranked_documents": ranked_docs
            })
            
            start_idx = end_idx
        
        return {"results": results}
        
    except Exception as e:
        logger.error(f"批量重排序过程中发生错误: {str(e)}")
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
