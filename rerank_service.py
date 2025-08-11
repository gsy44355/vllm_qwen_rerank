# Requires vllm>=0.8.5
import logging
import os
from typing import Dict, Optional, List
import json
import math
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs.data import TokensPrompt
import asyncio
from concurrent.futures import ThreadPoolExecutor

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI应用
app = FastAPI(title="Rerank Service", description="基于vLLM的文档重排序服务")

# 请求和响应模型
class RerankRequest(BaseModel):
    query: str
    documents: List[str]
    instruction: str = "判断文档是否满足查询要求。答案只能是'yes'或'no'。"
    max_length: int = 8192

class RerankResponse(BaseModel):
    scores: List[float]
    ranked_documents: List[Dict[str, any]]

class ModelConfig(BaseModel):
    model_path: str
    model_size: str = "4B"  # 0.6B, 4B, 8B
    gpu_memory_utilization: float = 0.8
    max_model_len: int = 10000

# 全局变量
tokenizer = None
model = None
suffix_tokens = None
true_token = None
false_token = None
sampling_params = None
model_config = None
# 线程池执行器，用于处理同步的model.generate调用
executor = ThreadPoolExecutor(max_workers=4)

def format_instruction(instruction, query, doc):
    """格式化指令"""
    text = [
        {"role": "system", "content": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."},
        {"role": "user", "content": f"<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {doc}"}
    ]
    return text

def process_inputs(pairs, instruction, max_length, suffix_tokens):
    """处理输入数据"""
    messages = [format_instruction(instruction, query, doc) for query, doc in pairs]
    messages = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, enable_thinking=False
    )
    messages = [ele[:max_length] + suffix_tokens for ele in messages]
    messages = [TokensPrompt(prompt_token_ids=ele) for ele in messages]
    return messages

async def compute_logits(model, messages, sampling_params, true_token, false_token):
    """计算logits并返回分数"""
    # 使用线程池执行同步的model.generate，避免阻塞事件循环
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    
    def _generate_sync():
        return model.generate(messages, sampling_params, use_tqdm=False)
    
    # 在线程池中执行同步调用
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        outputs = await loop.run_in_executor(executor, _generate_sync)
    
    scores = []
    for i in range(len(outputs)):
        final_logits = outputs[i].outputs[0].logprobs[-1]
        if true_token not in final_logits:
            true_logit = -10
        else:
            true_logit = final_logits[true_token].logprob
        if false_token not in final_logits:
            false_logit = -10
        else:
            false_logit = final_logits[false_token].logprob
        true_score = math.exp(true_logit)
        false_score = math.exp(false_logit)
        score = true_score / (true_score + false_score)
        scores.append(score)
    return scores

def initialize_model(config: ModelConfig = None):
    """初始化模型和tokenizer"""
    global tokenizer, model, suffix_tokens, true_token, false_token, sampling_params, model_config
    
    # 使用默认配置或传入的配置
    if config is None:
        config = ModelConfig(
            model_path='Qwen/Qwen3-Reranker-4B',
            model_size='4B',
            gpu_memory_utilization=0.8,
            max_model_len=10000
        )
    
    model_config = config
    
    logger.info(f"正在初始化模型: {config.model_path}")
    logger.info(f"模型大小: {config.model_size}")
    logger.info(f"GPU内存使用率: {config.gpu_memory_utilization}")
    logger.info(f"最大模型长度: {config.max_model_len}")
    
    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    
    # 初始化模型（删除GPU并行处理逻辑）
    model = LLM(
        model=config.model_path, 
        tensor_parallel_size=1,  # 设置为1，不使用GPU并行
        max_model_len=config.max_model_len, 
        enable_prefix_caching=True, 
        gpu_memory_utilization=config.gpu_memory_utilization
    )
    
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
    
    logger.info("模型初始化完成")

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化模型"""
    # 从环境变量读取配置
    model_path = os.getenv('MODEL_PATH', 'Qwen/Qwen3-Reranker-4B')
    model_size = os.getenv('MODEL_SIZE', '4B')
    gpu_memory_utilization = float(os.getenv('GPU_MEMORY_UTILIZATION', '0.8'))
    max_model_len = int(os.getenv('MAX_MODEL_LEN', '10000'))
    
    config = ModelConfig(
        model_path=model_path,
        model_size=model_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len
    )
    
    initialize_model(config)

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时清理资源"""
    global executor
    executor.shutdown(wait=True)
    logger.info("线程池已关闭")

@app.get("/")
async def root():
    """根路径"""
    return {"message": "Rerank Service is running"}

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy", 
        "model_loaded": model is not None,
        "model_config": {
            "model_path": model_config.model_path if model_config else None,
            "model_size": model_config.model_size if model_config else None,
            "gpu_memory_utilization": model_config.gpu_memory_utilization if model_config else None,
            "max_model_len": model_config.max_model_len if model_config else None
        } if model_config else None
    }

@app.post("/rerank", response_model=RerankResponse)
async def rerank_documents(request: RerankRequest):
    """重排序文档"""
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
        
        # 计算分数
        scores = await compute_logits(model, inputs, sampling_params, true_token, false_token)
        
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
    """批量重排序"""
    try:
        results = []
        for request in batch_requests:
            result = await rerank_documents(request)
            results.append(result)
        return {"results": results}
    except Exception as e:
        logger.error(f"批量重排序过程中发生错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")

@app.post("/reload_model")
async def reload_model(config: ModelConfig):
    """重新加载模型"""
    try:
        logger.info("开始重新加载模型...")
        initialize_model(config)
        logger.info("模型重新加载完成")
        return {"message": "模型重新加载成功", "config": config.dict()}
    except Exception as e:
        logger.error(f"重新加载模型时发生错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"重新加载模型失败: {str(e)}")

if __name__ == "__main__":
    # 启动服务
    uvicorn.run(
        "rerank_service:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
