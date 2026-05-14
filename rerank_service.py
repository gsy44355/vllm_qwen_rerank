# Requires vllm>=0.10.0
import logging
import os
import sys
from typing import Dict, Optional, List, Any, Literal, Union
import json
import math
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
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
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)


## 删除原先的批量聚合队列与后台任务，避免跨请求共享 instruction


def _truncate_text(text: str, max_len: int = 200) -> str:
    """日志打印时截断长文本，避免刷屏。"""
    if len(text) <= max_len:
        return text
    return f"{text[:max_len]}...(len={len(text)})"


def _get_prompt_token_ids(prompt) -> List[int]:
    """从 TokensPrompt 中安全提取 prompt_token_ids。

    vLLM 中 TokensPrompt 通常是 TypedDict（dict），少数版本是对象，
    这里同时兼容两种形态，且任何异常都退化为空列表，避免影响主流程。
    """
    if prompt is None:
        return []
    if isinstance(prompt, dict):
        return prompt.get("prompt_token_ids") or []
    return getattr(prompt, "prompt_token_ids", None) or []


# 默认指令（Qwen3-Reranker 训练所用 prompt 模板）
DEFAULT_INSTRUCTION = "判断文档是否满足查询要求。答案只能是'yes'或'no'。"


# ===== vLLM 风格 Rerank API 数据模型 =====
# 参考: https://docs.vllm.ai/en/latest/models/pooling_models/scoring/#rerank-api
# 兼容 Jina AI / Cohere v1 & v2 rerank 接口

class RerankRequest(BaseModel):
    """vLLM 风格的 rerank 请求体。"""

    model: Optional[str] = None
    query: str
    documents: List[str]
    top_n: int = Field(
        default=0,
        description="返回前 N 条结果；0 或不传时返回全部文档（与 vLLM 一致）。",
    )
    truncate_prompt_tokens: Optional[int] = Field(
        default=None,
        description="对拼接后的 query+document prompt 截断到该 token 数；None 表示不截断。",
    )
    truncation_side: Optional[Literal["left", "right"]] = Field(
        default=None,
        description="truncate_prompt_tokens 生效时的截断方向。",
    )
    max_tokens_per_query: int = Field(
        default=0,
        description="单条 query 的最大 token 数（0 表示不限制）。",
    )
    max_tokens_per_doc: int = Field(
        default=0,
        description="单条 document 的最大 token 数（0 表示不限制）。",
    )
    instruction: Optional[str] = Field(
        default=None,
        description="拼接到 prompt 中的任务指令；不传则使用服务端默认指令。",
    )
    chat_template_kwargs: Optional[Dict[str, Any]] = Field(
        default=None,
        description="额外的 chat template 参数；其中 instruction 字段等价于顶层 instruction。",
    )
    user: Optional[str] = None
    request_id: Optional[str] = None
    priority: int = 0
    use_activation: Optional[bool] = None


class RerankDocument(BaseModel):
    """vLLM rerank 响应中的 document 对象。"""

    text: str


class RerankResult(BaseModel):
    """vLLM rerank 响应中的单条结果。"""

    index: int
    document: RerankDocument
    relevance_score: float


class RerankUsage(BaseModel):
    total_tokens: int = 0


class RerankResponse(BaseModel):
    """vLLM 风格的 rerank 响应体。"""

    id: str
    model: str
    usage: RerankUsage
    results: List[RerankResult]

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
    logger.info("messages", messages)
    messages = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, enable_thinking=False
    )
    logger.info(f"messages2: {messages}")
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
            # if token_ids is None:
            #     rid = f"rerank_{hashlib.md5(str(msg).encode()).hexdigest()[:16]}"
            # else:
            #     rid = generate_stable_request_id_from_tokens(token_ids)
            rid = str(uuid.uuid4())

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

async def _do_rerank(request: RerankRequest) -> RerankResponse:
    """vLLM 风格 rerank 的核心实现，被 /rerank、/v1/rerank、/v2/rerank 共用。"""

    if not request.documents:
        raise HTTPException(status_code=400, detail="documents 列表不能为空")

    if request.top_n < 0:
        raise HTTPException(status_code=400, detail="top_n 必须 >= 0")

    if request.truncate_prompt_tokens is not None and request.truncate_prompt_tokens <= 0:
        raise HTTPException(
            status_code=400, detail="truncate_prompt_tokens 必须 > 0"
        )

    # top_n=0 或未传时返回全部
    top_n = request.top_n if request.top_n > 0 else len(request.documents)
    top_n = min(top_n, len(request.documents))

    # 解析 instruction：顶层 instruction > chat_template_kwargs.instruction > 默认值
    instruction = request.instruction
    instruction_source = "body.instruction"
    if instruction is None and request.chat_template_kwargs:
        ct_instruction = request.chat_template_kwargs.get("instruction")
        if ct_instruction:
            instruction = ct_instruction
            instruction_source = "body.chat_template_kwargs.instruction"
    if instruction is None:
        instruction = DEFAULT_INSTRUCTION
        instruction_source = "default"

    # 解析 prompt 截断长度
    if request.truncate_prompt_tokens is not None:
        max_prompt_len = request.truncate_prompt_tokens
    else:
        max_prompt_len = model_config.max_model_len if model_config else 8192
    max_pair_len = max(1, max_prompt_len - len(suffix_tokens))

    served_model = request.model or (
        model_config.model_path if model_config else "rerank-model"
    )

    logger.info(
        "收到 rerank 请求: model=%s, query=%s, documents_count=%d, top_n=%d, "
        "truncate_prompt_tokens=%s, instruction_source=%s, instruction=%s, document_preview=%s",
        served_model,
        _truncate_text(request.query),
        len(request.documents),
        top_n,
        request.truncate_prompt_tokens,
        instruction_source,
        _truncate_text(instruction),
        [_truncate_text(doc, 120) for doc in request.documents[:3]],
    )

    # 可选的 query / doc 维度截断（按 token 截断）
    query_text = request.query
    if request.max_tokens_per_query and request.max_tokens_per_query > 0:
        q_ids = tokenizer.encode(query_text, add_special_tokens=False)
        if len(q_ids) > request.max_tokens_per_query:
            q_ids = q_ids[: request.max_tokens_per_query]
            query_text = tokenizer.decode(q_ids, skip_special_tokens=True)

    documents = request.documents
    if request.max_tokens_per_doc and request.max_tokens_per_doc > 0:
        truncated_docs: List[str] = []
        for doc in documents:
            d_ids = tokenizer.encode(doc, add_special_tokens=False)
            if len(d_ids) > request.max_tokens_per_doc:
                d_ids = d_ids[: request.max_tokens_per_doc]
                truncated_docs.append(tokenizer.decode(d_ids, skip_special_tokens=True))
            else:
                truncated_docs.append(doc)
        documents = truncated_docs

    pairs = [(query_text, doc) for doc in documents]
    inputs = process_inputs(pairs, instruction, max_pair_len, suffix_tokens)

    # 统计 prompt token 数用于 usage.total_tokens（兼容 dict 与对象两种形态）
    total_tokens = sum(len(_get_prompt_token_ids(p)) for p in inputs)

    scores = await compute_logits_batch(
        engine, inputs, sampling_params, true_token, false_token
    )

    scored_with_index = [
        (idx, doc, score)
        for idx, (doc, score) in enumerate(zip(request.documents, scores))
    ]
    scored_with_index.sort(key=lambda x: x[2], reverse=True)

    results = [
        RerankResult(
            index=idx,
            document=RerankDocument(text=doc),
            relevance_score=score,
        )
        for idx, doc, score in scored_with_index[:top_n]
    ]

    response_id = request.request_id or f"rerank-{uuid.uuid4().hex}"

    return RerankResponse(
        id=response_id,
        model=served_model,
        usage=RerankUsage(total_tokens=total_tokens),
        results=results,
    )


@app.post("/rerank", response_model=RerankResponse)
@app.post("/v1/rerank", response_model=RerankResponse)
@app.post("/v2/rerank", response_model=RerankResponse)
async def rerank_documents(request: RerankRequest):
    """vLLM 风格的 rerank 接口，兼容 Jina AI / Cohere v1 & v2 rerank API。"""
    try:
        return await _do_rerank(request)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"重排序过程中发生错误: {str(e)}")
        logger.error(traceback.format_exc())
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
