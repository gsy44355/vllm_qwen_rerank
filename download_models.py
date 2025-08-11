#!/usr/bin/env python3
"""
Qwen Rerank 模型下载脚本
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SUPPORTED_MODELS = {
    "0.6B": "Qwen/Qwen3-Reranker-0.6B",
    "4B": "Qwen/Qwen3-Reranker-4B", 
    "8B": "Qwen/Qwen3-Reranker-8B"
}

def download_model(model_size: str, output_dir: str, use_auth_token: str = None):
    """下载指定大小的模型"""
    if model_size not in SUPPORTED_MODELS:
        raise ValueError(f"不支持的模型大小: {model_size}")
    
    model_name = SUPPORTED_MODELS[model_size]
    output_path = Path(output_dir) / model_size
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"下载模型: {model_name}")
    
    # 下载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=output_path,
        use_auth_token=use_auth_token,
        trust_remote_code=True
    )
    tokenizer.save_pretrained(output_path)
    
    # 下载模型配置
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=output_path,
        use_auth_token=use_auth_token,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.save_pretrained(output_path)
    
    return str(output_path)

def main():
    parser = argparse.ArgumentParser(description="下载Qwen Rerank模型")
    parser.add_argument("--model-size", choices=list(SUPPORTED_MODELS.keys()), 
                       help="要下载的模型大小")
    parser.add_argument("--output-dir", default="./models", 
                       help="模型输出目录")
    parser.add_argument("--auth-token", help="HuggingFace认证token")
    
    args = parser.parse_args()
    
    if not args.model_size:
        parser.error("请指定 --model-size")
    
    try:
        model_path = download_model(args.model_size, args.output_dir, args.auth_token)
        print(f"✓ 模型下载完成: {model_path}")
    except Exception as e:
        logger.error(f"下载失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
