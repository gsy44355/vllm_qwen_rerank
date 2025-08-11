#!/usr/bin/env python3
"""
测试环境变量传递
"""

import os

def print_env_vars():
    """打印关键环境变量"""
    print("=" * 50)
    print("环境变量检查")
    print("=" * 50)
    
    env_vars = [
        'PYTHONUNBUFFERED',
        'CUDA_VISIBLE_DEVICES', 
        'MODEL_PATH',
        'MODEL_SIZE',
        'GPU_MEMORY_UTILIZATION',
        'MAX_MODEL_LEN',
        'VLLM_WORKER_MULTIPROC_METHOD'
    ]
    
    for var in env_vars:
        value = os.getenv(var, '未设置')
        print(f"{var}: {value}")
    
    print("=" * 50)

if __name__ == "__main__":
    print_env_vars()
