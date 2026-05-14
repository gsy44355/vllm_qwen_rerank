#!/usr/bin/env python3
"""
vLLM Rerank 服务使用示例
展示如何使用不同配置的模型
"""

import requests
import json
import time

# 服务地址
BASE_URL = "http://localhost:8000"

def test_health():
    """测试健康检查"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print("✓ 健康检查通过")
            print(f"模型配置: {data.get('model_config', 'N/A')}")
            return True
        else:
            print(f"✗ 健康检查失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ 健康检查异常: {e}")
        return False

def test_rerank():
    """测试 vLLM 风格 rerank 接口（兼容 Jina/Cohere rerank API）"""
    data = {
        "model": "Qwen/Qwen3-Reranker-4B",
        "query": "人工智能",
        "documents": [
            "机器学习是人工智能的一个子领域",
            "深度学习是机器学习的一种方法",
            "自然语言处理是AI的重要分支",
            "计算机视觉处理图像和视频",
        ],
        "top_n": 3,
        "instruction": "判断文档是否与查询相关。答案只能是'yes'或'no'。",
    }

    try:
        response = requests.post(f"{BASE_URL}/v1/rerank", json=data)
        if response.status_code == 200:
            result = response.json()
            print("✓ 重排序测试通过")
            print(f"id={result['id']}, model={result['model']}, "
                  f"total_tokens={result['usage']['total_tokens']}")
            print("重排序结果:")
            for rank, item in enumerate(result["results"], start=1):
                print(
                    f"  排名 {rank}: index={item['index']}  "
                    f"score={item['relevance_score']:.4f}  "
                    f"doc={item['document']['text']}"
                )
            return True
        else:
            print(f"✗ 重排序测试失败: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"✗ 重排序测试异常: {e}")
        return False

def test_reload_model():
    """测试重新加载模型"""
    config = {
        "model_path": "Qwen/Qwen3-Reranker-4B",
        "model_size": "4B",
        "gpu_memory_utilization": 0.8,
        "max_model_len": 10000
    }
    
    try:
        response = requests.post(f"{BASE_URL}/reload_model", json=config)
        if response.status_code == 200:
            result = response.json()
            print("✓ 模型重新加载测试通过")
            print(f"新配置: {result.get('config', 'N/A')}")
            return True
        else:
            print(f"✗ 模型重新加载测试失败: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"✗ 模型重新加载测试异常: {e}")
        return False

def main():
    print("=" * 50)
    print("vLLM Rerank 服务使用示例")
    print("=" * 50)
    
    # 等待服务启动
    print("等待服务启动...")
    time.sleep(5)
    
    # 测试健康检查
    if not test_health():
        print("服务未启动，请先启动服务")
        return
    
    print("\n" + "-" * 30)
    
    # 测试重排序
    test_rerank()
    
    print("\n" + "-" * 30)
    
    print("所有测试完成")

if __name__ == "__main__":
    main()
