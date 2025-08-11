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
    """测试重排序功能"""
    data = {
        "query": "人工智能",
        "documents": [
            "机器学习是人工智能的一个子领域",
            "深度学习是机器学习的一种方法",
            "自然语言处理是AI的重要分支",
            "计算机视觉处理图像和视频"
        ],
        "instruction": "判断文档是否与查询相关。答案只能是'yes'或'no'。"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/rerank", json=data)
        if response.status_code == 200:
            result = response.json()
            print("✓ 重排序测试通过")
            print("重排序结果:")
            for doc in result['ranked_documents']:
                print(f"  排名 {doc['rank']}: {doc['document']} (分数: {doc['score']:.4f})")
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

def test_batch_rerank():
    """测试批量重排序"""
    batch_data = [
        {
            "query": "机器学习",
            "documents": ["深度学习", "神经网络", "支持向量机", "决策树"],
            "instruction": "判断文档是否与查询相关。"
        },
        {
            "query": "自然语言处理",
            "documents": ["文本分类", "机器翻译", "情感分析", "语音识别"],
            "instruction": "判断文档是否与查询相关。"
        }
    ]
    
    try:
        response = requests.post(f"{BASE_URL}/batch_rerank", json=batch_data)
        if response.status_code == 200:
            result = response.json()
            print("✓ 批量重排序测试通过")
            for i, batch_result in enumerate(result['results']):
                print(f"\n批次 {i+1}:")
                for doc in batch_result['ranked_documents']:
                    print(f"  排名 {doc['rank']}: {doc['document']} (分数: {doc['score']:.4f})")
            return True
        else:
            print(f"✗ 批量重排序测试失败: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"✗ 批量重排序测试异常: {e}")
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
    
    # 测试批量重排序
    test_batch_rerank()
    
    print("\n" + "-" * 30)
    
    # 测试重新加载模型
    test_reload_model()
    
    print("\n" + "=" * 50)
    print("所有测试完成")

if __name__ == "__main__":
    main()
