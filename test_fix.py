#!/usr/bin/env python3
"""
测试修复后的rerank服务
"""

import requests
import time

def test_simple_rerank():
    """测试简单的重排序功能"""
    data = {
        "query": "机器学习",
        "documents": [
            "机器学习是人工智能的一个子领域",
            "深度学习是机器学习的一种方法",
            "自然语言处理是AI的重要分支"
        ],
        "instruction": "判断文档是否与查询相关。答案只能是'yes'或'no'。"
    }
    
    try:
        print("测试单个重排序...")
        start_time = time.time()
        response = requests.post("http://localhost:8000/rerank", json=data)
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ 成功! 处理时间: {end_time - start_time:.3f}秒")
            print("重排序结果:")
            for doc in result['ranked_documents']:
                print(f"  排名 {doc['rank']}: {doc['document']} (分数: {doc['score']:.4f})")
            return True
        else:
            print(f"✗ 失败: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"✗ 异常: {e}")
        return False

def test_batch_rerank():
    """测试批量重排序功能"""
    batch_requests = [
        {
            "query": "机器学习",
            "documents": ["文档1", "文档2", "文档3"],
            "instruction": "判断文档是否与查询相关。"
        },
        {
            "query": "深度学习",
            "documents": ["文档4", "文档5", "文档6"],
            "instruction": "判断文档是否与查询相关。"
        }
    ]
    
    try:
        print("\n测试批量重排序...")
        start_time = time.time()
        response = requests.post("http://localhost:8000/batch_rerank", json=batch_requests)
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ 成功! 处理时间: {end_time - start_time:.3f}秒")
            print(f"处理了 {len(result['results'])} 个批次")
            return True
        else:
            print(f"✗ 失败: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"✗ 异常: {e}")
        return False

def test_health():
    """测试健康检查"""
    try:
        print("测试健康检查...")
        response = requests.get("http://localhost:8000/health")
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

def main():
    print("=" * 50)
    print("测试修复后的Rerank服务")
    print("=" * 50)
    
    # 等待服务启动
    print("等待服务启动...")
    time.sleep(3)
    
    # 测试健康检查
    if not test_health():
        print("服务未启动，请先启动服务")
        return
    
    # 测试单个重排序
    test_simple_rerank()
    
    # 测试批量重排序
    test_batch_rerank()
    
    print("\n" + "=" * 50)
    print("测试完成")

if __name__ == "__main__":
    main()
