#!/usr/bin/env python3
"""
简单的 rerank 接口测试脚本
"""

import requests
import json
import time

# 服务配置
BASE_URL = "http://localhost:8000"

def test_health_check():
    """测试健康检查接口"""
    print("=" * 50)
    print("测试健康检查接口")
    print("=" * 50)
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"状态码: {response.status_code}")
        print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        return response.status_code == 200
    except Exception as e:
        print(f"健康检查失败: {e}")
        return False

def test_rerank_simple():
    """测试简单的重排序请求"""
    print("\n" + "=" * 50)
    print("测试简单重排序接口")
    print("=" * 50)
    
    # 测试数据
    test_data = {
        "query": "什么是人工智能？",
        "documents": [
            "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
            "机器学习是人工智能的一个子集，它使计算机能够在没有明确编程的情况下学习和改进。",
            "深度学习是机器学习的一个分支，使用神经网络来模拟人脑的工作方式。",
            "自然语言处理是人工智能的一个领域，专注于计算机理解和生成人类语言的能力。"
        ],
        "instruction": "判断文档是否与查询相关。答案只能是'yes'或'no'。",
        "max_length": 4096
    }
    
    try:
        print("发送请求...")
        print(f"查询: {test_data['query']}")
        print(f"文档数量: {len(test_data['documents'])}")
        
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/rerank",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        end_time = time.time()
        
        print(f"状态码: {response.status_code}")
        print(f"响应时间: {end_time - start_time:.2f}秒")
        
        if response.status_code == 200:
            result = response.json()
            print("\n重排序结果:")
            print(f"分数: {result['scores']}")
            
            print("\n排序后的文档:")
            for i, doc in enumerate(result['ranked_documents']):
                print(f"{i+1}. 分数: {doc['score']:.4f}, 排名: {doc['rank']}")
                print(f"   文档: {doc['document'][:100]}...")
                print()
        else:
            print(f"请求失败: {response.text}")
            
        return response.status_code == 200
        
    except Exception as e:
        print(f"测试失败: {e}")
        return False

def test_rerank_batch():
    """测试批量重排序接口"""
    print("\n" + "=" * 50)
    print("测试批量重排序接口")
    print("=" * 50)
    
    # 批量测试数据
    batch_data = [
        {
            "query": "Python编程语言",
            "documents": [
                "Python是一种高级编程语言，以其简洁的语法和强大的功能而闻名。",
                "Java是一种面向对象的编程语言，广泛用于企业级应用开发。",
                "Python在数据科学和机器学习领域非常流行。"
            ],
            "instruction": "判断文档是否与查询相关。答案只能是'yes'或'no'。",
            "max_length": 2048
        },
        {
            "query": "机器学习算法",
            "documents": [
                "监督学习算法需要标记的训练数据来学习模式。",
                "无监督学习算法在没有标记数据的情况下发现隐藏模式。",
                "强化学习算法通过与环境的交互来学习最优策略。"
            ],
            "instruction": "判断文档是否与查询相关。答案只能是'yes'或'no'。",
            "max_length": 2048
        }
    ]
    
    try:
        print("发送批量请求...")
        print(f"批量请求数量: {len(batch_data)}")
        
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/batch_rerank",
            json=batch_data,
            headers={"Content-Type": "application/json"}
        )
        end_time = time.time()
        
        print(f"状态码: {response.status_code}")
        print(f"响应时间: {end_time - start_time:.2f}秒")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n批量重排序结果 (共{len(result['results'])}个):")
            
            for i, batch_result in enumerate(result['results']):
                print(f"\n第{i+1}个查询结果:")
                print(f"查询: {batch_data[i]['query']}")
                print(f"分数: {batch_result['scores']}")
                
                for j, doc in enumerate(batch_result['ranked_documents']):
                    print(f"  {j+1}. 分数: {doc['score']:.4f}, 排名: {doc['rank']}")
                    print(f"      文档: {doc['document'][:80]}...")
        else:
            print(f"批量请求失败: {response.text}")
            
        return response.status_code == 200
        
    except Exception as e:
        print(f"批量测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("vLLM Rerank 接口测试")
    print("确保服务已启动在 http://localhost:8000")
    print()
    
    # 测试健康检查
    health_ok = test_health_check()
    if not health_ok:
        print("健康检查失败，请确保服务正在运行")
        return
    
    # 测试简单重排序
    simple_ok = test_rerank_simple()
    
    # 测试批量重排序
    batch_ok = test_rerank_batch()
    
    # 总结
    print("\n" + "=" * 50)
    print("测试总结")
    print("=" * 50)
    print(f"健康检查: {'✓' if health_ok else '✗'}")
    print(f"简单重排序: {'✓' if simple_ok else '✗'}")
    print(f"批量重排序: {'✓' if batch_ok else '✗'}")
    
    if all([health_ok, simple_ok, batch_ok]):
        print("\n🎉 所有测试通过！")
    else:
        print("\n❌ 部分测试失败，请检查服务状态")

if __name__ == "__main__":
    main()
