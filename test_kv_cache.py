#!/usr/bin/env python3
"""
测试KV缓存是否生效
通过重复相同请求来验证缓存效果
"""

import requests
import time
import statistics

def test_kv_cache():
    """测试KV缓存效果"""
    print("=" * 60)
    print("KV缓存效果测试")
    print("=" * 60)
    
    # 测试数据
    test_data = {
        "query": "机器学习",
        "documents": [
            "机器学习是人工智能的一个子领域",
            "深度学习是机器学习的一种方法", 
            "自然语言处理是AI的重要分支",
            "计算机视觉处理图像和视频",
            "强化学习通过与环境交互来学习"
        ],
        "instruction": "判断文档是否与查询相关。答案只能是'yes'或'no'。"
    }
    
    # 第一次请求 - 冷启动
    print("第一次请求 (冷启动)...")
    start_time = time.time()
    response1 = requests.post("http://localhost:8000/rerank", json=test_data)
    first_time = time.time() - start_time
    
    if response1.status_code != 200:
        print(f"请求失败: {response1.status_code}")
        return
    
    print(f"第一次请求时间: {first_time:.3f}秒")
    
    # 重复相同请求 - 应该使用KV缓存
    print("\n重复相同请求 (应该使用KV缓存)...")
    times_with_cache = []
    
    for i in range(5):
        start_time = time.time()
        response = requests.post("http://localhost:8000/rerank", json=test_data)
        request_time = time.time() - start_time
        times_with_cache.append(request_time)
        
        if response.status_code == 200:
            print(f"第{i+1}次重复请求时间: {request_time:.3f}秒")
        else:
            print(f"第{i+1}次请求失败: {response.status_code}")
    
    # 计算统计信息
    avg_with_cache = statistics.mean(times_with_cache)
    min_with_cache = min(times_with_cache)
    max_with_cache = max(times_with_cache)
    
    print(f"\n统计信息:")
    print(f"冷启动时间: {first_time:.3f}秒")
    print(f"缓存后平均时间: {avg_with_cache:.3f}秒")
    print(f"缓存后最快时间: {min_with_cache:.3f}秒")
    print(f"缓存后最慢时间: {max_with_cache:.3f}秒")
    
    # 计算性能提升
    if first_time > 0:
        speedup = first_time / avg_with_cache
        print(f"性能提升: {speedup:.2f}x")
        
        if speedup > 1.5:
            print("✓ KV缓存生效！性能显著提升")
        elif speedup > 1.1:
            print("⚠ KV缓存部分生效，性能略有提升")
        else:
            print("✗ KV缓存可能未生效，性能提升不明显")

def test_concurrent_same_requests():
    """测试并发相同请求的缓存效果"""
    print("\n" + "=" * 60)
    print("并发相同请求测试")
    print("=" * 60)
    
    import asyncio
    import aiohttp
    
    async def make_request(session, request_id):
        data = {
            "query": "深度学习",
            "documents": [
                "深度学习是机器学习的一种方法",
                "神经网络是深度学习的基础",
                "卷积神经网络用于图像处理"
            ],
            "instruction": "判断文档是否与查询相关。"
        }
        
        start_time = time.time()
        async with session.post("http://localhost:8000/rerank", json=data) as response:
            request_time = time.time() - start_time
            result = await response.json()
            return request_id, request_time, response.status
    
    async def run_concurrent_test():
        async with aiohttp.ClientSession() as session:
            # 创建10个并发请求
            tasks = [make_request(session, i) for i in range(10)]
            results = await asyncio.gather(*tasks)
            
            times = [result[1] for result in results if result[2] == 200]
            
            if times:
                print(f"并发请求结果:")
                print(f"请求数量: {len(times)}")
                print(f"平均时间: {statistics.mean(times):.3f}秒")
                print(f"最快时间: {min(times):.3f}秒")
                print(f"最慢时间: {max(times):.3f}秒")
                print(f"时间标准差: {statistics.stdev(times):.3f}秒")
                
                # 如果时间差异很小，说明缓存生效
                if max(times) - min(times) < 0.1:
                    print("✓ 并发请求时间一致，KV缓存生效")
                else:
                    print("⚠ 并发请求时间差异较大，KV缓存可能未完全生效")
    
    asyncio.run(run_concurrent_test())

def test_different_requests():
    """测试不同请求的缓存效果"""
    print("\n" + "=" * 60)
    print("不同请求测试")
    print("=" * 60)
    
    # 测试不同的查询
    queries = [
        "机器学习",
        "深度学习", 
        "自然语言处理",
        "计算机视觉",
        "强化学习"
    ]
    
    documents = [
        "机器学习是人工智能的一个子领域",
        "深度学习是机器学习的一种方法",
        "自然语言处理是AI的重要分支"
    ]
    
    times = []
    
    for query in queries:
        data = {
            "query": query,
            "documents": documents,
            "instruction": "判断文档是否与查询相关。"
        }
        
        start_time = time.time()
        response = requests.post("http://localhost:8000/rerank", json=data)
        request_time = time.time() - start_time
        
        if response.status_code == 200:
            times.append(request_time)
            print(f"查询 '{query}' 处理时间: {request_time:.3f}秒")
        else:
            print(f"查询 '{query}' 失败: {response.status_code}")
    
    if times:
        print(f"\n不同查询统计:")
        print(f"平均时间: {statistics.mean(times):.3f}秒")
        print(f"时间范围: {min(times):.3f} - {max(times):.3f}秒")

def main():
    print("等待服务启动...")
    time.sleep(3)
    
    # 测试健康检查
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code != 200:
            print("服务未启动，请先启动服务")
            return
    except:
        print("服务未启动，请先启动服务")
        return
    
    # 运行测试
    test_kv_cache()
    test_concurrent_same_requests()
    test_different_requests()
    
    print("\n" + "=" * 60)
    print("测试完成")

if __name__ == "__main__":
    main()
