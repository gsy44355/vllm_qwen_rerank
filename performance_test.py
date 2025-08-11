#!/usr/bin/env python3
"""
性能测试脚本
对比同步和异步版本的性能差异
"""

import asyncio
import aiohttp
import time
import statistics
from typing import List, Dict
import json

# 测试配置
BASE_URL = "http://localhost:8000"
CONCURRENT_REQUESTS = 10
TOTAL_REQUESTS = 50

# 测试数据
test_data = {
    "query": "人工智能",
    "documents": [
        "机器学习是人工智能的一个子领域",
        "深度学习是机器学习的一种方法", 
        "自然语言处理是AI的重要分支",
        "计算机视觉处理图像和视频",
        "强化学习通过试错来学习最优策略"
    ],
    "instruction": "判断文档是否与查询相关。答案只能是'yes'或'no'。"
}

async def make_request(session: aiohttp.ClientSession, request_id: int) -> Dict:
    """发送单个请求"""
    start_time = time.time()
    try:
        async with session.post(f"{BASE_URL}/rerank", json=test_data) as response:
            end_time = time.time()
            if response.status == 200:
                return {
                    "request_id": request_id,
                    "status": "success",
                    "response_time": end_time - start_time,
                    "status_code": response.status
                }
            else:
                return {
                    "request_id": request_id,
                    "status": "error",
                    "response_time": end_time - start_time,
                    "status_code": response.status,
                    "error": await response.text()
                }
    except Exception as e:
        end_time = time.time()
        return {
            "request_id": request_id,
            "status": "exception",
            "response_time": end_time - start_time,
            "error": str(e)
        }

async def run_concurrent_requests() -> List[Dict]:
    """运行并发请求测试"""
    print(f"开始并发测试: {TOTAL_REQUESTS} 个请求，并发数: {CONCURRENT_REQUESTS}")
    
    async with aiohttp.ClientSession() as session:
        # 创建信号量限制并发数
        semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
        
        async def limited_request(request_id: int):
            async with semaphore:
                return await make_request(session, request_id)
        
        # 创建所有请求任务
        tasks = [limited_request(i) for i in range(TOTAL_REQUESTS)]
        
        # 执行所有请求
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        return results, total_time

def analyze_results(results: List[Dict], total_time: float):
    """分析测试结果"""
    successful_requests = [r for r in results if r["status"] == "success"]
    failed_requests = [r for r in results if r["status"] != "success"]
    
    if not successful_requests:
        print("❌ 没有成功的请求")
        return
    
    response_times = [r["response_time"] for r in successful_requests]
    
    print("\n" + "="*50)
    print("性能测试结果")
    print("="*50)
    print(f"总请求数: {len(results)}")
    print(f"成功请求数: {len(successful_requests)}")
    print(f"失败请求数: {len(failed_requests)}")
    print(f"成功率: {len(successful_requests)/len(results)*100:.2f}%")
    print(f"总耗时: {total_time:.2f}秒")
    print(f"平均吞吐量: {len(successful_requests)/total_time:.2f} 请求/秒")
    
    print(f"\n响应时间统计:")
    print(f"  平均响应时间: {statistics.mean(response_times):.3f}秒")
    print(f"  中位数响应时间: {statistics.median(response_times):.3f}秒")
    print(f"  最小响应时间: {min(response_times):.3f}秒")
    print(f"  最大响应时间: {max(response_times):.3f}秒")
    print(f"  标准差: {statistics.stdev(response_times):.3f}秒")
    
    if failed_requests:
        print(f"\n失败请求详情:")
        for req in failed_requests:
            print(f"  请求 {req['request_id']}: {req['status']} - {req.get('error', 'N/A')}")

async def test_health():
    """测试服务健康状态"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{BASE_URL}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    print("✓ 服务健康检查通过")
                    print(f"模型配置: {data.get('model_config', 'N/A')}")
                    return True
                else:
                    print(f"✗ 服务健康检查失败: {response.status}")
                    return False
    except Exception as e:
        print(f"✗ 服务健康检查异常: {e}")
        return False

async def main():
    """主测试函数"""
    print("="*50)
    print("vLLM Rerank 服务性能测试")
    print("="*50)
    
    # 检查服务状态
    if not await test_health():
        print("服务未启动或不可用，请先启动服务")
        return
    
    print(f"\n测试配置:")
    print(f"  服务地址: {BASE_URL}")
    print(f"  总请求数: {TOTAL_REQUESTS}")
    print(f"  并发数: {CONCURRENT_REQUESTS}")
    print(f"  测试数据: {len(test_data['documents'])} 个文档")
    
    # 运行性能测试
    results, total_time = await run_concurrent_requests()
    
    # 分析结果
    analyze_results(results, total_time)
    
    print("\n" + "="*50)
    print("测试完成")
    print("="*50)

if __name__ == "__main__":
    asyncio.run(main())
