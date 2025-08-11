import requests
import json

# 服务地址
BASE_URL = "http://localhost:8000"

def test_health():
    """测试健康检查"""
    response = requests.get(f"{BASE_URL}/health")
    print("健康检查:", response.json())
    return response.status_code == 200

def test_rerank():
    """测试重排序功能"""
    data = {
        "query": "暴力",
        "documents": [
            "杀人",
            "放火", 
            "亲吻",
            "睡觉"
        ],
        "instruction": "给出一个词，判断候选列表的词和给出词语最相近的"
    }
    
    response = requests.post(f"{BASE_URL}/rerank", json=data)
    if response.status_code == 200:
        result = response.json()
        print("重排序结果:")
        print(f"原始分数: {result['scores']}")
        print("排序后的文档:")
        for doc in result['ranked_documents']:
            print(f"  排名 {doc['rank']}: {doc['document']} (分数: {doc['score']:.4f})")
        return True
    else:
        print(f"请求失败: {response.status_code}")
        print(response.text)
        return False

def test_batch_rerank():
    """测试批量重排序功能"""
    batch_data = [
        {
            "query": "暴力",
            "documents": ["杀人", "放火", "亲吻", "睡觉"],
            "instruction": "给出一个词，判断候选列表的词和给出词语最相近的"
        },
        {
            "query": "学习",
            "documents": ["读书", "游戏", "运动", "思考"],
            "instruction": "给出一个词，判断候选列表的词和给出词语最相近的"
        }
    ]
    
    response = requests.post(f"{BASE_URL}/batch_rerank", json=batch_data)
    if response.status_code == 200:
        result = response.json()
        print("批量重排序结果:")
        for i, batch_result in enumerate(result['results']):
            print(f"\n批次 {i+1}:")
            print(f"查询: {batch_data[i]['query']}")
            print(f"原始分数: {batch_result['scores']}")
            for doc in batch_result['ranked_documents']:
                print(f"  排名 {doc['rank']}: {doc['document']} (分数: {doc['score']:.4f})")
        return True
    else:
        print(f"批量请求失败: {response.status_code}")
        print(response.text)
        return False

if __name__ == "__main__":
    print("开始测试rerank服务...")
    
    # 等待服务启动
    import time
    print("等待服务启动...")
    time.sleep(5)
    
    # 测试健康检查
    if test_health():
        print("✓ 健康检查通过")
        
        # 测试重排序
        if test_rerank():
            print("✓ 重排序测试通过")
        
        # 测试批量重排序
        if test_batch_rerank():
            print("✓ 批量重排序测试通过")
    else:
        print("✗ 健康检查失败，请确保服务正在运行")
