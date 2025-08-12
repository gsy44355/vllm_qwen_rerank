#!/usr/bin/env python3
"""
测试 vLLM 异步引擎实现
"""

import asyncio
import sys
import os

# 添加当前目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_async_engine():
    """测试异步引擎的基本功能"""
    try:
        from vllm.engine.async_llm_engine import AsyncLLMEngine, AsyncEngineArgs
        
        print("✅ AsyncLLMEngine 导入成功")
        
        # 测试引擎参数创建
        engine_args = AsyncEngineArgs(
            model="microsoft/DialoGPT-small",  # 使用小模型进行测试
            max_model_len=512,
            enable_prefix_caching=True,
            gpu_memory_utilization=0.1,
            trust_remote_code=True,
            tensor_parallel_size=1,
        )
        print("✅ AsyncEngineArgs 创建成功")
        
        # 测试引擎初始化
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        print("✅ AsyncLLMEngine 初始化成功")
        
        # 测试异步生成
        test_prompt = "Hello, how are you?"
        sampling_params = {
            "temperature": 0,
            "max_tokens": 10,
        }
        
        print("测试异步生成...")
        async for output in engine.generate(test_prompt, sampling_params):
            print(f"✅ 异步生成成功: {output.outputs[0].text}")
            break
        
        # 关闭引擎
        await engine.close()
        print("✅ 引擎关闭成功")
        
        return True
        
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

async def test_compatible_service():
    """测试兼容性服务"""
    try:
        from rerank_service_compatible import initialize_model, ModelConfig
        
        print("\n测试兼容性服务...")
        
        # 创建测试配置
        config = ModelConfig(
            model_path="microsoft/DialoGPT-small",
            model_size="0.1B",
            gpu_memory_utilization=0.1,
            max_model_len=512
        )
        
        # 初始化模型
        await initialize_model(config)
        print("✅ 兼容性服务初始化成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 兼容性服务测试失败: {e}")
        return False

async def main():
    """主测试函数"""
    print("=" * 50)
    print("vLLM 异步引擎测试")
    print("=" * 50)
    
    # 测试异步引擎
    engine_ok = await test_async_engine()
    
    # 测试兼容性服务
    service_ok = await test_compatible_service()
    
    print("\n" + "=" * 50)
    print("测试结果")
    print("=" * 50)
    
    if engine_ok and service_ok:
        print("🎉 所有测试通过！")
        print("✅ 可以使用 rerank_service_compatible.py")
        print("✅ 异步引擎工作正常")
    else:
        print("❌ 部分测试失败")
        
        if not engine_ok:
            print("  - 异步引擎测试失败")
        if not service_ok:
            print("  - 兼容性服务测试失败")

if __name__ == "__main__":
    asyncio.run(main())
