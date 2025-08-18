#!/usr/bin/env python3
"""
CUDA MPS 使用示例
演示如何在 Python 中检测和使用 CUDA MPS
"""

import os
import torch
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_cuda_mps():
    """检查 CUDA MPS 状态"""
    print("=" * 50)
    print("CUDA MPS 状态检查")
    print("=" * 50)
    
    # 检查环境变量
    mps_enabled = os.getenv('CUDA_MPS_ENABLE', '0') == '1'
    mps_pipe_dir = os.getenv('CUDA_MPS_PIPE_DIRECTORY', '/tmp/nvidia-mps')
    mps_log_dir = os.getenv('CUDA_MPS_LOG_DIRECTORY', '/tmp/nvidia-log')
    
    print(f"CUDA_MPS_ENABLE: {mps_enabled}")
    print(f"CUDA_MPS_PIPE_DIRECTORY: {mps_pipe_dir}")
    print(f"CUDA_MPS_LOG_DIRECTORY: {mps_log_dir}")
    
    # 检查目录
    print(f"\n目录检查:")
    print(f"MPS 管道目录存在: {os.path.exists(mps_pipe_dir)}")
    print(f"MPS 日志目录存在: {os.path.exists(mps_log_dir)}")
    
    # 检查 CUDA 可用性
    print(f"\nCUDA 检查:")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU 数量: {torch.cuda.device_count()}")
        print(f"当前 GPU: {torch.cuda.current_device()}")
        print(f"GPU 名称: {torch.cuda.get_device_name(0)}")
    
    return mps_enabled

def test_cuda_operations():
    """测试 CUDA 操作"""
    print("\n" + "=" * 50)
    print("CUDA 操作测试")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("CUDA 不可用，跳过测试")
        return
    
    try:
        # 创建张量
        print("创建 CUDA 张量...")
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        
        # 执行矩阵乘法
        print("执行矩阵乘法...")
        z = torch.mm(x, y)
        
        # 检查结果
        print(f"结果张量形状: {z.shape}")
        print(f"结果张量设备: {z.device}")
        
        # 清理内存
        del x, y, z
        torch.cuda.empty_cache()
        
        print("✓ CUDA 操作测试成功")
        
    except Exception as e:
        print(f"✗ CUDA 操作测试失败: {e}")

def simulate_vllm_initialization():
    """模拟 vLLM 初始化过程"""
    print("\n" + "=" * 50)
    print("vLLM 初始化模拟")
    print("=" * 50)
    
    # 检查 MPS 配置
    mps_enabled = os.getenv('CUDA_MPS_ENABLE', '0') == '1'
    
    if mps_enabled:
        print("✓ CUDA MPS 已启用")
        print("设置 vLLM MPS 环境变量...")
        
        # 设置 vLLM 相关环境变量
        os.environ['VLLM_USE_MPS'] = '1'
        os.environ['VLLM_DISABLE_CUSTOM_ALL_REDUCE'] = '1'
        
        print("✓ vLLM MPS 配置完成")
    else:
        print("⚠ CUDA MPS 未启用，使用标准 CUDA 模式")
    
    # 模拟多进程环境
    print(f"\n多进程配置:")
    workers = int(os.getenv('WORKERS', '1'))
    print(f"工作进程数: {workers}")
    
    if workers > 1 and not mps_enabled:
        print("⚠ 警告: 多进程模式但未启用 MPS，可能出现 CUDA 上下文冲突")
    elif workers > 1 and mps_enabled:
        print("✓ 多进程 + MPS 模式，应该可以正常工作")

def main():
    """主函数"""
    print("CUDA MPS 使用示例")
    print("=" * 60)
    
    # 检查 MPS 状态
    mps_enabled = check_cuda_mps()
    
    # 测试 CUDA 操作
    test_cuda_operations()
    
    # 模拟 vLLM 初始化
    simulate_vllm_initialization()
    
    print("\n" + "=" * 60)
    print("检查完成")
    print("=" * 60)
    
    if mps_enabled:
        print("✓ 建议: 可以安全使用多进程模式")
    else:
        print("⚠ 建议: 使用单进程模式或启用 CUDA MPS")

if __name__ == "__main__":
    main()
