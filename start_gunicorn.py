#!/usr/bin/env python3
"""
使用 Gunicorn 启动 vLLM Rerank 服务 - 支持 CUDA MPS
"""

import os
import sys
import argparse
import logging
import subprocess

def setup_logging(log_level="INFO"):
    """设置日志配置"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('gunicorn_rerank_service.log')
        ]
    )

def check_dependencies():
    """检查依赖是否安装"""
    required_packages = [
        'vllm',
        'fastapi', 
        'uvicorn',
        'gunicorn',
        'transformers',
        'torch'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"缺少以下依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    return True

def check_gpu():
    """检查GPU可用性"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"检测到 {gpu_count} 个GPU设备")
            print(f"主要GPU: {gpu_name}")
            return True
        else:
            print("警告: 未检测到可用的GPU，服务可能无法正常运行")
            return False
    except Exception as e:
        print(f"检查GPU时出错: {e}")
        return False

def check_cuda_mps():
    """检查 CUDA MPS 状态"""
    try:
        # 检查 MPS 相关环境变量
        mps_enabled = os.getenv('CUDA_MPS_ENABLE', '0') == '1'
        mps_pipe_dir = os.getenv('CUDA_MPS_PIPE_DIRECTORY', '/tmp/nvidia-mps')
        mps_log_dir = os.getenv('CUDA_MPS_LOG_DIRECTORY', '/tmp/nvidia-log')
        
        print(f"CUDA MPS 启用状态: {mps_enabled}")
        print(f"MPS 管道目录: {mps_pipe_dir}")
        print(f"MPS 日志目录: {mps_log_dir}")
        
        # 检查目录是否存在
        if os.path.exists(mps_pipe_dir):
            print(f"✓ MPS 管道目录存在: {mps_pipe_dir}")
        else:
            print(f"⚠ MPS 管道目录不存在: {mps_pipe_dir}")
            
        if os.path.exists(mps_log_dir):
            print(f"✓ MPS 日志目录存在: {mps_log_dir}")
        else:
            print(f"⚠ MPS 日志目录不存在: {mps_log_dir}")
        
        return mps_enabled
    except Exception as e:
        print(f"检查 CUDA MPS 时出错: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="使用 Gunicorn 启动 vLLM Rerank 服务 - 支持 CUDA MPS")
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"), help="服务主机地址")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")), help="服务端口")
    parser.add_argument("--workers", type=int, default=int(os.getenv("WORKERS", "2")), help="工作进程数")
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"), choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="日志级别")
    parser.add_argument("--check-only", action="store_true", help="仅检查环境，不启动服务")
    parser.add_argument("--model-path", default=os.getenv("MODEL_PATH", "Qwen/Qwen3-Reranker-4B"), help="模型路径")
    parser.add_argument("--model-size", default=os.getenv("MODEL_SIZE", "4B"), choices=["0.6B", "4B", "8B"], help="模型大小")
    parser.add_argument("--gpu-memory-utilization", type=float, default=float(os.getenv("GPU_MEMORY_UTILIZATION", "0.8")), help="GPU内存使用率")
    parser.add_argument("--max-model-len", type=int, default=int(os.getenv("MAX_MODEL_LEN", "10000")), help="最大模型长度")
    parser.add_argument("--config", default="gun.py", help="Gunicorn配置文件路径")
    parser.add_argument("--enable-mps", action="store_true", help="启用 CUDA MPS 支持")
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    print("=" * 60)
    print("vLLM Rerank 服务 - Gunicorn 启动器 (支持 CUDA MPS)")
    print("=" * 60)
    
    # 检查依赖
    print("检查依赖...")
    if not check_dependencies():
        sys.exit(1)
    print("✓ 依赖检查通过")
    
    # 检查GPU
    print("检查GPU...")
    gpu_available = check_gpu()
    if gpu_available:
        print("✓ GPU检查通过")
    else:
        print("⚠ GPU不可用，但服务仍可启动")
    
    # 检查 CUDA MPS
    print("检查 CUDA MPS...")
    mps_enabled = check_cuda_mps()
    if mps_enabled:
        print("✓ CUDA MPS 已启用")
    else:
        print("⚠ CUDA MPS 未启用，将使用标准 CUDA 模式")
    
    if args.check_only:
        print("环境检查完成")
        return
    
    # 设置环境变量
    os.environ['MODEL_PATH'] = args.model_path
    os.environ['MODEL_SIZE'] = args.model_size
    os.environ['GPU_MEMORY_UTILIZATION'] = str(args.gpu_memory_utilization)
    os.environ['MAX_MODEL_LEN'] = str(args.max_model_len)
    os.environ['BIND'] = f"{args.host}:{args.port}"
    os.environ['WORKERS'] = str(args.workers)
    os.environ['LOG_LEVEL'] = args.log_level.lower()
    
    # CUDA MPS 相关环境变量
    if args.enable_mps or mps_enabled:
        os.environ['CUDA_MPS_ENABLE'] = '1'
        os.environ['VLLM_USE_MPS'] = '1'
        os.environ['VLLM_DISABLE_CUSTOM_ALL_REDUCE'] = '1'
        print("✓ 已启用 CUDA MPS 支持")
    
    # 启动服务
    print(f"启动 Gunicorn 服务: http://{args.host}:{args.port}")
    print(f"工作进程数: {args.workers}")
    print(f"模型路径: {args.model_path}")
    print(f"模型大小: {args.model_size}")
    print(f"GPU内存使用率: {args.gpu_memory_utilization}")
    print(f"最大模型长度: {args.max_model_len}")
    print(f"配置文件: {args.config}")
    print(f"CUDA MPS: {'启用' if mps_enabled else '禁用'}")
    print("按 Ctrl+C 停止服务")
    print("-" * 60)
    
    try:
        # 构建 Gunicorn 命令
        cmd = [
            "gunicorn",
            "--config", args.config,
            "--bind", f"{args.host}:{args.port}",
            "--workers", str(args.workers),
            "--worker-class", "uvicorn.workers.UvicornWorker",
            "--timeout", "300",
            "--keep-alive", "5",
            # 移除 max-requests 限制，避免频繁重启
            # "--max-requests", "1000",
            # "--max-requests-jitter", "100",
            "--preload", "false",  # 对于 CUDA MPS，禁用预加载
            "--worker-tmp-dir", "/dev/shm",
            "--log-level", args.log_level.lower(),
            # 添加 vLLM 特定的优化配置
            "--max-requests-jitter", "0",  # 禁用 jitter，避免意外重启
            "--graceful-timeout", "60",    # 优雅关闭超时时间
            "rerank_service:app"
        ]
        
        # 启动 Gunicorn
        subprocess.run(cmd, check=True)
        
    except KeyboardInterrupt:
        print("\n服务已停止")
    except subprocess.CalledProcessError as e:
        logger.error(f"Gunicorn 启动失败: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"启动服务时出错: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
