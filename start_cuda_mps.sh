#!/bin/bash

# CUDA MPS 管理脚本
# 用于在宿主机上启动和停止 CUDA MPS 服务

set -e

MPS_PIPE_DIR="/tmp/nvidia-mps"
MPS_LOG_DIR="/tmp/nvidia-log"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查是否以 root 权限运行
check_root() {
    if [[ $EUID -eq 0 ]]; then
        print_warning "检测到 root 权限，建议使用普通用户运行"
    fi
}

# 检查 NVIDIA 驱动
check_nvidia_driver() {
    if ! command -v nvidia-smi &> /dev/null; then
        print_error "未找到 nvidia-smi，请确保已安装 NVIDIA 驱动"
        exit 1
    fi
    
    if ! nvidia-smi &> /dev/null; then
        print_error "nvidia-smi 运行失败，请检查 NVIDIA 驱动状态"
        exit 1
    fi
    
    print_info "NVIDIA 驱动检查通过"
}

# 检查 CUDA 工具包
check_cuda_toolkit() {
    if ! command -v nvidia-cuda-mps-control &> /dev/null; then
        print_error "未找到 nvidia-cuda-mps-control，请确保已安装 CUDA 工具包"
        exit 1
    fi
    
    print_info "CUDA 工具包检查通过"
}

# 创建必要的目录
create_directories() {
    print_info "创建 MPS 目录..."
    
    if [[ $EUID -eq 0 ]]; then
        # root 用户创建目录
        mkdir -p "$MPS_PIPE_DIR"
        mkdir -p "$MPS_LOG_DIR"
        chmod 777 "$MPS_PIPE_DIR"
        chmod 777 "$MPS_LOG_DIR"
    else
        # 普通用户创建目录
        mkdir -p "$MPS_PIPE_DIR"
        mkdir -p "$MPS_LOG_DIR"
    fi
    
    print_info "MPS 目录创建完成"
}

# 启动 CUDA MPS
start_mps() {
    print_info "启动 CUDA MPS 服务..."
    
    # 检查是否已经运行
    if pgrep -f "nvidia-cuda-mps" > /dev/null; then
        print_warning "CUDA MPS 服务已经在运行"
        return 0
    fi
    
    # 设置环境变量
    export CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE_DIR"
    export CUDA_MPS_LOG_DIRECTORY="$MPS_LOG_DIR"
    
    # 启动 MPS 守护进程
    if [[ $EUID -eq 0 ]]; then
        # root 用户启动
        nvidia-cuda-mps-control -d
    else
        # 普通用户启动
        nvidia-cuda-mps-control -d
    fi
    
    # 等待服务启动
    sleep 2
    
    # 检查服务状态
    if pgrep -f "nvidia-cuda-mps" > /dev/null; then
        print_info "CUDA MPS 服务启动成功"
        
        # 显示 MPS 状态
        echo "MPS 状态:"
        echo "QUERY MPS" | nvidia-cuda-mps-control
    else
        print_error "CUDA MPS 服务启动失败"
        exit 1
    fi
}

# 停止 CUDA MPS
stop_mps() {
    print_info "停止 CUDA MPS 服务..."
    
    # 检查是否在运行
    if ! pgrep -f "nvidia-cuda-mps" > /dev/null; then
        print_warning "CUDA MPS 服务未在运行"
        return 0
    fi
    
    # 停止 MPS 服务
    echo "SHUTDOWN" | nvidia-cuda-mps-control
    
    # 等待服务停止
    sleep 2
    
    # 强制终止进程（如果还在运行）
    if pgrep -f "nvidia-cuda-mps" > /dev/null; then
        print_warning "强制终止 MPS 进程..."
        pkill -f "nvidia-cuda-mps"
        sleep 1
    fi
    
    print_info "CUDA MPS 服务已停止"
}

# 显示 MPS 状态
show_status() {
    print_info "CUDA MPS 服务状态:"
    
    if pgrep -f "nvidia-cuda-mps" > /dev/null; then
        print_info "✓ MPS 服务正在运行"
        echo "MPS 状态:"
        echo "QUERY MPS" | nvidia-cuda-mps-control 2>/dev/null || echo "无法查询 MPS 状态"
    else
        print_warning "✗ MPS 服务未运行"
    fi
    
    echo ""
    print_info "目录状态:"
    if [[ -d "$MPS_PIPE_DIR" ]]; then
        echo "✓ MPS 管道目录: $MPS_PIPE_DIR"
    else
        echo "✗ MPS 管道目录不存在: $MPS_PIPE_DIR"
    fi
    
    if [[ -d "$MPS_LOG_DIR" ]]; then
        echo "✓ MPS 日志目录: $MPS_LOG_DIR"
    else
        echo "✗ MPS 日志目录不存在: $MPS_LOG_DIR"
    fi
}

# 清理 MPS 目录
cleanup() {
    print_info "清理 MPS 目录..."
    
    if [[ -d "$MPS_PIPE_DIR" ]]; then
        rm -rf "$MPS_PIPE_DIR"
        print_info "已删除 MPS 管道目录"
    fi
    
    if [[ -d "$MPS_LOG_DIR" ]]; then
        rm -rf "$MPS_LOG_DIR"
        print_info "已删除 MPS 日志目录"
    fi
}

# 显示帮助信息
show_help() {
    echo "CUDA MPS 管理脚本"
    echo ""
    echo "用法: $0 [命令]"
    echo ""
    echo "命令:"
    echo "  start     启动 CUDA MPS 服务"
    echo "  stop      停止 CUDA MPS 服务"
    echo "  restart   重启 CUDA MPS 服务"
    echo "  status    显示 MPS 服务状态"
    echo "  cleanup   清理 MPS 目录"
    echo "  help      显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 start    # 启动 MPS 服务"
    echo "  $0 status   # 查看状态"
    echo "  $0 stop     # 停止服务"
}

# 主函数
main() {
    case "${1:-help}" in
        start)
            check_root
            check_nvidia_driver
            check_cuda_toolkit
            create_directories
            start_mps
            ;;
        stop)
            stop_mps
            ;;
        restart)
            stop_mps
            sleep 1
            check_nvidia_driver
            check_cuda_toolkit
            create_directories
            start_mps
            ;;
        status)
            show_status
            ;;
        cleanup)
            stop_mps
            cleanup
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_error "未知命令: $1"
            show_help
            exit 1
            ;;
    esac
}

# 运行主函数
main "$@"
