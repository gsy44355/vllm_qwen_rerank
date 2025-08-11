#!/bin/bash

# vLLM Rerank 服务 Docker 部署脚本

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 配置
IMAGE_NAME="vllm-rerank-service"
CONTAINER_NAME="vllm-rerank-service"
PORT=8000

# 默认模型配置
DEFAULT_MODEL_PATH="Qwen/Qwen3-Reranker-4B"
DEFAULT_MODEL_SIZE="4B"
DEFAULT_GPU_MEMORY_UTILIZATION="0.8"
DEFAULT_MAX_MODEL_LEN="10000"

# 打印带颜色的消息
print_message() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}  vLLM Rerank 服务 Docker 部署${NC}"
    echo -e "${BLUE}================================${NC}"
}

# 检查Docker是否安装
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker 未安装，请先安装 Docker"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose 未安装，请先安装 Docker Compose"
        exit 1
    fi
    
    print_message "Docker 环境检查通过"
}

# 检查NVIDIA Docker支持
check_nvidia_docker() {
    if command -v nvidia-docker &> /dev/null; then
        print_message "检测到 NVIDIA Docker 支持"
        return 0
    elif docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi &> /dev/null; then
        print_message "检测到 Docker GPU 支持"
        return 0
    else
        print_warning "未检测到 GPU 支持，服务将在 CPU 模式下运行"
        return 1
    fi
}

# 构建镜像
build_image() {
    print_message "开始构建 Docker 镜像..."
    docker build -t $IMAGE_NAME .
    print_message "镜像构建完成"
}

# 启动服务
start_service() {
    print_message "启动服务..."
    
    # 创建日志目录
    mkdir -p logs
    
    # 读取环境变量或使用默认值
    MODEL_PATH=${MODEL_PATH:-$DEFAULT_MODEL_PATH}
    MODEL_SIZE=${MODEL_SIZE:-$DEFAULT_MODEL_SIZE}
    GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-$DEFAULT_GPU_MEMORY_UTILIZATION}
    MAX_MODEL_LEN=${MAX_MODEL_LEN:-$DEFAULT_MAX_MODEL_LEN}
    
    print_message "模型配置:"
    print_message "  路径: $MODEL_PATH"
    print_message "  大小: $MODEL_SIZE"
    print_message "  GPU内存使用率: $GPU_MEMORY_UTILIZATION"
    print_message "  最大模型长度: $MAX_MODEL_LEN"
    
    # 使用docker-compose启动
    docker-compose up -d
    
    print_message "服务启动完成"
    print_message "服务地址: http://localhost:$PORT"
    print_message "API文档: http://localhost:$PORT/docs"
    print_message "健康检查: http://localhost:$PORT/health"
}

# 停止服务
stop_service() {
    print_message "停止服务..."
    docker-compose down
    print_message "服务已停止"
}

# 重启服务
restart_service() {
    print_message "重启服务..."
    docker-compose restart
    print_message "服务重启完成"
}

# 查看日志
show_logs() {
    print_message "显示服务日志..."
    docker-compose logs -f
}

# 查看状态
show_status() {
    print_message "服务状态:"
    docker-compose ps
    
    echo ""
    print_message "容器资源使用情况:"
    docker stats --no-stream $CONTAINER_NAME 2>/dev/null || print_warning "容器未运行"
}

# 清理
cleanup() {
    print_warning "这将删除所有相关的容器、镜像和数据"
    read -p "确定要继续吗? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_message "清理中..."
        docker-compose down -v
        docker rmi $IMAGE_NAME 2>/dev/null || true
        docker system prune -f
        print_message "清理完成"
    else
        print_message "取消清理"
    fi
}

# 测试服务
test_service() {
    print_message "测试服务..."
    
    # 等待服务启动
    sleep 10
    
    # 测试健康检查
    if curl -f http://localhost:$PORT/health > /dev/null 2>&1; then
        print_message "✓ 健康检查通过"
    else
        print_error "✗ 健康检查失败"
        return 1
    fi
    
    # 测试重排序API
    test_data='{
        "query": "测试",
        "documents": ["文档1", "文档2", "文档3"],
        "instruction": "测试指令"
    }'
    
    if curl -X POST http://localhost:$PORT/rerank \
        -H "Content-Type: application/json" \
        -d "$test_data" > /dev/null 2>&1; then
        print_message "✓ API测试通过"
    else
        print_error "✗ API测试失败"
        return 1
    fi
    
    print_message "所有测试通过"
}

# 显示帮助信息
show_help() {
    echo "用法: $0 [命令]"
    echo ""
    echo "命令:"
    echo "  build     构建 Docker 镜像"
    echo "  start     启动服务"
    echo "  stop      停止服务"
    echo "  restart   重启服务"
    echo "  logs      查看日志"
    echo "  status    查看状态"
    echo "  test      测试服务"
    echo "  cleanup   清理所有容器和镜像"
    echo "  deploy    完整部署（构建+启动）"
    echo "  help      显示此帮助信息"
    echo ""
    echo "环境变量:"
    echo "  MODEL_PATH                模型路径（默认: $DEFAULT_MODEL_PATH）"
    echo "  MODEL_SIZE                模型大小 0.6B/4B/8B（默认: $DEFAULT_MODEL_SIZE）"
    echo "  GPU_MEMORY_UTILIZATION    GPU内存使用率 0.1-1.0（默认: $DEFAULT_GPU_MEMORY_UTILIZATION）"
    echo "  MAX_MODEL_LEN             最大模型长度（默认: $DEFAULT_MAX_MODEL_LEN）"
    echo "  LOCAL_MODEL_PATH          本地模型目录路径"
    echo ""
    echo "示例:"
    echo "  $0 deploy    # 完整部署"
    echo "  MODEL_PATH=/models/qwen-rerank-4b MODEL_SIZE=4B $0 deploy  # 使用本地模型"
    echo "  GPU_MEMORY_UTILIZATION=0.6 $0 deploy  # 降低GPU内存使用率"
    echo "  $0 logs      # 查看日志"
    echo "  $0 status    # 查看状态"
}

# 主函数
main() {
    print_header
    
    case "${1:-help}" in
        build)
            check_docker
            build_image
            ;;
        start)
            check_docker
            start_service
            ;;
        stop)
            stop_service
            ;;
        restart)
            restart_service
            ;;
        logs)
            show_logs
            ;;
        status)
            show_status
            ;;
        test)
            test_service
            ;;
        cleanup)
            cleanup
            ;;
        deploy)
            check_docker
            check_nvidia_docker
            build_image
            start_service
            print_message "等待服务完全启动..."
            sleep 30
            test_service
            ;;
        help|*)
            show_help
            ;;
    esac
}

# 执行主函数
main "$@"
