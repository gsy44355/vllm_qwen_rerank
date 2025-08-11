#!/bin/bash

# Docker 部署脚本
set -e

echo "=================================================="
echo "vLLM Rerank 服务 Docker 部署"
echo "=================================================="

# 检查 Docker 是否运行
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker 未运行，请启动 Docker"
    exit 1
fi

# 检查 docker-compose 是否可用
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose 未安装"
    exit 1
fi

# 创建日志目录
mkdir -p logs

echo "📦 构建 Docker 镜像..."
docker-compose build

echo "🚀 启动服务..."
docker-compose up -d

echo "⏳ 等待服务启动..."
sleep 10

echo "🔍 检查服务状态..."
docker-compose ps

echo "🏥 检查健康状态..."
if docker-compose exec rerank-service curl -f http://localhost:8000/health; then
    echo "✅ 服务健康检查通过"
else
    echo "⚠️  健康检查失败，但服务可能仍在启动中"
fi

echo "🔧 测试环境变量..."
docker-compose exec rerank-service python test_env.py

echo "=================================================="
echo "部署完成！"
echo "服务地址: http://localhost:8888"
echo "健康检查: http://localhost:8888/health"
echo "API 文档: http://localhost:8888/docs"
echo "=================================================="

echo ""
echo "常用命令:"
echo "  查看日志: docker-compose logs -f"
echo "  停止服务: docker-compose down"
echo "  重启服务: docker-compose restart"
echo "  进入容器: docker-compose exec rerank-service bash"
