#!/bin/bash

# 运行缓存资源分配模拟

# 确保在正确的目录中执行
cd "$(dirname "$0")"

echo "启动缓存资源分配模拟..."
echo "使用配置文件: config.yml"
echo ""

# 执行模拟
python simulation_code.py

echo ""
echo "模拟完成!"
echo "结果保存在: results/ 目录"
echo "图表保存在: figures/ 目录"
