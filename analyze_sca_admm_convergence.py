#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析SCA_ADMM在不同参数下的分配稳定性（收敛性分析）
分析维度：
1. 不同cache_nodes_ratio下的收敛性
2. 不同single_cache_capacity下的收敛性
3. 不同request_rate下的收敛性
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast
import os
import glob
import argparse

def calculate_allocation_change(allocation1, allocation2):
    """
    计算两个分配之间的变化幅度
    """
    total_l1 = 0
    total_l2 = 0
    max_change = 0
    total_changes = 0

    for node in allocation1:
        if node in allocation2:
            for sp_idx in range(len(allocation1[node])):
                change = abs(allocation1[node][sp_idx] - allocation2[node][sp_idx])
                total_l1 += change
                total_l2 += change ** 2
                max_change = max(max_change, change)
                total_changes += 1

    avg_change = total_l1 / total_changes if total_changes > 0 else 0
    l2_norm = np.sqrt(total_l2)

    return total_l1, l2_norm, max_change, avg_change

def load_allocation_file(filepath):
    """
    加载分配文件
    """
    allocations = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    alloc = ast.literal_eval(line)
                    allocations.append(alloc)
                except:
                    pass
    return allocations

def compute_l1_changes(allocations):
    """
    计算L1变化量序列
    """
    l1_changes = []
    for i in range(1, len(allocations)):
        l1, _, _, _ = calculate_allocation_change(allocations[i-1], allocations[i])
        l1_changes.append(l1)
    return l1_changes

def compute_moving_average(data, window_size):
    """
    计算滑动窗口平均
    """
    window_means = []
    intervals = []
    for i in range(len(data) - window_size + 1):
        window = data[i:i+window_size]
        window_means.append(np.mean(window))
        intervals.append(i + window_size // 2 + 2)  # +2因为interval从2开始
    return intervals, window_means

def find_sca_admm_file(input_dir):
    """
    在指定目录中查找SCA_ADMM的txt文件
    """
    patterns = [
        "*SCA_ADMM*.txt",
        "*sca_admm*.txt"
    ]

    for pattern in patterns:
        search_path = os.path.join(input_dir, pattern)
        files = glob.glob(search_path)
        if files:
            return files[0]

    # 尝试更宽松的搜索
    all_txt_files = glob.glob(os.path.join(input_dir, "*.txt"))
    for file_path in all_txt_files:
        filename = os.path.basename(file_path)
        if 'SCA_ADMM' in filename.upper():
            return file_path

    return None

def load_sca_admm_data(input_dir):
    """
    从指定目录加载SCA_ADMM的数据
    """
    file_path = find_sca_admm_file(input_dir)

    if not file_path:
        return None

    try:
        allocs = load_allocation_file(file_path)

        if len(allocs) < 2:
            return None

        l1_changes = compute_l1_changes(allocs)

        if len(l1_changes) < 3:
            return None

        # 使用固定窗口大小保持一致性
        window_size = 20
        if len(l1_changes) < window_size:
            window_size = max(3, len(l1_changes) // 5)

        intervals, ma = compute_moving_average(l1_changes, window_size=window_size)

        if len(ma) == 0:
            return None

        return {
            'intervals': intervals,
            'ma': ma,
            'filename': os.path.basename(file_path)
        }

    except Exception as e:
        print(f"  加载失败: {e}")
        return None

def analyze_cache_nodes_ratio(base_dir, output_dir):
    """
    分析不同cache_nodes_ratio下的收敛性
    """
    print("\n" + "="*80)
    print("分析1: 不同cache_nodes_ratio下的收敛性")
    print("="*80)

    # 定义要测试的cache_nodes_ratio值
    cache_nodes_ratios = [0.2, 0.4, 0.6, 0.8]

    # 颜色设置
    colors = ['#3182ce', '#38a169', '#ed8936', '#e53e3e']
    markers = ['o', 's', '^', 'D']

    fig, ax = plt.subplots(figsize=(8, 8))

    results = {}

    for i, ratio in enumerate(cache_nodes_ratios):
        # 构建搜索路径模式
        input_dir = os.path.join(base_dir, f'cache_nodes_ratio_{ratio}')

        if not os.path.exists(input_dir):
            print(f"\n⚠ 未找到目录: {input_dir}")
            continue

        print(f"\n处理 cache_nodes_ratio = {ratio}...")
        data = load_sca_admm_data(input_dir)

        if data:
            label = f'cache_nodes_ratio = {ratio}'
            ax.plot(data['intervals'], data['ma'],
                   label=label, color=colors[i],
                   linewidth=2.5, alpha=0.8, marker=markers[i],
                   markersize=4, markevery=10)

            results[ratio] = data
            print(f"  ✓ 加载成功: {data['filename']}")

            # 计算收敛指标
            ma_values = data['ma']
            threshold = np.mean(ma_values) * 0.5
            converge_idx = next((j for j, val in enumerate(ma_values) if val < threshold), None)
            if converge_idx is not None:
                print(f"  收敛interval: {data['intervals'][converge_idx]}")
        else:
            print(f"  ✗ 加载失败")

    # 设置图表属性
    ax.set_xlabel('Interval', fontsize=14, fontweight='bold')
    ax.set_ylabel('Moving Average L1 Change', fontsize=14, fontweight='bold')
    ax.set_title('SCA_ADMM Convergence Analysis\nUnder Different cache_nodes_ratio',
                fontsize=16, fontweight='bold', pad=20)

    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim(0, 310)

    plt.tight_layout()

    # 保存图像
    output_path = os.path.join(output_dir, 'convergence_cache_nodes_ratio.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ 图片已保存: {output_path}")

    plt.close()

    return results

def analyze_single_cache_capacity(base_dir, output_dir):
    """
    分析不同single_cache_capacity下的收敛性
    """
    print("\n" + "="*80)
    print("分析2: 不同single_cache_capacity下的收敛性")
    print("="*80)

    # 定义要测试的single_cache_capacity值
    cache_capacities = [20, 40, 60, 80]

    # 颜色设置
    colors = ['#3182ce', '#38a169', '#ed8936', '#e53e3e']
    markers = ['o', 's', '^', 'D']

    fig, ax = plt.subplots(figsize=(8, 8))

    results = {}

    for i, capacity in enumerate(cache_capacities):
        # 构建搜索路径模式
        input_dir = os.path.join(base_dir, f'single_cache_capacity_{capacity}')

        if not os.path.exists(input_dir):
            print(f"\n⚠ 未找到目录: {input_dir}")
            continue

        print(f"\n处理 single_cache_capacity = {capacity}...")
        data = load_sca_admm_data(input_dir)

        if data:
            label = f'cache_capacity = {capacity}'
            ax.plot(data['intervals'], data['ma'],
                   label=label, color=colors[i],
                   linewidth=2.5, alpha=0.8, marker=markers[i],
                   markersize=4, markevery=10)

            results[capacity] = data
            print(f"  ✓ 加载成功: {data['filename']}")

            # 计算收敛指标
            ma_values = data['ma']
            threshold = np.mean(ma_values) * 0.5
            converge_idx = next((j for j, val in enumerate(ma_values) if val < threshold), None)
            if converge_idx is not None:
                print(f"  收敛interval: {data['intervals'][converge_idx]}")
        else:
            print(f"  ✗ 加载失败")

    # 设置图表属性
    ax.set_xlabel('Interval', fontsize=14, fontweight='bold')
    ax.set_ylabel('Moving Average L1 Change', fontsize=14, fontweight='bold')
    ax.set_title('SCA_ADMM Convergence Analysis\nUnder Different single_cache_capacity',
                fontsize=16, fontweight='bold', pad=20)

    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim(0, 310)

    plt.tight_layout()

    # 保存图像
    output_path = os.path.join(output_dir, 'convergence_single_cache_capacity.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ 图片已保存: {output_path}")

    plt.close()

    return results

def analyze_request_rate(base_dir, output_dir):
    """
    分析不同request_rate下的收敛性
    """
    print("\n" + "="*80)
    print("分析3: 不同request_rate下的收敛性")
    print("="*80)

    # 定义要测试的request_rate值
    request_rates = [50, 100, 150, 200]

    # 颜色设置
    colors = ['#3182ce', '#38a169', '#ed8936', '#e53e3e']
    markers = ['o', 's', '^', 'D']

    fig, ax = plt.subplots(figsize=(8, 8))

    results = {}

    for i, rate in enumerate(request_rates):
        # 构建搜索路径模式
        input_dir = os.path.join(base_dir, f'request_rate_{rate}')

        if not os.path.exists(input_dir):
            print(f"\n⚠ 未找到目录: {input_dir}")
            continue

        print(f"\n处理 request_rate = {rate}...")
        data = load_sca_admm_data(input_dir)

        if data:
            label = f'request_rate = {rate}'
            ax.plot(data['intervals'], data['ma'],
                   label=label, color=colors[i],
                   linewidth=2.5, alpha=0.8, marker=markers[i],
                   markersize=4, markevery=10)

            results[rate] = data
            print(f"  ✓ 加载成功: {data['filename']}")

            # 计算收敛指标
            ma_values = data['ma']
            threshold = np.mean(ma_values) * 0.5
            converge_idx = next((j for j, val in enumerate(ma_values) if val < threshold), None)
            if converge_idx is not None:
                print(f"  收敛interval: {data['intervals'][converge_idx]}")
        else:
            print(f"  ✗ 加载失败")

    # 设置图表属性
    ax.set_xlabel('Interval', fontsize=14, fontweight='bold')
    ax.set_ylabel('Moving Average L1 Change', fontsize=14, fontweight='bold')
    ax.set_title('SCA_ADMM Convergence Analysis\nUnder Different request_rate',
                fontsize=16, fontweight='bold', pad=20)

    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim(0, 310)

    plt.tight_layout()

    # 保存图像
    output_path = os.path.join(output_dir, 'convergence_request_rate.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ 图片已保存: {output_path}")

    plt.close()

    return results

def print_summary(results_dict, param_name, param_values):
    """
    打印收敛性分析摘要
    """
    print("\n" + "="*80)
    print(f"收敛性分析摘要 - {param_name}")
    print("="*80)
    print(f"{param_name:<15} {'平均L1变化':<15} {'最小值':<15} {'最大值':<15} {'收敛interval':<15}")
    print("-" * 75)

    for param_val in param_values:
        if param_val in results_dict:
            data = results_dict[param_val]
            ma_values = data['ma']
            threshold = np.mean(ma_values) * 0.5
            converge_idx = next((j for j, val in enumerate(ma_values) if val < threshold), None)
            converge_interval = data['intervals'][converge_idx] if converge_idx is not None else 'N/A'

            print(f"{param_val:<15} {np.mean(ma_values):<15.4f} {np.min(ma_values):<15.4f} "
                  f"{np.max(ma_values):<15.4f} {converge_interval:<15}")

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='分析SCA_ADMM在不同参数下的收敛性')
    parser.add_argument('base_dir', help='包含不同参数实验结果的根目录路径')
    parser.add_argument('-o', '--output', help='输出图片的目录路径（可选）')

    args = parser.parse_args()

    # 验证输入目录
    if not os.path.isdir(args.base_dir):
        print(f"✗ 错误：输入目录 '{args.base_dir}' 不存在")
        return

    # 设置输出目录
    if args.output:
        output_dir = args.output
    else:
        output_dir = os.path.join(args.base_dir, 'convergence_analysis')
        os.makedirs(output_dir, exist_ok=True)

    print(f"输入目录: {args.base_dir}")
    print(f"输出目录: {output_dir}")

    # 执行三项分析
    results1 = analyze_cache_nodes_ratio(args.base_dir, output_dir)
    results2 = analyze_single_cache_capacity(args.base_dir, output_dir)
    results3 = analyze_request_rate(args.base_dir, output_dir)

    # 打印摘要
    if results1:
        print_summary(results1, 'cache_nodes_ratio', [0.2, 0.4, 0.6, 0.8])

    if results2:
        print_summary(results2, 'single_cache_capacity', [20, 40, 60, 80])

    if results3:
        print_summary(results3, 'request_rate', [50, 100, 150, 200])

    print("\n" + "="*80)
    print("所有分析完成！")
    print("="*80)
    print(f"\n生成的图片保存在: {output_dir}")
    print("  - convergence_cache_nodes_ratio.png")
    print("  - convergence_single_cache_capacity.png")
    print("  - convergence_request_rate.png")

if __name__ == '__main__':
    main()