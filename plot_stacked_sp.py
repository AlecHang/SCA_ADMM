#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据分配结果绘制不同SP的堆叠图，并横向合并三个子图
"""

import os
import ast
import numpy as np
import matplotlib.pyplot as plt

def load_allocation_file(filepath):
    """加载分配文件"""
    allocations = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    alloc = ast.literal_eval(line)
                    allocations.append(alloc)
                except:
                    pass
    return allocations

def calculate_sp_distribution(allocations):
    """计算每个SP的分配比例随时间的变化"""
    if not allocations:
        return None

    first_alloc = allocations[0]
    nodes = list(first_alloc.keys())
    nSP = len(first_alloc[nodes[0]])

    sp_distributions = []

    for alloc in allocations:
        total_allocation = [0] * nSP

        for node in alloc:
            for sp_idx in range(nSP):
                total_allocation[sp_idx] += alloc[node][sp_idx]

        total = sum(total_allocation)
        if total > 0:
            proportions = [x / total * 100 for x in total_allocation]
        else:
            proportions = [0] * nSP

        sp_distributions.append(proportions)

    return np.array(sp_distributions)

def plot_combined_stacked_areas(methods_data, output_path):
    """绘制横向合并的三个堆叠面积图"""
    # 颜色设置（参考用户提供的图片格式）
    colors = [
        '#d4d4d4',  # SP1 - 浅灰色
        '#8a8a8a',  # SP2 - 中灰色
        '#2d2d2d'   # SP3 - 深灰色/黑色
    ]
    
    # 创建横向布局的子图（1行3列）
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 6), sharey=True)
    
    method_names = ['SCA_ADMM', 'SCA_NS', 'Q_learning']
    method_labels = ['(a) SCA_ADMM', '(b) SCA_NS', '(c) Q_learning']
    
    for i, (method_name, ax) in enumerate(zip(method_names, axes)):
        if method_name in methods_data:
            sp_distributions = methods_data[method_name]
            time_points = np.arange(len(sp_distributions))
            
            # 绘制堆叠面积图
            ax.stackplot(time_points,
                         sp_distributions[:, 0],
                         sp_distributions[:, 1],
                         sp_distributions[:, 2],
                         labels=['SP1', 'SP2', 'SP3'],
                         colors=colors,
                         alpha=0.9)
            
            # 设置x轴标签（只在底部显示）
            ax.set_xlabel('Time (Intervals)', fontsize=14)
            
            # 将方法标签放在x轴标签下方
            ax.text(0.5, -0.15, method_labels[i], transform=ax.transAxes,
                    ha='center', va='top', fontsize=16, fontweight='bold')
            
            # 设置y轴标签（只在左侧显示）
            if i == 0:
                ax.set_ylabel('Cache Allocation (%)', fontsize=14)
            
            # 设置y轴范围
            ax.set_ylim(0, 100)
            
            # 添加网格
            ax.grid(True, linestyle='--', alpha=0.6)
    
    # 在图的顶部添加统一的图例（居中）
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), 
               ncol=3, fontsize=16, frameon=True)
    
    # 调整布局，为底部标签留出空间
    plt.tight_layout(rect=[0, 0.15, 1, 0.95])
    
    # 保存图像
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"已保存合并图: {output_path}")
    
    plt.close()

def plot_individual_stacked_area(sp_distributions, method_name, output_path):
    """绘制单个堆叠面积图（保留原有功能）"""
    n_intervals = len(sp_distributions)
    time_points = np.arange(n_intervals)

    colors = [
        '#d4d4d4',  # SP1 - 浅灰色
        '#8a8a8a',  # SP2 - 中灰色
        '#2d2d2d'   # SP3 - 深灰色/黑色
    ]

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.stackplot(time_points,
                 sp_distributions[:, 0],
                 sp_distributions[:, 1],
                 sp_distributions[:, 2],
                 labels=['SP1', 'SP2', 'SP3'],
                 colors=colors,
                 alpha=0.9)

    ax.set_xlabel('Time (Intervals)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cache Allocation (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'{method_name}', fontsize=14, fontweight='bold', pad=20)

    ax.set_ylim(0, 100)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=3, fontsize=16, frameon=True)

    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"已保存: {output_path}")

    plt.close()

def main():
    base_path = 'results/GEANT_1/stability'
    output_dir = 'results/GEANT_1/stability/plots'

    os.makedirs(output_dir, exist_ok=True)

    # 按指定顺序排列方法
    methods = [
        ('SCA_ADMM', 'allocations_single_cache_capacity40_request_rate100_SCA_ADMM.txt'),
        ('SCA_NS', 'allocations_single_cache_capacity40_request_rate100_SCA_neighborhood_search.txt'),
        ('Q_learning', 'allocations_single_cache_capacity40_request_rate100_Q_learning.txt')
    ]
    
    # 存储所有方法的数据
    methods_data = {}
    
    # 处理每种方法
    for method_name, filename in methods:
        filepath = os.path.join(base_path, filename)
        
        if not os.path.exists(filepath):
            print(f"警告：未找到文件 {filepath}")
            continue
        
        # 加载分配数据
        allocations = load_allocation_file(filepath)
        print(f"\n处理 {method_name}...")
        print(f"  分配数量: {len(allocations)}")
        
        if not allocations:
            print("  警告：分配数据为空")
            continue
        
        # 计算SP分配比例
        sp_distributions = calculate_sp_distribution(allocations)
        
        if sp_distributions is None or len(sp_distributions) == 0:
            print("  警告：无法计算SP分配")
            continue
        
        methods_data[method_name] = sp_distributions
        print(f"  SP分布数据点: {len(sp_distributions)}")
        
        # 保存单个图
        output_path = os.path.join(output_dir, f'stacked_plot_{method_name}.png')
        plot_individual_stacked_area(sp_distributions, method_name, output_path)
    
    # 生成横向合并的大图
    if methods_data:
        combined_output_path = os.path.join(output_dir, 'stacked_plot_combined.png')
        plot_combined_stacked_areas(methods_data, combined_output_path)
    else:
        print("\n警告：没有足够的数据生成合并图")
    
    print("\n所有图已保存完成！")

if __name__ == '__main__':
    main()