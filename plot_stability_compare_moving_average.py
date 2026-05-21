#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
比较不同动态分配方法的滑动窗口平均L1变化
支持从指定文件夹路径读取包含方法名字的txt文件
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

def find_method_file(input_dir, method_name):
    """
    在指定目录中查找包含方法名的txt文件
    """
    # 构建搜索模式
    patterns = [
        f"*{method_name.lower()}*.txt",
        f"*{method_name}*.txt",
        f"*_{method_name}_*.txt",
        f"*{method_name.replace('_', '')}*.txt"
    ]
    
    for pattern in patterns:
        search_path = os.path.join(input_dir, pattern)
        files = glob.glob(search_path)
        if files:
            # 返回找到的第一个文件
            return files[0]
    
    # 如果上面的模式都没找到，尝试更宽松的搜索
    all_txt_files = glob.glob(os.path.join(input_dir, "*.txt"))
    for file_path in all_txt_files:
        filename = os.path.basename(file_path)
        if method_name.lower() in filename.lower():
            return file_path
    
    return None

def load_method_data(input_dir, method_name):
    """
    从指定目录加载指定方法的数据
    """
    # 查找文件
    file_path = find_method_file(input_dir, method_name)
    
    if not file_path:
        print(f"✗ 未找到 {method_name} 的txt文件")
        return None
    
    print(f"✓ 找到 {method_name} 文件: {os.path.basename(file_path)}")
    
    try:
        # 加载分配数据
        allocs = load_allocation_file(file_path)
        print(f"  {method_name}分配数量: {len(allocs)}")
        
        if len(allocs) < 2:
            print(f"✗ {method_name} 文件数据不足（至少需要2个分配）")
            return None
        
        # 计算L1变化
        l1_changes = compute_l1_changes(allocs)
        print(f"  {method_name}变化量数量: {len(l1_changes)}")
        
        # 根据数据量动态调整窗口大小
        data_len = len(l1_changes)
        window_size = min(20, max(3, data_len // 5))
        print(f"  {method_name}窗口大小: {window_size}")
        
        if data_len < window_size:
            print(f"✗ {method_name} 数据量不足 ({data_len} < {window_size})")
            return None
        
        # 计算滑动平均
        intervals, ma = compute_moving_average(l1_changes, window_size=window_size)
        print(f"  {method_name}滑动平均数量: {len(ma)}")
        
        if len(ma) == 0:
            print(f"✗ {method_name} 滑动平均结果为空")
            return None
        
        return {'intervals': intervals, 'ma': ma}
    
    except Exception as e:
        print(f"✗ 加载 {method_name} 数据失败: {e}")
        return None

def plot_comparison(input_dir, output_file=None):
    """
    绘制不同方法的滑动窗口平均L1变化对比图
    """
    # 调整图片大小
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 颜色和标记设置
    colors = {
        'SCA_ADMM': "#e53e3e",      # 红色
        'SCA_NS': '#38a169',        # 绿色
        'Q_learning': '#3182ce'     # 蓝色
    }
    
    #科研常用颜色示例
    # 颜色和标记设置            
    # 蓝色 #3182ce
    # 绿色 #38a169
    # 橙色 #ed8936
    # 红色 #e53e3e
    # 紫色 #7c3aed
    # 黄色 #ecc94b
    # 灰色 #4a5568
    # 白色 #f7fafc
    # 黑色 #000000
    # 橙色 #ed8936
    
    markers = {
        'SCA_ADMM': 'o',            # 圆形
        'SCA_NS': '^',              # 三角形
        'Q_learning': 's'           # 方形
    }
    
    methods_data = {}
    
    # 从指定目录加载三种方法的数据
    print(f"\n{'='*80}")
    print(f"从目录加载数据: {input_dir}")
    print(f"{'='*80}")
    
    # 1. 加载SCA_ADMM数据
    sca_admm_data = load_method_data(input_dir, 'SCA_ADMM')
    if sca_admm_data:
        methods_data['SCA_ADMM'] = sca_admm_data
        print("✓ 已加载 SCA_ADMM 数据")
    
    # 2. 加载SCA_NS数据（也尝试SCA_neighborhood_search）
    sca_ns_data = load_method_data(input_dir, 'SCA_NS')
    if not sca_ns_data:
        sca_ns_data = load_method_data(input_dir, 'SCA_neighborhood_search')
    
    if sca_ns_data:
        methods_data['SCA_NS'] = sca_ns_data
        print("✓ 已加载 SCA_NS 数据")
    
    # 3. 加载Q_learning数据
    q_learning_data = load_method_data(input_dir, 'Q_learning')
    if q_learning_data:
        methods_data['Q_learning'] = q_learning_data
        print("✓ 已加载 Q_learning 数据")
    
    # 如果没有加载到任何数据，报错
    if not methods_data:
        print("\n✗ 未加载到任何方法的数据，请检查输入目录是否正确")
        return
    
    # 绘制所有可用方法的数据
    for method, data in methods_data.items():
        ax.plot(data['intervals'], data['ma'], 
                label=method, color=colors[method], 
                linewidth=2.5, alpha=0.8, marker=markers[method], 
                markersize=4, markevery=10)
    
    # 设置图表属性
    ax.set_xlabel('Interval', fontsize=14, fontweight='bold')
    ax.set_ylabel('Moving Average L1 Change', fontsize=14, fontweight='bold')
    ax.set_title('Comparison of Moving Average L1 Change\n(Different Dynamic Allocation Methods)', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # 计算并添加收敛阈值参考线
    all_ma_values = []
    for data in methods_data.values():
        all_ma_values.extend(data['ma'])
    
    stability_threshold = None
    if all_ma_values:
        # 使用SCA_ADMM的平均值作为基准（如果存在）
        if 'SCA_ADMM' in methods_data:
            baseline_mean = np.mean(methods_data['SCA_ADMM']['ma'])
        else:
            baseline_mean = np.mean(all_ma_values)
        
        stability_threshold = baseline_mean * 0.5
        ax.axhline(y=stability_threshold, color='#ed8936', linestyle='--', 
                   label=f'Stability Threshold ({stability_threshold:.2f})', alpha=0.6)
    
    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 动态设置x轴范围，显示所有数据
    max_interval = max(max(data['intervals']) for data in methods_data.values())
    ax.set_xlim(0, max_interval + 10)
    
    plt.tight_layout()
    
    # 保存图像
    if output_file:
        plot_filename = output_file
    else:
        plot_filename = os.path.join(input_dir, "moving_average_comparison.png")
    
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    
    # 打印统计对比
    print("\n" + "="*80)
    print("不同方法的滑动窗口平均L1变化对比")
    print("="*80)
    print(f"\n{'方法':<15} {'平均L1变化':<15} {'标准差':<15} {'最小值':<15} {'最大值':<15}")
    print(f"{'-'*15} {'-'*15} {'-'*15} {'-'*15} {'-'*15}")
    
    for method, data in methods_data.items():
        ma_values = data['ma']
        print(f"{method:<15} {np.mean(ma_values):<15.4f} {np.std(ma_values):<15.4f} {np.min(ma_values):<15.4f} {np.max(ma_values):<15.4f}")
    
    # 分析收敛速度
    print("\n收敛速度分析:")
    if stability_threshold is not None:
        threshold = stability_threshold
        
        for method, data in methods_data.items():
            ma_values = data['ma']
            intervals = data['intervals']
            converge_idx = next((i for i, val in enumerate(ma_values) if val < threshold), None)
            if converge_idx is not None:
                converge_interval = intervals[converge_idx]
                print(f"{method} 在 Interval {converge_interval} 首次低于收敛阈值")
            else:
                print(f"{method} 未达到收敛阈值")
    
    print(f"\n对比图已保存到: {plot_filename}")
    print("\n" + "="*80)
    
    plt.show()

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='比较不同动态分配方法的滑动窗口平均L1变化')
    parser.add_argument('input_dir', help='包含分配文件的输入目录路径')
    parser.add_argument('-o', '--output', help='输出图片的路径（可选）')
    
    args = parser.parse_args()
    
    # 验证输入目录
    if not os.path.isdir(args.input_dir):
        print(f"✗ 错误：输入目录 '{args.input_dir}' 不存在")
        return
    
    # 绘制对比图
    plot_comparison(args.input_dir, args.output)

if __name__ == '__main__':
    main()