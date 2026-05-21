#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析动态分配方法的分配调整幅度随interval变化的情况
直接读取现有的分配历史文件进行分析
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

def calculate_allocation_change(allocation1, allocation2):
    """
    计算两个分配之间的变化幅度
    
    参数:
    allocation1: 第一个分配 {node: [sp1, sp2, ...]}
    allocation2: 第二个分配 {node: [sp1, sp2, ...]}
    
    返回:
    L1_norm: L1范数变化量
    L2_norm: L2范数变化量
    max_change: 最大单个变化量
    avg_change: 平均变化量
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

def parse_allocation_file(filepath):
    """
    解析分配历史文件
    
    参数:
    filepath: 分配历史文件路径
    
    返回:
    allocation_history: 分配历史列表
    """
    allocation_history = []
    
    if not os.path.exists(filepath):
        print(f"警告：文件 {filepath} 不存在")
        return allocation_history
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        
        # 解析分配历史
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line and '{' in line:
                try:
                    # 解析分配字典
                    allocation = eval(line)
                    allocation_history.append(allocation)
                except Exception as e:
                    print(f"解析行时出错: {e}")
                    pass
    
    return allocation_history

def analyze_stability(allocation_history, method_name):
    """
    分析分配稳定性
    
    参数:
    allocation_history: 分配历史
    method_name: 方法名称
    
    返回:
    stability_results: 稳定性分析结果
    """
    if len(allocation_history) < 2:
        print(f"{method_name}: 分配历史记录不足，无法分析")
        return None
    
    l1_changes = []
    l2_changes = []
    max_changes = []
    avg_changes = []
    
    for i in range(1, len(allocation_history)):
        l1, l2, max_change, avg_change = calculate_allocation_change(
            allocation_history[i-1], allocation_history[i]
        )
        l1_changes.append(l1)
        l2_changes.append(l2)
        max_changes.append(max_change)
        avg_changes.append(avg_change)
    
    stability_results = {
        'method': method_name,
        'l1_changes': l1_changes,
        'l2_changes': l2_changes,
        'max_changes': max_changes,
        'avg_changes': avg_changes,
        'avg_l1': np.mean(l1_changes) if l1_changes else 0,
        'avg_l2': np.mean(l2_changes) if l2_changes else 0,
        'avg_max': np.mean(max_changes) if max_changes else 0,
        'avg_avg': np.mean(avg_changes) if avg_changes else 0,
        'std_l1': np.std(l1_changes) if l1_changes else 0,
        'std_l2': np.std(l2_changes) if l2_changes else 0,
        'std_max': np.std(max_changes) if max_changes else 0,
        'std_avg': np.std(avg_changes) if avg_changes else 0
    }
    
    # 分析稳态收敛
    window_size = min(20, len(l1_changes) // 5)
    if window_size >= 5:
        # 计算滑动窗口的平均变化量
        window_means = []
        for i in range(len(l1_changes) - window_size + 1):
            window = l1_changes[i:i+window_size]
            window_means.append(np.mean(window))
        
        # 检测稳态：当连续几个窗口的平均变化量低于阈值时
        stability_threshold = np.mean(window_means) * 0.3
        steady_state_intervals = []
        
        for i in range(len(window_means) - 5):
            recent_windows = window_means[i:i+5]
            if all(w < stability_threshold for w in recent_windows):
                steady_state_intervals.append(i + window_size)
                break
        
        if steady_state_intervals:
            steady_state_interval = steady_state_intervals[0]
            stability_results['steady_state_interval'] = steady_state_interval
            stability_results['steady_state_change'] = window_means[steady_state_interval - window_size]
            stability_results['is_steady'] = True
        else:
            stability_results['steady_state_interval'] = None
            stability_results['steady_state_change'] = None
            stability_results['is_steady'] = False
    
    return stability_results

def main():
    """
    主函数
    """
    print("="*80)
    print("动态分配方法稳定性分析")
    print("="*80)
    
    # 设置结果目录
    results_dir = Path(os.path.join(os.path.dirname(__file__), 'results'))
    
    # 动态分配方法列表
    dynamic_methods = ['SCA_ADMM', 'SCA_neighborhood_search', 'Q_learning']
    
    # 读取所有方法的分配历史
    all_allocation_histories = {}
    
    for method in dynamic_methods:
        allocation_file = results_dir / f"allocations_single_cache_capacity40_request_rate100_{method}.txt"
        
        if allocation_file.exists():
            print(f"\n读取 {method} 的分配历史: {allocation_file}")
            allocation_history = parse_allocation_file(str(allocation_file))
            print(f"  读取到 {len(allocation_history)} 个分配记录")
            all_allocation_histories[method] = allocation_history
        else:
            print(f"警告：未找到 {method} 的分配历史文件")
            all_allocation_histories[method] = []
    
    # 分析每个方法的稳定性
    print(f"\n{'='*80}")
    print("分析分配调整幅度")
    print(f"{'='*80}")
    
    stability_results = {}
    
    for method in dynamic_methods:
        if method in all_allocation_histories and len(all_allocation_histories[method]) > 1:
            result = analyze_stability(all_allocation_histories[method], method)
            if result:
                stability_results[method] = result
                
                print(f"\n{method} 分配调整幅度统计:")
                print(f"  平均L1变化量: {result['avg_l1']:.4f} ± {result['std_l1']:.4f}")
                print(f"  平均L2变化量: {result['avg_l2']:.4f} ± {result['std_l2']:.4f}")
                print(f"  平均最大变化量: {result['avg_max']:.4f} ± {result['std_max']:.4f}")
                print(f"  平均变化量: {result['avg_avg']:.4f} ± {result['std_avg']:.4f}")
                
                if result['is_steady']:
                    print(f"  稳态收敛点: Interval {result['steady_state_interval']}")
                    print(f"  收敛时的变化量: {result['steady_state_change']:.4f}")
                else:
                    print(f"  未检测到明显的稳态收敛点")
    
    # 保存结果到CSV文件
    print(f"\n{'='*80}")
    print("保存结果到CSV文件")
    print(f"{'='*80}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"allocation_stability_analysis_{timestamp}.csv"
    filepath = results_dir / filename
    
    # 写入CSV文件
    with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['method', 'interval', 'l1_change', 'l2_change', 'max_change', 'avg_change']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        for method in dynamic_methods:
            if method in stability_results:
                for i in range(len(stability_results[method]['l1_changes'])):
                    writer.writerow({
                        'method': method,
                        'interval': i + 2,  # 从第2个interval开始有变化
                        'l1_change': stability_results[method]['l1_changes'][i],
                        'l2_change': stability_results[method]['l2_changes'][i],
                        'max_change': stability_results[method]['max_changes'][i],
                        'avg_change': stability_results[method]['avg_changes'][i]
                    })
        
        # 写入空行分隔
        writer.writerow({})
        
        # 写入汇总数据
        summary_fieldnames = ['method', 'avg_l1', 'std_l1', 'avg_l2', 'std_l2', 'avg_max', 'std_max', 'avg_avg', 'std_avg', 'steady_state_interval', 'steady_state_change']
        summary_writer = csv.DictWriter(csvfile, fieldnames=summary_fieldnames)
        summary_writer.writeheader()
        
        for method in dynamic_methods:
            if method in stability_results:
                summary_writer.writerow({
                    'method': method,
                    'avg_l1': stability_results[method]['avg_l1'],
                    'std_l1': stability_results[method]['std_l1'],
                    'avg_l2': stability_results[method]['avg_l2'],
                    'std_l2': stability_results[method]['std_l2'],
                    'avg_max': stability_results[method]['avg_max'],
                    'std_max': stability_results[method]['std_max'],
                    'avg_avg': stability_results[method]['avg_avg'],
                    'std_avg': stability_results[method]['std_avg'],
                    'steady_state_interval': stability_results[method].get('steady_state_interval', ''),
                    'steady_state_change': stability_results[method].get('steady_state_change', '')
                })
    
    print(f"结果已保存到: {filepath}")
    
    # 绘制分配调整幅度图
    print(f"\n{'='*80}")
    print("绘制分配调整幅度图")
    print(f"{'='*80}")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('动态分配方法分配调整幅度分析', fontsize=16, fontweight='bold')
    
    # L1变化量
    ax1 = axes[0, 0]
    for method in dynamic_methods:
        if method in stability_results and stability_results[method]['l1_changes']:
            intervals = range(2, len(stability_results[method]['l1_changes']) + 2)
            ax1.plot(intervals, stability_results[method]['l1_changes'], 
                    label=method, marker='o', markersize=3, linewidth=1.5)
    ax1.set_xlabel('Interval', fontsize=12)
    ax1.set_ylabel('L1 Change', fontsize=12)
    ax1.set_title('L1范数变化量', fontsize=14)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # L2变化量
    ax2 = axes[0, 1]
    for method in dynamic_methods:
        if method in stability_results and stability_results[method]['l2_changes']:
            intervals = range(2, len(stability_results[method]['l2_changes']) + 2)
            ax2.plot(intervals, stability_results[method]['l2_changes'], 
                    label=method, marker='s', markersize=3, linewidth=1.5)
    ax2.set_xlabel('Interval', fontsize=12)
    ax2.set_ylabel('L2 Change', fontsize=12)
    ax2.set_title('L2范数变化量', fontsize=14)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # 最大变化量
    ax3 = axes[1, 0]
    for method in dynamic_methods:
        if method in stability_results and stability_results[method]['max_changes']:
            intervals = range(2, len(stability_results[method]['max_changes']) + 2)
            ax3.plot(intervals, stability_results[method]['max_changes'], 
                    label=method, marker='^', markersize=3, linewidth=1.5)
    ax3.set_xlabel('Interval', fontsize=12)
    ax3.set_ylabel('Max Change', fontsize=12)
    ax3.set_title('最大单个变化量', fontsize=14)
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    # 平均变化量
    ax4 = axes[1, 1]
    for method in dynamic_methods:
        if method in stability_results and stability_results[method]['avg_changes']:
            intervals = range(2, len(stability_results[method]['avg_changes']) + 2)
            ax4.plot(intervals, stability_results[method]['avg_changes'], 
                    label=method, marker='D', markersize=3, linewidth=1.5)
    ax4.set_xlabel('Interval', fontsize=12)
    ax4.set_ylabel('Average Change', fontsize=12)
    ax4.set_title('平均变化量', fontsize=14)
    ax4.legend()
    ax4.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # 保存图像
    plot_filename = f"allocation_stability_analysis_{timestamp}.png"
    plot_filepath = results_dir / plot_filename
    plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
    print(f"图像已保存到: {plot_filepath}")
    
    plt.show()
    
    # 打印汇总结果
    print(f"\n{'='*80}")
    print("实验汇总")
    print(f"{'='*80}")
    print(f"{'方法':<30} {'平均L1变化':<15} {'平均L2变化':<15} {'平均最大变化':<15} {'平均变化':<15}")
    print("-"*80)
    
    for method in dynamic_methods:
        if method in stability_results:
            print(f"{method:<30} {stability_results[method]['avg_l1']:<15.4f} "
                  f"{stability_results[method]['avg_l2']:<15.4f} "
                  f"{stability_results[method]['avg_max']:<15.4f} "
                  f"{stability_results[method]['avg_avg']:<15.4f}")
    
    print("="*80)
    print("实验完成！")
    print("="*80)
    
    return stability_results

if __name__ == '__main__':
    results = main()