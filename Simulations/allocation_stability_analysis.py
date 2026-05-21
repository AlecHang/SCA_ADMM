#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析动态分配方法的分配调整幅度随interval变化的情况
判断在什么程度达到稳态
"""

import os
import sys
import csv
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(__file__))

# 导入simulation_code模块
import simulation_code as sc

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

def run_allocation_stability_analysis(simulation_time=None):
    """
    运行分配稳定性分析实验
    
    分析动态分配方法的分配调整幅度随interval变化的情况
    """
    print("="*80)
    print("动态分配方法稳定性分析")
    print("="*80)
    
    # 初始化配置
    sc.init()
    
    # 获取配置参数
    config = sc.config
    topology_type = config['topology']['type']
    cache_nodes_ratio = config['topology']['cache_nodes_ratio']
    single_cache_capacity = config['simulation']['single_cache_capacity']
    request_rate = config['simulation']['request_rate']
    interval_size = config['simulation']['interval_size']
    
    # 使用指定的模拟时间或配置文件中的默认值
    if simulation_time is None:
        simulation_time = config['simulation']['time']
    
    delta = config['simulation']['delta']
    D = config['simulation']['D']
    
    print(f"\n实验参数:")
    print(f"  拓扑类型: {topology_type}")
    print(f"  缓存节点比例: {cache_nodes_ratio}")
    print(f"  单缓存容量: {single_cache_capacity}")
    print(f"  请求率: {request_rate}")
    print(f"  Interval大小: {interval_size}")
    print(f"  模拟时间: {simulation_time}")
    print(f"  Delta: {delta}")
    print(f"  D: {D}")
    
    # 计算interval数量
    nb_interval = int(simulation_time / interval_size)
    print(f"  Interval数量: {nb_interval}")
    
    # 动态分配方法列表
    dynamic_methods = ['SCA_ADMM', 'SCA_neighborhood_search', 'Q_learning']
    
    print(f"\n动态分配方法: {dynamic_methods}")
    
    # 生成视频概率分布
    videos_proba = sc.catalog()
    best_allocation = sc.decide_opt_alloc(videos_proba)
    
    # 初始化多缓存节点的分配
    cache_nodes = sc.topology_manager.get_cache_nodes()
    initial_allocation = {}
    node_cache_capacity = single_cache_capacity
    
    for i, node in enumerate(cache_nodes):
        node_allocation = [0] * sc.nSP
        actual_capacity = node_cache_capacity
        avg_per_sp = actual_capacity // sc.nSP
        rem_per_sp = actual_capacity % sc.nSP
        
        for sp in range(sc.nSP):
            node_allocation[sp] = avg_per_sp + (1 if sp < rem_per_sp else 0)
        
        initial_allocation[node] = node_allocation
    
    print(f"\n初始分配: {initial_allocation}")
    print(f"最佳分配: {best_allocation}")
    
    # 生成固定的请求序列
    print(f"\n生成固定的请求序列...")
    request_nb = int(interval_size * request_rate)
    fixed_requests = []
    
    sp_nodes = sc.topology_manager.get_sp_nodes()
    receiver_nodes = sc.topology_manager.get_receiver_nodes()
    router_nodes = sc.topology_manager.get_router_nodes()
    cache_nodes = sc.topology_manager.get_cache_nodes()
    
    for interval in range(nb_interval):
        interval_requests = {
            'requests': [],
            'source_nodes': []
        }
        
        for r in range(request_nb):
            request = sc.request_creation(videos_proba)
            interval_requests['requests'].append(request)
            
            possible_sources = []
            if router_nodes:
                possible_sources.extend(router_nodes)
            if cache_nodes:
                possible_sources.extend(cache_nodes)
            
            if possible_sources:
                import random as rd
                source_node = rd.choice(possible_sources)
            else:
                source_node = None
            interval_requests['source_nodes'].append(source_node)
        
        fixed_requests.append(interval_requests)
    
    print(f"生成了 {nb_interval} 个interval的固定请求，每个interval包含 {request_nb} 个请求")
    
    # 存储所有方法的结果和分配历史
    all_results = {}
    allocation_histories = {}
    
    # 运行每个动态分配方法
    for method in dynamic_methods:
        print(f"\n{'='*80}")
        print(f"运行方法: {method}")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        try:
            # 修改simulation_code以保存分配历史
            # 这里我们需要临时修改全局变量来捕获分配历史
            original_save_allocations = sc.save_allocations
            sc.save_allocations = True
            
            # 运行优化
            result = sc.optimize_nSP(
                initial_allocation, videos_proba, best_allocation, 
                request_rate, nb_interval, interval_size, 0.9, delta, D, method, 
                fixed_requests=fixed_requests, debug_interval=None
            )
            
            sc.save_allocations = original_save_allocations
            
            end_time = time.time()
            runtime = end_time - start_time
            
            # 解析结果
            if method == 'Q_learning':
                allocation, L_total_cost, L_nominal_cost, L_first_cost, L_best_cost, L_avg_latency = result
            else:
                allocation, L_total_cost, L_nominal_cost, L_first_cost, L_best_cost, L_avg_latency = result
            
            # 计算每个interval的命中率
            interval_hit_rates = [1 - cost for cost in L_nominal_cost]
            
            # 存储结果
            all_results[method] = {
                'runtime': runtime,
                'interval_hit_rates': interval_hit_rates,
                'interval_latencies': L_avg_latency,
                'avg_hit_rate': sum(interval_hit_rates) / len(interval_hit_rates),
                'avg_latency': sum(L_avg_latency) / len(L_avg_latency),
                'final_allocation': allocation
            }
            
            print(f"方法 {method} 运行时间: {runtime:.2f} 秒")
            print(f"平均命中率: {all_results[method]['avg_hit_rate']:.4f}")
            print(f"平均时延: {all_results[method]['avg_latency']:.4f}")
            
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            print(f"方法 {method} 执行失败: {e}")
            print(f"错误详情:")
            print(error_traceback)
            continue
    
    # 读取分配历史文件
    print(f"\n{'='*80}")
    print("读取分配历史文件")
    print(f"{'='*80}")
    
    for method in dynamic_methods:
        allocation_file = f"results/allocations_single_cache_capacity{single_cache_capacity}_request_rate{request_rate}_{method}.txt"
        
        if os.path.exists(allocation_file):
            print(f"读取 {method} 的分配历史: {allocation_file}")
            allocation_history = []
            
            with open(allocation_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # 解析分配历史
                lines = content.split('\n')
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('Interval') and '{' in line:
                        try:
                            # 解析分配字典
                            allocation = eval(line)
                            allocation_history.append(allocation)
                        except:
                            pass
            
            print(f"  读取到 {len(allocation_history)} 个分配记录")
            allocation_histories[method] = allocation_history
        else:
            print(f"警告：未找到 {method} 的分配历史文件")
            allocation_histories[method] = []
    
    # 分析分配调整幅度
    print(f"\n{'='*80}")
    print("分析分配调整幅度")
    print(f"{'='*80}")
    
    stability_results = {}
    
    for method in dynamic_methods:
        if method in allocation_histories and len(allocation_histories[method]) > 1:
            allocation_history = allocation_histories[method]
            
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
            
            stability_results[method] = {
                'l1_changes': l1_changes,
                'l2_changes': l2_changes,
                'max_changes': max_changes,
                'avg_changes': avg_changes,
                'avg_l1': np.mean(l1_changes) if l1_changes else 0,
                'avg_l2': np.mean(l2_changes) if l2_changes else 0,
                'avg_max': np.mean(max_changes) if max_changes else 0,
                'avg_avg': np.mean(avg_changes) if avg_changes else 0
            }
            
            print(f"\n{method} 分配调整幅度统计:")
            print(f"  平均L1变化量: {stability_results[method]['avg_l1']:.4f}")
            print(f"  平均L2变化量: {stability_results[method]['avg_l2']:.4f}")
            print(f"  平均最大变化量: {stability_results[method]['avg_max']:.4f}")
            print(f"  平均变化量: {stability_results[method]['avg_avg']:.4f}")
            
            # 分析稳态收敛
            # 使用滑动窗口方法检测稳态
            window_size = min(20, len(l1_changes) // 5)
            if window_size >= 5:
                # 计算滑动窗口的平均变化量
                window_means = []
                for i in range(len(l1_changes) - window_size + 1):
                    window = l1_changes[i:i+window_size]
                    window_means.append(np.mean(window))
                
                # 检测稳态：当连续几个窗口的平均变化量低于阈值时
                stability_threshold = np.mean(window_means) * 0.5
                steady_state_intervals = []
                
                for i in range(len(window_means) - 5):
                    recent_windows = window_means[i:i+5]
                    if all(w < stability_threshold for w in recent_windows):
                        steady_state_intervals.append(i + window_size)
                        break
                
                if steady_state_intervals:
                    steady_state_interval = steady_state_intervals[0]
                    print(f"  稳态收敛点: Interval {steady_state_interval}")
                    print(f"  收敛时的变化量: {window_means[steady_state_interval - window_size]:.4f}")
                else:
                    print(f"  未检测到明显的稳态收敛点")
    
    # 保存结果到CSV文件
    print(f"\n{'='*80}")
    print("保存结果到CSV文件")
    print(f"{'='*80}")
    
    # 创建结果目录
    results_dir = Path(os.path.join(os.path.dirname(__file__), '..', 'results'))
    results_dir.mkdir(exist_ok=True)
    
    # 生成文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"allocation_stability_{topology_type}_time{simulation_time}_ratio{cache_nodes_ratio}_cap{single_cache_capacity}_{timestamp}.csv"
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
        summary_fieldnames = ['method', 'avg_l1', 'avg_l2', 'avg_max', 'avg_avg']
        summary_writer = csv.DictWriter(csvfile, fieldnames=summary_fieldnames)
        summary_writer.writeheader()
        
        for method in dynamic_methods:
            if method in stability_results:
                summary_writer.writerow({
                    'method': method,
                    'avg_l1': stability_results[method]['avg_l1'],
                    'avg_l2': stability_results[method]['avg_l2'],
                    'avg_max': stability_results[method]['avg_max'],
                    'avg_avg': stability_results[method]['avg_avg']
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
    plot_filename = f"allocation_stability_{topology_type}_time{simulation_time}_ratio{cache_nodes_ratio}_cap{single_cache_capacity}_{timestamp}.png"
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
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='动态分配方法稳定性分析')
    parser.add_argument('--time', type=int, default=None, 
                        help='指定模拟时间（秒），默认使用配置文件中的值')
    
    args = parser.parse_args()
    
    # 运行实验
    results = run_allocation_stability_analysis(simulation_time=args.time)