#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interval级别统计实验
统计所有动态分配方法在每个interval内部的平均缓存命中率和平均时延
支持指定模拟时间参数
"""

import os
import sys
import csv
import time
import argparse
from datetime import datetime
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(__file__))

# 导入simulation_code模块
import simulation_code as sc

def run_interval_statistics_experiment(simulation_time=None):
    """
    运行interval级别统计实验
    
    运行所有动态分配方法和静态方法，统计每个interval的平均缓存命中率和平均时延
    
    参数:
    simulation_time: 指定模拟时间，如果为None则使用配置文件中的默认值
    """
    print("="*80)
    print("Interval级别统计实验")
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
    
    # 静态分配方法列表（用于对比）
    static_methods = ['global_opt_allocation']
    
    # 所有方法
    all_methods = static_methods + dynamic_methods
    
    print(f"\n静态分配方法(对比): {static_methods}")
    print(f"动态分配方法: {dynamic_methods}")
    
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
    
    # 存储所有方法的结果
    all_results = {}
    
    # 运行每个方法（包括静态和动态方法）
    for method in all_methods:
        print(f"\n{'='*80}")
        print(f"运行方法: {method}")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        try:
            result = sc.optimize_nSP(
                initial_allocation, videos_proba, best_allocation, 
                request_rate, nb_interval, interval_size, 0.9, delta, D, method, 
                fixed_requests=fixed_requests, debug_interval=None
            )
            
            end_time = time.time()
            runtime = end_time - start_time
            
            # 解析结果
            if method == 'Q_learning':
                allocation, L_total_cost, L_nominal_cost, L_first_cost, L_best_cost, L_avg_latency = result
            else:
                allocation, L_total_cost, L_nominal_cost, L_first_cost, L_best_cost, L_avg_latency = result
            
            # 计算每个interval的命中率 (hit_rate = 1 - nominal_cost)
            interval_hit_rates = [1 - cost for cost in L_nominal_cost]
            
            # 存储结果
            all_results[method] = {
                'runtime': runtime,
                'interval_hit_rates': interval_hit_rates,
                'interval_latencies': L_avg_latency,
                'avg_hit_rate': sum(interval_hit_rates) / len(interval_hit_rates),
                'avg_latency': sum(L_avg_latency) / len(L_avg_latency)
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
    
    # 保存结果到CSV文件
    print(f"\n{'='*80}")
    print("保存结果到CSV文件")
    print(f"{'='*80}")
    
    # 创建结果目录
    results_dir = Path(os.path.join(os.path.dirname(__file__), '..', 'results'))
    results_dir.mkdir(exist_ok=True)
    
    # 生成文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"interval_statistics_{topology_type}_time{simulation_time}_ratio{cache_nodes_ratio}_cap{single_cache_capacity}_{timestamp}.csv"
    filepath = results_dir / filename
    
    # 写入CSV文件
    with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
        # 写入interval级别的详细数据
        fieldnames = ['method', 'interval', 'hit_rate', 'latency']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        for method in all_methods:
            if method in all_results:
                for interval in range(nb_interval):
                    writer.writerow({
                        'method': method,
                        'interval': interval + 1,
                        'hit_rate': all_results[method]['interval_hit_rates'][interval],
                        'latency': all_results[method]['interval_latencies'][interval]
                    })
        
        # 写入空行分隔
        writer.writerow({})
        
        # 写入汇总数据
        summary_fieldnames = ['method', 'runtime', 'avg_hit_rate', 'avg_latency']
        summary_writer = csv.DictWriter(csvfile, fieldnames=summary_fieldnames)
        summary_writer.writeheader()
        
        for method in all_methods:
            if method in all_results:
                summary_writer.writerow({
                    'method': method,
                    'runtime': all_results[method]['runtime'],
                    'avg_hit_rate': all_results[method]['avg_hit_rate'],
                    'avg_latency': all_results[method]['avg_latency']
                })
    
    print(f"结果已保存到: {filepath}")
    
    # 打印汇总结果
    print(f"\n{'='*80}")
    print("实验汇总")
    print(f"{'='*80}")
    print(f"{'方法':<30} {'运行时间(秒)':<15} {'平均命中率':<15} {'平均时延':<15}")
    print("-"*80)
    
    for method in all_methods:
        if method in all_results:
            print(f"{method:<30} {all_results[method]['runtime']:<15.2f} "
                  f"{all_results[method]['avg_hit_rate']:<15.4f} "
                  f"{all_results[method]['avg_latency']:<15.4f}")
    
    print("="*80)
    print("实验完成！")
    print("="*80)
    
    return all_results

if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Interval级别统计实验')
    parser.add_argument('--time', type=int, default=None, 
                        help='指定模拟时间（秒），默认使用配置文件中的值')
    
    args = parser.parse_args()
    
    # 运行实验
    results = run_interval_statistics_experiment(simulation_time=args.time)
