#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重复实验脚本 - 真实数据版本
支持多种优化方法和随机种子重复运行
"""

import os
import sys
import time
import uuid
import pandas as pd
import numpy as np
import json
import random
from datetime import datetime
from copy import deepcopy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import simulation_real_data as srd
from simulation_real_data import (
    init, create_real_data_requests, evaluate_cost, decide_opt_alloc,
    catalog, topology_manager,
    real_data_config, SP_proba, cacheable_content,
    user_to_cache_latency, cache_to_sp_latency, real_data_rankings
)

videos_proba = None


def run_simulation_with_seed(seed, method, nb_intervals, interval_size, request_rate, debug=False):
    """使用指定种子运行一次完整模拟"""
    random.seed(seed)
    np.random.seed(seed)

    srd.rd.seed(seed)
    if hasattr(srd, 'topology_manager') and srd.topology_manager:
        pass

    sp_names = srd.real_data_config.get('sp_names', ['youtube', 'netflix', 'douyin'])
    requests_per_interval = srd.real_data_config.get('requests_per_interval', 100)

    fixed_requests = create_real_data_requests(sp_names, requests_per_interval, nb_intervals)

    initial_allocation = {}
    cache_nodes = srd.topology_manager.get_cache_nodes()
    for node in cache_nodes:
        node_allocation = [0] * srd.nSP
        actual_capacity = srd.single_cache_capacity
        avg_per_sp = actual_capacity // srd.nSP
        rem_per_sp = actual_capacity % srd.nSP
        for sp in range(srd.nSP):
            node_allocation[sp] = avg_per_sp + (1 if sp < rem_per_sp else 0)
        initial_allocation[node] = node_allocation

    current_videos_proba = srd.catalog()
    best_allocation = srd.decide_opt_alloc(current_videos_proba)

    delta = srd.config['simulation'].get('delta', 0.9)
    D = srd.config['simulation'].get('D', 10)

    result = srd.optimize_nSP(
        initial_allocation, current_videos_proba, best_allocation,
        request_rate, nb_intervals, interval_size, 0.9, delta, D, method,
        fixed_requests=fixed_requests, debug_interval=None, debug=debug
    )

    return result, fixed_requests


def run_repeated_experiments(nb_iterations=10, method_list=None, debug=None, fixed_seed=True, base_seed=42):
    """运行重复实验"""
    if method_list is None:
        method_list = ['SCA_ADMM', 'SCA_neighborhood_search', 'best_allocation', 'equal_allocation']

    # 从配置读取debug标志
    if debug is None:
        debug = srd.config.get('simulation', {}).get('debug', False)

    print("=" * 60)
    print("重复实验开始 - 真实数据版本")
    print("=" * 60)
    print(f"迭代次数: {nb_iterations}")
    print(f"方法列表: {method_list}")
    print(f"调试模式: {'开启' if debug else '关闭'}")
    print(f"固定种子: {'是' if fixed_seed else '否'}")
    if fixed_seed:
        print(f"基础种子: {base_seed}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 从 config 读取 data_ratio（优先从 real_data，其次从 simulation）
    data_ratio = srd.config.get('real_data', {}).get('data_ratio', 0.01)
    if data_ratio == 0.01 and 'simulation' in srd.config:
        data_ratio = srd.config['simulation'].get('data_ratio', 0.01)
    print(f"\n配置文件 data_ratio: {data_ratio}")
    
    # 读取统计文件获取原始总请求数
    stats_file = os.path.join(os.path.dirname(__file__), 'real_data', 'sp_statistics.json')
    total_original_requests = 0
    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            sp_stats = json.load(f)
            total_original_requests = sum(stats['original_total_requests'] for stats in sp_stats.values())
            print(f"原始总请求数: {total_original_requests}")
    
    # 计算预期筛选后总请求数
    total_filtered_requests = int(total_original_requests * data_ratio) if total_original_requests > 0 else 0
    print(f"预期筛选后总请求数: {total_filtered_requests} (原始总数 × {data_ratio})")

    interval_size = srd.config['simulation']['interval_size']
    request_rate = srd.config['simulation']['request_rate']
    
    # 根据预期筛选后总请求数调整间隔数
    if total_filtered_requests > 0:
        # 保持每个间隔的请求数不变，调整间隔数使得总请求数匹配
        nb_intervals = max(1, total_filtered_requests // request_rate)
    else:
        nb_intervals = int(srd.config['simulation']['time'] / interval_size)

    print(f"\n模拟参数:")
    print(f"  时间间隔: {interval_size}")
    print(f"  请求率: {request_rate}")
    print(f"  间隔数: {nb_intervals}")
    print(f"  总请求数: {nb_intervals * request_rate}")

    all_results = {}

    for method in method_list:
        print(f"\n{'='*60}")
        print(f"运行方法: {method}")
        print(f"{'='*60}")

        method_results = {
            'seed': [],
            'avg_latency': [],
            'cache_hit_rate': [],
            'runtime': []
        }

        start_time_total = time.time()

        for iteration in range(nb_iterations):
            if fixed_seed:
                # 使用固定种子序列，与zipf版本一致
                seed = base_seed + iteration * 111
            else:
                # 使用随机种子
                seed = random.randint(1, 10000)

            print(f"\n迭代 {iteration + 1}/{nb_iterations} (种子: {seed})")

            iter_start_time = time.time()

            result, fixed_requests = run_simulation_with_seed(
                seed, method, nb_intervals, interval_size, request_rate, debug=debug
            )

            iter_end_time = time.time()
            iter_runtime = iter_end_time - iter_start_time

            if result:
                allocation, L_total_cost, L_nominal_cost, L_first_cost, L_best_cost, L_avg_latency = result

                cache_hit_rate = 1.0 - (sum(L_nominal_cost) / len(L_nominal_cost)) if L_nominal_cost else 0.0
                avg_latency = sum(L_avg_latency) / len(L_avg_latency) if L_avg_latency else 0.0

                method_results['seed'].append(seed)
                method_results['avg_latency'].append(avg_latency)
                method_results['cache_hit_rate'].append(cache_hit_rate)
                method_results['runtime'].append(iter_runtime)

                print(f"  平均时延: {avg_latency:.4f}")
                print(f"  缓存命中率: {cache_hit_rate:.4f}")
                print(f"  运行时间: {iter_runtime:.2f}秒")

            elapsed = time.time() - start_time_total
            avg_time_per_iter = elapsed / (iteration + 1)
            remaining_iters = nb_iterations - (iteration + 1)
            eta = avg_time_per_iter * remaining_iters

            print(f"  预计剩余时间: {eta:.1f}秒")

        all_results[method] = method_results

    print("\n" + "=" * 60)
    print("实验结果汇总")
    print("=" * 60)

    summary_data = []
    for method in method_list:
        if method in all_results:
            results = all_results[method]
            avg_latency = np.mean(results['avg_latency'])
            std_latency = np.std(results['avg_latency'])
            min_latency = np.min(results['avg_latency'])
            max_latency = np.max(results['avg_latency'])

            avg_hit = np.mean(results['cache_hit_rate'])
            std_hit = np.std(results['cache_hit_rate'])
            min_hit = np.min(results['cache_hit_rate'])
            max_hit = np.max(results['cache_hit_rate'])

            avg_runtime = np.mean(results['runtime'])

            print(f"\n方法: {method}")
            print(f"  平均时延: {avg_latency:.4f} ± {std_latency:.4f} (范围: {min_latency:.4f} - {max_latency:.4f})")
            print(f"  缓存命中率: {avg_hit:.4f} ± {std_hit:.4f} (范围: {min_hit:.4f} - {max_hit:.4f})")
            print(f"  平均运行时间: {avg_runtime:.2f}秒")

            summary_data.append({
                'Method': method,
                'Avg_Latency': avg_latency,
                'Std_Latency': std_latency,
                'Min_Latency': min_latency,
                'Max_Latency': max_latency,
                'Avg_Hit_Rate': avg_hit,
                'Std_Hit_Rate': std_hit,
                'Min_Hit_Rate': min_hit,
                'Max_Hit_Rate': max_hit,
                'Avg_Runtime': avg_runtime
            })

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_id = str(uuid.uuid4())[:8]

    topology_type = srd.topology_type
    capacity = srd.single_cache_capacity
    request_rate = srd.config['simulation']['request_rate']

    results_dir = os.path.join(os.path.dirname(__file__), 'results', topology_type, 'real_data')
    os.makedirs(results_dir, exist_ok=True)

    summary_file = os.path.join(results_dir, f"repeated_experiments_summary_{topology_type}_{request_rate}_cap{capacity}_N{nb_intervals}_{timestamp}_{unique_id}.csv")
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(summary_file, index=False)
    print(f"\n汇总结果已保存到: {summary_file}")

    for method in method_list:
        if method in all_results:
            method_file = os.path.join(results_dir, f"repeated_experiments_{method}_{topology_type}_{request_rate}_cap{capacity}_N{nb_intervals}_{timestamp}_{unique_id}.csv")
            df_method = pd.DataFrame(all_results[method])
            df_method.to_csv(method_file, index=False)
            print(f"{method} 详细结果已保存到: {method_file}")

    print("\n" + "=" * 60)
    print("所有实验完成")
    print("=" * 60)

    return all_results, summary_data


def main():
    """主函数"""
    nb_iterations = 10
    method_list = ['SCA_neighborhood_search'] #['SCA_ADMM', 'SCA_neighborhood_search', 'global_opt_allocation', 'best_allocation', 'equal_allocation', 'proportional_allocation', 'cooperative_best_allocation', 'Q_learning']
    base_seed = 42
    fixed_seed = True

    # 解析命令行参数
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '--iterations' or sys.argv[i] == '-n':
            if i + 1 < len(sys.argv):
                try:
                    nb_iterations = int(sys.argv[i + 1])
                except ValueError:
                    print(f"无效的迭代次数: {sys.argv[i + 1]}，使用默认值 10")
                i += 2
            else:
                i += 1
        elif sys.argv[i] == '--method' or sys.argv[i] == '-m':
            if i + 1 < len(sys.argv):
                method_input = sys.argv[i + 1]
                if method_input == 'compare':
                    method_list = ['SCA_neighborhood_search', 'SCA_ADMM', 'global_opt_allocation', 'best_allocation', 'equal_allocation', 'proportional_allocation', 'cooperative_best_allocation', 'Q_learning']
                elif method_input == 'sca':
                    method_list = ['SCA_ADMM', 'SCA_neighborhood_search']
                elif method_input == 'static':
                    method_list = ['best_allocation', 'equal_allocation']
                else:
                    method_list = [method_input]
                i += 2
            else:
                i += 1
        elif sys.argv[i] == '--random-seed':
            fixed_seed = False
            i += 1
        elif sys.argv[i] == '--seed' or sys.argv[i] == '-s':
            if i + 1 < len(sys.argv):
                try:
                    base_seed = int(sys.argv[i + 1])
                    fixed_seed = True
                except ValueError:
                    print(f"无效的种子值: {sys.argv[i + 1]}，使用默认值 42")
                i += 2
            else:
                i += 1
        elif sys.argv[i] == '--help' or sys.argv[i] == '-h':
            print("用法: python repeated_experiments_real_data.py [选项]")
            print("选项:")
            print("  -n, --iterations N    设置迭代次数 (默认: 10)")
            print("  -m, --method METHOD   设置方法 (可用值: compare, sca, static, 或方法名)")
            print("  -s, --seed SEED       设置基础随机种子 (默认: 42)")
            print("  --random-seed         使用随机种子而不是固定种子")
            print("  -h, --help            显示此帮助信息")
            return
        else:
            i += 1

    print(f"将运行以下方法: {method_list}")
    print(f"迭代次数: {nb_iterations}")
    print(f"固定种子: {'是' if fixed_seed else '否'} (基础种子: {base_seed})")

    # 使用基础种子初始化，确保拓扑创建的可重复性
    init(seed=base_seed if fixed_seed else None)

    run_repeated_experiments(nb_iterations=nb_iterations, method_list=method_list, fixed_seed=fixed_seed, base_seed=base_seed)


if __name__ == "__main__":
    main()