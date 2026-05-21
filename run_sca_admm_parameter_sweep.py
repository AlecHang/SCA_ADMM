#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCA_ADMM参数扫描实验
在不同参数下运行SCA_ADMM实验，生成缓存分配文件用于收敛性分析
参数维度：
1. cache_nodes_ratio: 0.2, 0.4, 0.6, 0.8
2. single_cache_capacity: 20, 40, 60, 80
3. request_rate: 50, 100, 150, 200
"""

import os
import sys
import yaml
import shutil
from datetime import datetime

# 添加Simulations目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Simulations'))

# 导入simulation_code模块
import simulation_code as sc

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_config(config, config_path):
    """保存配置文件"""
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

def run_single_experiment(base_output_dir, experiment_name, cache_nodes_ratio, single_cache_capacity, request_rate, simulation_time=300):
    """
    运行单次SCA_ADMM实验并保存分配文件
    """
    print(f"\n{'='*80}")
    print(f"运行实验: {experiment_name}")
    print(f"{'='*80}")
    print(f"  cache_nodes_ratio: {cache_nodes_ratio}")
    print(f"  single_cache_capacity: {single_cache_capacity}")
    print(f"  request_rate: {request_rate}")
    print(f"  simulation_time: {simulation_time}")

    # 创建实验输出目录
    experiment_dir = os.path.join(base_output_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    # 原始配置文件路径
    original_config_path = os.path.join('Simulations', 'config.yml')
    backup_config_path = original_config_path + '.sweep_backup'

    # 保存原始配置
    shutil.copy2(original_config_path, backup_config_path)

    try:
        # 加载原始配置
        config = load_config(original_config_path)

        # 修改配置参数
        config['simulation']['cache_nodes_ratio'] = cache_nodes_ratio
        config['simulation']['single_cache_capacity'] = single_cache_capacity
        config['simulation']['request_rate'] = request_rate
        config['simulation']['time'] = simulation_time

        # 确保启用分配结果保存
        config['output']['save_allocations'] = True

        # 保存修改后的配置
        save_config(config, original_config_path)

        print(f"  配置已更新，save_allocations=True")

        # 重新初始化simulation_code模块（使配置生效）
        sc.init()

        print(f"  运行SCA_ADMM实验...")

        # 运行SCA_ADMM实验
        # simulation_code.main() 会运行默认的优化方法
        sc.main()

        print(f"  ✓ 实验运行完成")

        # 查找生成的分配文件并复制到实验目录
        results_dir = 'results'
        allocation_files = []

        for filename in os.listdir(results_dir):
            if 'SCA_ADMM' in filename and filename.endswith('.txt'):
                allocation_files.append(filename)
                print(f"  找到分配文件: {filename}")

        if allocation_files:
            # 取最新的文件（按时间戳排序）
            latest_file = sorted(allocation_files)[-1]
            source_path = os.path.join(results_dir, latest_file)
            dest_path = os.path.join(experiment_dir, latest_file)

            shutil.copy2(source_path, dest_path)
            print(f"  ✓ 分配文件已保存: {dest_path}")
        else:
            print(f"  ✗ 未找到分配文件")

    finally:
        # 恢复原始配置
        shutil.move(backup_config_path, original_config_path)
        print(f"  配置已恢复")

    return experiment_dir

def run_parameter_sweep(base_output_dir='results/parameter_sweep'):
    """
    执行参数扫描实验
    """
    print("\n" + "="*80)
    print("SCA_ADMM参数扫描实验")
    print("="*80)
    print(f"输出目录: {base_output_dir}")

    # 创建时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    sweep_dir = os.path.join(base_output_dir, f'sca_admm_sweep_{timestamp}')
    os.makedirs(sweep_dir, exist_ok=True)

    # 定义参数范围
    cache_nodes_ratios = [0.2, 0.4, 0.6, 0.8]
    single_cache_capacities = [20, 40, 60, 80]
    request_rates = [50, 100, 150, 200]
    simulation_time = 300

    all_experiments = []

    # 基准参数（用于单变量分析）
    base_params = {
        'cache_nodes_ratio': 0.4,
        'single_cache_capacity': 40,
        'request_rate': 100
    }

    # 1. 分析不同cache_nodes_ratio的影响
    print("\n" + "="*80)
    print("实验组1: 不同cache_nodes_ratio下的收敛性分析")
    print("="*80)
    print("基准参数: single_cache_capacity=40, request_rate=100")

    for ratio in cache_nodes_ratios:
        experiment_name = f'cache_nodes_ratio_{ratio}'
        experiment_dir = run_single_experiment(
            sweep_dir,
            experiment_name,
            cache_nodes_ratio=ratio,
            single_cache_capacity=base_params['single_cache_capacity'],
            request_rate=base_params['request_rate'],
            simulation_time=simulation_time
        )
        all_experiments.append({
            'group': 'cache_nodes_ratio',
            'name': experiment_name,
            'dir': experiment_dir,
            'cache_nodes_ratio': ratio,
            'single_cache_capacity': base_params['single_cache_capacity'],
            'request_rate': base_params['request_rate']
        })

    # 2. 分析不同single_cache_capacity的影响
    print("\n" + "="*80)
    print("实验组2: 不同single_cache_capacity下的收敛性分析")
    print("="*80)
    print("基准参数: cache_nodes_ratio=0.4, request_rate=100")

    for capacity in single_cache_capacities:
        experiment_name = f'single_cache_capacity_{capacity}'
        experiment_dir = run_single_experiment(
            sweep_dir,
            experiment_name,
            cache_nodes_ratio=base_params['cache_nodes_ratio'],
            single_cache_capacity=capacity,
            request_rate=base_params['request_rate'],
            simulation_time=simulation_time
        )
        all_experiments.append({
            'group': 'single_cache_capacity',
            'name': experiment_name,
            'dir': experiment_dir,
            'cache_nodes_ratio': base_params['cache_nodes_ratio'],
            'single_cache_capacity': capacity,
            'request_rate': base_params['request_rate']
        })

    # 3. 分析不同request_rate的影响
    print("\n" + "="*80)
    print("实验组3: 不同request_rate下的收敛性分析")
    print("="*80)
    print("基准参数: cache_nodes_ratio=0.4, single_cache_capacity=40")

    for rate in request_rates:
        experiment_name = f'request_rate_{rate}'
        experiment_dir = run_single_experiment(
            sweep_dir,
            experiment_name,
            cache_nodes_ratio=base_params['cache_nodes_ratio'],
            single_cache_capacity=base_params['single_cache_capacity'],
            request_rate=rate,
            simulation_time=simulation_time
        )
        all_experiments.append({
            'group': 'request_rate',
            'name': experiment_name,
            'dir': experiment_dir,
            'cache_nodes_ratio': base_params['cache_nodes_ratio'],
            'single_cache_capacity': base_params['single_cache_capacity'],
            'request_rate': rate
        })

    # 保存实验信息
    info = {
        'timestamp': timestamp,
        'base_dir': sweep_dir,
        'base_params': base_params,
        'parameter_ranges': {
            'cache_nodes_ratios': cache_nodes_ratios,
            'single_cache_capacities': single_cache_capacities,
            'request_rates': request_rates
        },
        'simulation_time': simulation_time,
        'total_experiments': len(all_experiments),
        'experiments': all_experiments
    }

    info_path = os.path.join(sweep_dir, 'experiment_info.yml')
    with open(info_path, 'w', encoding='utf-8') as f:
        yaml.dump(info, f, default_flow_style=False, allow_unicode=True)

    print("\n" + "="*80)
    print("参数扫描实验完成")
    print("="*80)
    print(f"\n实验目录: {sweep_dir}")
    print(f"实验总数: {len(all_experiments)}")
    print(f"\n实验信息已保存: {info_path}")
    print("\n下一步：")
    print(f"  python analyze_sca_admm_convergence.py {sweep_dir}")

    return sweep_dir

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='SCA_ADMM参数扫描实验')
    parser.add_argument('-o', '--output', default='results/parameter_sweep',
                       help='输出目录路径（默认: results/parameter_sweep）')

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    # 运行参数扫描
    sweep_dir = run_parameter_sweep(args.output)

    print(f"\n参数扫描实验已全部完成！")
    print(f"您现在可以运行收敛性分析：")
    print(f"  python analyze_sca_admm_convergence.py {sweep_dir}")