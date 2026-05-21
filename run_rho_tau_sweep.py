#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCA_ADMM rho和tau参数扫描实验
扫描不同rho和tau参数组合对SCA_ADMM性能的影响
其他参数使用config.yml配置
"""

import os
import sys
import yaml
import shutil
import csv
from datetime import datetime
import numpy as np
import re
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Simulations'))

import simulation_code as sc

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_config(config, config_path):
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

def run_single_experiment(rho, tau, config):
    """
    运行单次实验，返回平均命中率和时延
    """
    original_config_path = os.path.join('Simulations', 'config.yml')
    backup_config_path = original_config_path + '.rho_tau_backup'

    shutil.copy2(original_config_path, backup_config_path)

    try:
        config['output']['debug'] = False
        save_config(config, original_config_path)

        sc.init()
        sc.SCA_ADMM_RHO = rho
        sc.SCA_ADMM_TAU = tau

        total_cost, nominal_cost, cost_first, cost_best, avg_latency = sc.main(method='SCA_ADMM')

        if avg_latency and len(avg_latency) > 0:
            hit_rates = [1 - c for c in nominal_cost]
            return np.mean(hit_rates), np.std(hit_rates), np.mean(avg_latency), np.std(avg_latency)
        else:
            return None, None, None, None

    finally:
        shutil.move(backup_config_path, original_config_path)

def run_rho_tau_sweep(output_dir='results/rho_tau_sweep'):
    """
    执行rho和tau参数扫描
    """
    print("=" * 80)
    print("SCA_ADMM rho和tau参数扫描实验 (Zipf分布)")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    sweep_dir = os.path.join(output_dir, f'rho_tau_sweep_{timestamp}')
    os.makedirs(sweep_dir, exist_ok=True)

    config = load_config(os.path.join('Simulations', 'config.yml'))
    config['output']['save_allocations'] = False
    config['output']['save_results'] = True

    rho_values = [round(x * 0.05, 2) for x in range(1, 21)] #[round(x * 0.1, 1) for x in range(1, 11)]
    tau_values = [0.2] #[round(x * 0.1, 1) for x in range(1, 11)]
    
    #rho_values = [round(x * 0.05, 2) for x in range(1, 21)]
    #tau_values = [round(x * 0.02, 2) for x in range(1, 21)]

    results = []
    num_runs = 1

    total_experiments = len(rho_values) * len(tau_values)
    current_experiment = 0

    print(f"\n参数范围:")
    print(f"  rho: {rho_values}")
    print(f"  tau: {tau_values}")
    print(f"  总实验数: {total_experiments}")
    print(f"  每组重复次数: {num_runs}")

    pbar = tqdm(total=total_experiments, desc="参数扫描进度", unit="实验")

    for rho in rho_values:
        for tau in tau_values:
            current_experiment += 1
            print(f"\n[{current_experiment}/{total_experiments}] rho={rho}, tau={tau}")

            hit_rates = []
            latencies = []

            for run in range(num_runs):
                print(f"  运行 {run + 1}/{num_runs}...", end=' ')

                avg_hit_rate, std_hit_rate, avg_latency, std_latency = run_single_experiment(rho, tau, config)

                if avg_hit_rate is not None:
                    hit_rates.append(avg_hit_rate)
                    latencies.append(avg_latency)
                    print(f"命中率={avg_hit_rate:.4f}, 时延={avg_latency:.4f}")
                else:
                    print(f"失败")

            if hit_rates:
                results.append({
                    'rho': rho,
                    'tau': tau,
                    'avg_hit_rate': np.mean(hit_rates),
                    'std_hit_rate': np.std(hit_rates),
                    'avg_latency': np.mean(latencies),
                    'std_latency': np.std(latencies),
                    'num_runs': len(hit_rates)
                })

                print(f"  → rho={rho:.2f}, tau={tau:.2f}: 命中率={np.mean(hit_rates):.4f}±{np.std(hit_rates):.4f}, 时延={np.mean(latencies):.4f}±{np.std(latencies):.4f}")

            pbar.update(1)

    pbar.close()

    results_csv = os.path.join(sweep_dir, 'rho_tau_sweep_results.csv')
    with open(results_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['rho', 'tau', 'avg_hit_rate', 'std_hit_rate', 'avg_latency', 'std_latency', 'num_runs'])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{'=' * 80}")
    print("参数扫描完成")
    print(f"{'=' * 80}")
    print(f"结果已保存: {results_csv}")

    print("\n结果汇总表:")
    print("-" * 80)
    print(f"{'rho':<8} {'tau':<8} {'平均命中率':<15} {'平均时延':<15}")
    print("-" * 80)
    for r in results:
        print(f"{r['rho']:<8.2f} {r['tau']:<8.2f} {r['avg_hit_rate']:.4f}±{r['std_hit_rate']:.4f}   {r['avg_latency']:.4f}±{r['std_latency']:.4f}")
    print("-" * 80)

    create_heatmap(sweep_dir, results, rho_values, tau_values)

    return sweep_dir, results

def create_heatmap(sweep_dir, results, rho_values, tau_values):
    """
    创建rho-tau参数热力图
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        hit_rate_matrix = np.zeros((len(tau_values), len(rho_values)))
        latency_matrix = np.zeros((len(tau_values), len(rho_values)))

        for r in results:
            rho_idx = rho_values.index(r['rho'])
            tau_idx = tau_values.index(r['tau'])
            hit_rate_matrix[tau_idx, rho_idx] = r['avg_hit_rate']
            latency_matrix[tau_idx, rho_idx] = r['avg_latency']

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        im1 = axes[0].imshow(hit_rate_matrix, cmap='viridis', aspect='auto')
        axes[0].set_title('Cache Hit Rate')
        axes[0].set_xlabel('rho')
        axes[0].set_ylabel('tau')
        axes[0].set_xticks(range(len(rho_values)))
        axes[0].set_xticklabels([f'{r:.2f}' for r in rho_values])
        axes[0].set_yticks(range(len(tau_values)))
        axes[0].set_yticklabels([f'{t:.2f}' for t in tau_values])
        plt.colorbar(im1, ax=axes[0], label='Hit Rate')

        for i in range(len(tau_values)):
            for j in range(len(rho_values)):
                text = axes[0].text(j, i, f'{hit_rate_matrix[i, j]:.3f}',
                                   ha="center", va="center", color="w", fontsize=8)

        im2 = axes[1].imshow(latency_matrix, cmap='plasma', aspect='auto')
        axes[1].set_title('Average Latency')
        axes[1].set_xlabel('rho')
        axes[1].set_ylabel('tau')
        axes[1].set_xticks(range(len(rho_values)))
        axes[1].set_xticklabels([f'{r:.2f}' for r in rho_values])
        axes[1].set_yticks(range(len(tau_values)))
        axes[1].set_yticklabels([f'{t:.2f}' for t in tau_values])
        plt.colorbar(im2, ax=axes[1], label='Latency')

        for i in range(len(tau_values)):
            for j in range(len(rho_values)):
                text = axes[1].text(j, i, f'{latency_matrix[i, j]:.2f}',
                                   ha="center", va="center", color="w", fontsize=8)

        plt.tight_layout()
        heatmap_path = os.path.join(sweep_dir, 'rho_tau_heatmap.png')
        plt.savefig(heatmap_path, dpi=150)
        print(f"热力图已保存: {heatmap_path}")

    except Exception as e:
        print(f"创建热力图失败: {e}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='SCA_ADMM rho和tau参数扫描 (Zipf分布)')
    parser.add_argument('-o', '--output', default='results/rho_tau_sweep',
                       help='输出目录路径（默认: results/rho_tau_sweep）')

    args = parser.parse_args()

    sweep_dir, results = run_rho_tau_sweep(args.output)

    print(f"\n参数扫描完成！")
    print(f"结果目录: {sweep_dir}")