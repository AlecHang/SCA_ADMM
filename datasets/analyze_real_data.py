#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析真实请求数据，生成流行度排名
支持数据筛选功能
"""

import os
import json
import yaml
import random
from collections import Counter

DATASET_DIR = r"C:\Users\Admin\projects\cache\Cache-Allocation-Project-enhanced\datasets"
OUTPUT_DIR = r"C:\Users\Admin\projects\cache\Cache-Allocation-Project-enhanced\Simulations\real_data"
CONFIG_FILE = r"C:\Users\Admin\projects\cache\Cache-Allocation-Project-enhanced\Simulations\config_real_data.yml"

def analyze_sp_data(sp_name, file_path, ratio=0.01):
    """分析单个SP的请求数据
    
    Args:
        sp_name: SP名称
        file_path: 文件路径
        ratio: 读取随机多少比例的数据 (默认0.01，即随机1%)
    """
    print(f"\n=== 分析 {sp_name} ===")
    print(f"读取文件: {file_path}")
    print(f"数据筛选比例: {ratio*100:.2f}%")

    content_counter = Counter()
    requests_list = []

    # 先读取所有行
    all_lines = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                all_lines.append(line.strip())
    
    total_lines = len(all_lines)
    
    # 计算要读取的行数（四舍五入）
    num_lines_to_read = max(1, round(total_lines * ratio))
    print(f"原始总请求数: {total_lines}")
    print(f"筛选后请求数: {num_lines_to_read}")

    # 随机抽样
    random.seed(42)  # 固定随机种子保证可重复性
    sampled_indices = random.sample(range(total_lines), num_lines_to_read)
    # 按原顺序处理抽样结果
    sampled_indices_sorted = sorted(sampled_indices)
    for idx in sampled_indices_sorted:
        content_id = all_lines[idx]
        content_counter[content_id] += 1
        requests_list.append(content_id)

    total_requests = sum(content_counter.values())
    unique_contents = len(content_counter)

    print(f"筛选后总请求数: {total_requests}")
    print(f"筛选后唯一内容数: {unique_contents}")

    sorted_contents = content_counter.most_common()

    print(f"\n最热门的前20个内容:")
    for i, (content_id, count) in enumerate(sorted_contents[:20]):
        print(f"  {i+1}. {content_id}: {count} 次 ({count/total_requests*100:.2f}%)")

    return sorted_contents, total_requests, unique_contents, total_lines, requests_list

def main(ratio=0.01):
    """主函数
    
    Args:
        ratio: 数据筛选比例 (默认0.01，即前1%)
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    sp_files = {
        'youtube': os.path.join(DATASET_DIR, 'youtube-request.txt'),
        'netflix': os.path.join(DATASET_DIR, 'netflix-request.txt'),
        'douyin': os.path.join(DATASET_DIR, 'douyin-request.txt')
    }

    sp_rankings = {}
    sp_stats = {}
    sp_requests = {}

    for sp_name, file_path in sp_files.items():
        if os.path.exists(file_path):
            sorted_contents, total_requests, unique_contents, original_total, requests_list = analyze_sp_data(sp_name, file_path, ratio)
            sp_rankings[sp_name] = sorted_contents
            sp_stats[sp_name] = {
                'original_total_requests': original_total,
                'total_requests': total_requests,
                'unique_contents': unique_contents,
                'ratio_used': ratio
            }
            sp_requests[sp_name] = requests_list
        else:
            print(f"文件不存在: {file_path}")

    print("\n" + "="*60)
    print("总体统计")
    print("="*60)

    total_original = sum(stats['original_total_requests'] for stats in sp_stats.values())
    total_filtered = sum(stats['total_requests'] for stats in sp_stats.values())
    print(f"原始总请求数: {total_original}")
    print(f"筛选后总请求数: {total_filtered}")

    for sp_name, stats in sp_stats.items():
        print(f"  {sp_name}: {stats['total_requests']} 请求 (原始: {stats['original_total_requests']}), {stats['unique_contents']} 唯一内容")

    ranking_file = os.path.join(OUTPUT_DIR, 'sp_popularity_rankings.json')
    with open(ranking_file, 'w', encoding='utf-8') as f:
        json.dump({sp: list(rankings) for sp, rankings in sp_rankings.items()}, f, ensure_ascii=False, indent=2)

    print(f"\n流行度排名已保存到: {ranking_file}")

    stats_file = os.path.join(OUTPUT_DIR, 'sp_statistics.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(sp_stats, f, indent=2)

    print(f"统计数据已保存到: {stats_file}")

    requests_file = os.path.join(OUTPUT_DIR, 'sp_requests.json')
    with open(requests_file, 'w', encoding='utf-8') as f:
        json.dump(sp_requests, f, ensure_ascii=False, indent=2)

    print(f"请求数据已保存到: {requests_file}")

    return sp_rankings, sp_stats, sp_requests

def load_config():
    """从配置文件读取参数"""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    else:
        print(f"配置文件不存在: {CONFIG_FILE}")
        return None

if __name__ == "__main__":
    import sys
    # 从配置文件读取 data_ratio（优先从 real_data，其次从 simulation）
    config = load_config()
    ratio = 0.01  # 默认值
    if config:
        if 'real_data' in config and 'data_ratio' in config['real_data']:
            ratio = config['real_data']['data_ratio']
            print(f"从配置文件 real_data.data_ratio 读取数据筛选比例: {ratio*100:.2f}%")
        elif 'simulation' in config and 'data_ratio' in config['simulation']:
            ratio = config['simulation']['data_ratio']
            print(f"从配置文件 simulation.data_ratio 读取数据筛选比例: {ratio*100:.2f}%")
        else:
            print(f"未在配置文件找到 data_ratio，使用默认值: {ratio*100:.2f}%")
    else:
        print(f"未读取到配置文件，使用默认值: {ratio*100:.2f}%")
    
    # 也支持命令行参数覆盖
    if len(sys.argv) > 1:
        try:
            ratio = float(sys.argv[1])
            print(f"使用命令行参数覆盖数据筛选比例: {ratio*100:.2f}%")
        except ValueError:
            print(f"无效的命令行参数: {sys.argv[1]}，保持配置文件的值")
    
    main(ratio=ratio)