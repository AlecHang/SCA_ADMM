#!/usr/bin/env python3
"""
Compare moving average L1 changes for different dynamic allocation methods using real data results
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast
import os
import glob

def calculate_allocation_change(allocation1, allocation2):
    """
    Calculate L1 change magnitude between two allocations
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
    Load allocation file
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
    Compute L1 change sequence
    """
    l1_changes = []
    for i in range(1, len(allocations)):
        l1, _, _, _ = calculate_allocation_change(allocations[i-1], allocations[i])
        l1_changes.append(l1)
    return l1_changes

def compute_moving_average(data, window_size):
    """
    Compute moving window average
    """
    window_means = []
    intervals = []
    for i in range(len(data) - window_size + 1):
        window = data[i:i+window_size]
        window_means.append(np.mean(window))
        intervals.append(i + window_size // 2 + 2)
    return intervals, window_means

def load_method_data(input_dir, method_name):
    """
    Load data for specified method from directory
    """
    patterns = [
        f"*{method_name.lower()}*.txt",
        f"*{method_name}*.txt",
        f"*_{method_name}_*.txt"
    ]

    file_path = None
    for pattern in patterns:
        search_path = os.path.join(input_dir, pattern)
        files = glob.glob(search_path)
        if files:
            file_path = files[0]
            break

    if not file_path:
        print(f"✗ File not found for {method_name}")
        return None

    print(f"✓ Found {method_name} file: {os.path.basename(file_path)}")

    try:
        allocs = load_allocation_file(file_path)
        print(f"  {method_name} allocation count: {len(allocs)}")

        if len(allocs) < 2:
            print(f"✗ {method_name} insufficient data (need at least 2 allocations)")
            return None

        l1_changes = compute_l1_changes(allocs)
        print(f"  {method_name} change count: {len(l1_changes)}")

        data_len = len(l1_changes)
        window_size = min(20, max(3, data_len // 5))
        print(f"  {method_name} window size: {window_size}")

        if data_len < window_size:
            print(f"✗ {method_name} data insufficient ({data_len} < {window_size})")
            return None

        intervals, ma = compute_moving_average(l1_changes, window_size=window_size)
        print(f"  {method_name} moving average count: {len(ma)}")

        if len(ma) == 0:
            print(f"✗ {method_name} moving average result is empty")
            return None

        return {'intervals': intervals, 'ma': ma}

    except Exception as e:
        print(f"✗ Failed to load {method_name} data: {e}")
        return None

def plot_comparison(input_dir, output_file=None):
    """
    Plot moving window average L1 change comparison for different methods
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    colors = {
        'SCA_ADMM': "#e53e3e",
        'SCA_NS': '#38a169',
        'Q_learning': '#3182ce'
    }

    markers = {
        'SCA_ADMM': 'o',
        'SCA_NS': '^',
        'Q_learning': 's'
    }

    methods_data = {}

    print(f"\n{'='*80}")
    print(f"Loading data from directory: {input_dir}")
    print(f"{'='*80}")

    sca_admm_data = load_method_data(input_dir, 'SCA_ADMM')
    if sca_admm_data:
        methods_data['SCA_ADMM'] = sca_admm_data
        print("✓ Loaded SCA_ADMM data")

    sca_ns_data = load_method_data(input_dir, 'SCA_NS')
    if not sca_ns_data:
        sca_ns_data = load_method_data(input_dir, 'SCA_neighborhood_search')

    if sca_ns_data:
        methods_data['SCA_NS'] = sca_ns_data
        print("✓ Loaded SCA_NS data")

    q_learning_data = load_method_data(input_dir, 'Q_learning')
    if q_learning_data:
        methods_data['Q_learning'] = q_learning_data
        print("✓ Loaded Q_learning data")

    if not methods_data:
        print("\n✗ No method data loaded, please check input directory")
        return

    for method, data in methods_data.items():
        ax.plot(data['intervals'], data['ma'],
                label=method, color=colors[method],
                linewidth=2.5, alpha=0.8, marker=markers[method],
                markersize=4, markevery=10)

    ax.set_xlabel('Interval', fontsize=14, fontweight='bold')
    ax.set_ylabel('Moving Average L1 Change', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)

    max_interval = max(max(data['intervals']) for data in methods_data.values())
    ax.set_xlim(0, max_interval + 10)

    plt.tight_layout()

    if output_file:
        plot_filename = output_file
    else:
        plot_filename = os.path.join(input_dir, "stability_comparison_real_data.png")

    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {plot_filename}")

    print("\n" + "="*80)
    print("Moving Average L1 Change Comparison for Different Methods")
    print("="*80)
    for method, data in methods_data.items():
        mean_ma = np.mean(data['ma'])
        std_ma = np.std(data['ma'])
        min_ma = np.min(data['ma'])
        max_ma = np.max(data['ma'])
        print(f"{method:20s}: Mean={mean_ma:8.2f}, Std={std_ma:8.2f}, Min={min_ma:8.2f}, Max={max_ma:8.2f}")

    plt.close()

if __name__ == "__main__":
    input_dir = r"C:\Users\Admin\projects\cache\Cache-Allocation-Project-enhanced\results_real_data"
    plot_comparison(input_dir)