#!/usr/bin/env python3
"""
Plot boxplots for hit rate and latency from repeated experiment results
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import glob

results_dir = r'C:\Users\Admin\projects\cache\Cache-Allocation-Project-enhanced\Simulations\results\GEANT\real_data'

pattern = os.path.join(results_dir, 'repeated_experiments_*_GEANT_100_cap80_N200_*.csv')
files = glob.glob(pattern)

method_data = {}

for file_path in files:
    filename = os.path.basename(file_path)

    if 'summary' in filename:
        continue

    parts = filename.split('_')
    method_parts = []
    for i, part in enumerate(parts):
        if part == 'GEANT':
            break
        if i >= 2:
            method_parts.append(part)

    method = '_'.join(method_parts)

    if method not in method_data:
        df = pd.read_csv(file_path)
        method_data[method] = {
            'latency': df['avg_latency'].tolist(),
            'hit_rate': df['cache_hit_rate'].tolist()
        }

method_order = ['SCA_ADMM', 'SCA_neighborhood_search', 'Q_learning',
                'global_opt_allocation', 'best_allocation', 'equal_allocation',
                'proportional_allocation']

method_name_map = {
    'SCA_neighborhood_search': 'SCA_NS',
    'best_allocation': 'local_best_allocation',
    'global_opt_allocation': 'global_allocation'
}

colors = {
    'SCA_ADMM': '#e53e3e',        # 红色 (与 plot_stability_compare_moving_average.py 一致)
    'SCA_neighborhood_search': '#38a169',  # 绿色 (与 plot_stability_compare_moving_average.py 的 SCA_NS 一致)
    'Q_learning': '#3182ce',      # 蓝色 (与 plot_stability_compare_moving_average.py 一致)
    'global_opt_allocation': '#ed8936',   # 橙色
    'best_allocation': '#7c3aed',         # 紫色
    'equal_allocation': '#ecc94b',         # 黄色
    'proportional_allocation': '#4a5568',  # 灰色
    'cooperative_best_allocation': '#fc8181' # 浅红色
}

print(f"Found {len(method_data)} methods: {list(method_data.keys())}")

# Prepare data
latency_box_data = []
hitrate_box_data = []
method_labels = []
method_colors = []

for method in method_order:
    if method in method_data:
        latency_box_data.append(method_data[method]['latency'])
        hitrate_box_data.append(method_data[method]['hit_rate'])
        display_name = method_name_map.get(method, method)
        method_labels.append(display_name.replace('_', '\n'))
        method_colors.append(colors.get(method, '#1f77b4'))

# Create combined figure with 1 row and 2 columns
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 7))

# ============= Plot 1: Hit Rate =============
ax1 = axes[0]
print(f"Plotting {len(hitrate_box_data)} methods for hit rate")
bp1 = ax1.boxplot(hitrate_box_data, vert=True, patch_artist=True)
ax1.set_xticklabels(method_labels, rotation=45, fontsize=10, ha='right')
for patch, color in zip(bp1['boxes'], method_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
for median in bp1['medians']:
    median.set_color('black')
    median.set_linewidth(2)

ax1.set_ylabel('Cache Hit Rate', fontsize=14)
ax1.grid(axis='y', linestyle='--', alpha=0.7)
ax1.text(0.5, -0.25, '(a)', transform=ax1.transAxes, fontsize=16, fontweight='bold', ha='center', va='top')

# ============= Plot 2: Latency =============
ax2 = axes[1]
print(f"Plotting {len(latency_box_data)} methods for latency")
bp2 = ax2.boxplot(latency_box_data, vert=True, patch_artist=True)
ax2.set_xticklabels(method_labels, rotation=45, fontsize=10, ha='right')
for patch, color in zip(bp2['boxes'], method_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
for median in bp2['medians']:
    median.set_color('black')
    median.set_linewidth(2)

ax2.set_ylabel('Average Latency (ms)', fontsize=14)
ax2.grid(axis='y', linestyle='--', alpha=0.7)
ax2.text(0.5, -0.25, '(b)', transform=ax2.transAxes, fontsize=16, fontweight='bold', ha='center', va='top')

plt.subplots_adjust(bottom=0.30, wspace=0.3)

combined_output = os.path.join(results_dir, 'boxplot_combined.png')
plt.savefig(combined_output, dpi=300, bbox_inches='tight')
print(f"Combined boxplot saved to: {combined_output}")
plt.close()

print("\nData Summary:")
print("-" * 70)
print(f"{'Method':<30} {'Latency (ms)':<18} {'Hit Rate':<15}")
print("-" * 70)
for method in method_order:
    if method in method_data:
        lat_data = method_data[method]['latency']
        hr_data = method_data[method]['hit_rate']
        lat_mean = sum(lat_data) / len(lat_data)
        hr_mean = sum(hr_data) / len(hr_data)
        lat_std = (sum((x - lat_mean)**2 for x in lat_data) / len(lat_data)) ** 0.5
        hr_std = (sum((x - hr_mean)**2 for x in hr_data) / len(hr_data)) ** 0.5
        print(f"{method:<30} {lat_mean:.4f} +/- {lat_std:.4f}   {hr_mean:.4f} +/- {hr_std:.4f}")