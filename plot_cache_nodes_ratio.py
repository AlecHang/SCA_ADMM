import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.7

colors = {
    'SCA_ADMM': '#e53e3e',
    'SCA_neighborhood_search': '#38a169',
    'Q_learning': '#3182ce',
    'best_allocation': '#ed8936',
    'equal_allocation': '#ecc94b',
    'cooperative_best_allocation': '#fc8181',
    'proportional_allocation': '#4a5568',
    'global_opt_allocation': '#7c3aed'
}

# 方法名称显示映射
method_name_map = {
    'SCA_neighborhood_search': 'SCA_NS',
    'best_allocation': 'local_best_allocation',
    'global_opt_allocation': 'global_allocation'
}

markers = {
    'SCA_ADMM': 'o',
    'SCA_neighborhood_search': '^',
    'Q_learning': 's',
    'best_allocation': 'D',
    'equal_allocation': 'v',
    #'cooperative_best_allocation': '<',
    'proportional_allocation': '>',
    'global_opt_allocation': 'p'
}

TOPO = 'TISCALI'

# ============= Read Data =============
data_dir = f'results/{TOPO}/cache_nodes_ratio'
file_pattern = os.path.join(data_dir, '*.csv')
csv_files = sorted(glob.glob(file_pattern))

all_data = []
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    all_data.append(df)

full_df = pd.concat(all_data, ignore_index=True)

print(f"Read {len(csv_files)} files, total {len(full_df)} rows")
print(f"Methods: {full_df['method'].unique()}")
print(f"Ratios: {sorted(full_df['cache_nodes_ratio'].unique())}")

# 手动设定方法显示顺序（按算法类型和性能排序）
method_order = [
    'SCA_ADMM',
    'SCA_neighborhood_search',
    'Q_learning',
    #'cooperative_best_allocation',
    'global_opt_allocation',
    'best_allocation',
    'equal_allocation',
    'proportional_allocation'
]

# ============= Plot 1: Hit Rate vs Cache Nodes Ratio (Envelope) =============
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
methods = method_order  # 使用手动设定的顺序

for method in methods:
    method_data = full_df[full_df['method'] == method].sort_values('cache_nodes_ratio')
    ratios = method_data['cache_nodes_ratio'].values
    hit_avgs = method_data['hit_avg'].values
    hit_min = method_data['hit_min'].values
    hit_max = method_data['hit_max'].values

    color = colors.get(method, '#718096')
    ax1.fill_between(ratios, hit_min, hit_max, alpha=0.15, color=color)
    ax1.plot(ratios, hit_avgs, marker=markers.get(method, 'o'), 
             label=method_name_map.get(method, method), color=color, linewidth=2.5, markersize=8)

ax1.set_xlabel('Cache Nodes Ratio')
ax1.set_ylabel('Average Hit Rate')
ax1.legend(loc='lower right', fontsize=9)
ax1.grid(axis='y', linestyle='--', alpha=0.7)
ax1.text(0.5, -0.18, '(a)', transform=ax1.transAxes, fontsize=16, fontweight='bold', ha='center', va='top')

# ============= Plot 2: Latency vs Cache Nodes Ratio (Envelope) =============
ax2 = axes[1]

for method in methods:
    method_data = full_df[full_df['method'] == method].sort_values('cache_nodes_ratio')
    ratios = method_data['cache_nodes_ratio'].values
    lat_avgs = method_data['lat_avg'].values
    lat_min = method_data['lat_min'].values
    lat_max = method_data['lat_max'].values

    color = colors.get(method, '#718096')
    ax2.fill_between(ratios, lat_min, lat_max, alpha=0.15, color=color)
    ax2.plot(ratios, lat_avgs, marker=markers.get(method, 'o'),
             label=method_name_map.get(method, method), color=color, linewidth=2.5, markersize=8)

ax2.set_xlabel('Cache Nodes Ratio')
ax2.set_ylabel('Average Latency (ms)')
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(axis='y', linestyle='--', alpha=0.7)
ax2.text(0.5, -0.18, '(b)', transform=ax2.transAxes, fontsize=16, fontweight='bold', ha='center', va='top')

plt.tight_layout()
plt.savefig(f'results/{TOPO}/cache_nodes_ratio_comparison.png', dpi=150, bbox_inches='tight')
plt.savefig(f'results/{TOPO}/cache_nodes_ratio_comparison.pdf', bbox_inches='tight')
print(f"\nSaved to results/cache_nodes_ratio_comparison.png and .pdf")

# ============= Plot 3: Focus on key methods (Envelope) =============
key_methods = ['SCA_ADMM', 'SCA_neighborhood_search', 'Q_learning', 'best_allocation', 'equal_allocation']

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

ax3 = axes2[0]
for method in key_methods:
    method_data = full_df[full_df['method'] == method].sort_values('cache_nodes_ratio')
    ratios = method_data['cache_nodes_ratio'].values
    hit_avgs = method_data['hit_avg'].values
    hit_min = method_data['hit_min'].values
    hit_max = method_data['hit_max'].values

    color = colors.get(method, '#718096')
    ax3.fill_between(ratios, hit_min, hit_max, alpha=0.2, color=color)
    ax3.plot(ratios, hit_avgs, marker=markers.get(method, 'o'),
             label=method_name_map.get(method, method), color=color, linewidth=2.5, markersize=8)

ax3.set_xlabel('Cache Nodes Ratio')
ax3.set_ylabel('Average Hit Rate')
ax3.legend(loc='lower right', fontsize=9)
ax3.grid(axis='y', linestyle='--', alpha=0.7)
ax3.text(0.5, -0.18, '(c)', transform=ax3.transAxes, fontsize=16, fontweight='bold', ha='center', va='top')

ax4 = axes2[1]
for method in key_methods:
    method_data = full_df[full_df['method'] == method].sort_values('cache_nodes_ratio')
    ratios = method_data['cache_nodes_ratio'].values
    lat_avgs = method_data['lat_avg'].values
    lat_min = method_data['lat_min'].values
    lat_max = method_data['lat_max'].values

    color = colors.get(method, '#718096')
    ax4.fill_between(ratios, lat_min, lat_max, alpha=0.2, color=color)
    ax4.plot(ratios, lat_avgs, marker=markers.get(method, 'o'),
             label=method_name_map.get(method, method), color=color, linewidth=2.5, markersize=8)

ax4.set_xlabel('Cache Nodes Ratio')
ax4.set_ylabel('Average Latency (ms)')
ax4.legend(loc='upper right', fontsize=9)
ax4.grid(axis='y', linestyle='--', alpha=0.7)
ax4.text(0.5, -0.18, '(d)', transform=ax4.transAxes, fontsize=16, fontweight='bold', ha='center', va='top')

plt.tight_layout()
plt.savefig(f'results/{TOPO}/cache_nodes_ratio_key_methods.png', dpi=150, bbox_inches='tight')
plt.savefig(f'results/{TOPO}/cache_nodes_ratio_key_methods.pdf', bbox_inches='tight')
print(f"Saved key methods plot to results/cache_nodes_ratio_key_methods.png and .pdf")

# ============= Print Summary Statistics =============
print("\n" + "="*80)
print("Summary Statistics by Cache Nodes Ratio")
print("="*80)

ratios_sorted = sorted(full_df['cache_nodes_ratio'].unique())
for ratio in ratios_sorted:
    ratio_data = full_df[full_df['cache_nodes_ratio'] == ratio].sort_values('hit_avg', ascending=False)
    print(f"\nRatio: {ratio}")
    print(f"{'Method':<25} {'Hit Rate':<10} {'Latency':<10}")
    print("-"*45)
    for _, row in ratio_data.iterrows():
        print(f"{row['method']:<25} {row['hit_avg']:.4f}      {row['lat_avg']:.2f}")