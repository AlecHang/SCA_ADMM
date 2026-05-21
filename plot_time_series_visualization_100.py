import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 11
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
    'global_opt_allocation': '#7c3aed',
    'SCA_gradient_tracking': '#63b3ed',
    'manual_allocation': '#f6ad55'
}

markers = {
    'SCA_ADMM': 'o',
    'SCA_neighborhood_search': '^',
    'Q_learning': 's',
    'best_allocation': 'D',
    'equal_allocation': 'v',
    'cooperative_best_allocation': '<',
    'proportional_allocation': '>',
    'global_opt_allocation': 'p',
    'SCA_gradient_tracking': '*',
    'manual_allocation': 'h'
}

method_name_map = {
    'SCA_neighborhood_search': 'SCA_NS',
    'best_allocation': 'local_best_allocation',
    'global_opt_allocation': 'global_allocation'
}

window_size = 15

csv_path = 'results/compare_results_100.csv'
df = pd.read_csv(csv_path)

methods = df['Method'].unique()
intervals = df['Interval'].unique()

method_order = [
    'SCA_ADMM',
    'SCA_neighborhood_search',
    'Q_learning',
    'global_opt_allocation',
    'best_allocation',
    'equal_allocation',
    'proportional_allocation'
]

methods = [m for m in method_order if m in df['Method'].unique()]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ============= Hit Rate Time Series =============
ax1 = axes[0]
for method in methods:
    method_data = df[df['Method'] == method].sort_values('Interval')
    intervals_arr = method_data['Interval'].values
    hit_rates = method_data['Hit_Rate'].values

    smoothed = pd.Series(hit_rates).rolling(window=window_size, min_periods=1).mean()

    ax1.plot(intervals_arr, hit_rates, alpha=0.15, color=colors.get(method, '#718096'), linewidth=0.5)
    ax1.plot(intervals_arr, smoothed, label=method_name_map.get(method, method), color=colors.get(method, '#718096'), linewidth=2)

ax1.set_xlabel('Interval')
ax1.set_ylabel('Hit Rate')
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(axis='y', linestyle='--', alpha=0.7)
ax1.text(0.5, -0.18, '(a)', transform=ax1.transAxes, fontsize=16, fontweight='bold', ha='center', va='top')

# ============= Latency Time Series =============
ax2 = axes[1]
for method in methods:
    method_data = df[df['Method'] == method].sort_values('Interval')
    intervals_arr = method_data['Interval'].values
    latencies = method_data['Avg_Latency'].values

    smoothed = pd.Series(latencies).rolling(window=window_size, min_periods=1).mean()

    ax2.plot(intervals_arr, latencies, alpha=0.15, color=colors.get(method, '#718096'), linewidth=0.5)
    ax2.plot(intervals_arr, smoothed, label=method_name_map.get(method, method), color=colors.get(method, '#718096'), linewidth=2)

ax2.set_xlabel('Interval')
ax2.set_ylabel('Average Latency (ms)')
ax2.legend(loc='lower right', fontsize=9)
ax2.grid(axis='y', linestyle='--', alpha=0.7)
ax2.text(0.5, -0.18, '(b)', transform=ax2.transAxes, fontsize=16, fontweight='bold', ha='center', va='top')

plt.tight_layout()
plt.savefig('results/time_series_visualization_100.png', dpi=150, bbox_inches='tight')
plt.savefig('results/time_series_visualization_100.pdf', dpi=150, bbox_inches='tight')
plt.close()

print(f"Saved to results/time_series_visualization_100.png and .pdf")

# ============= Summary Statistics =============
print("\n" + "="*70)
print("Summary Statistics (Full Period)")
print("="*70)
print(f"{'Method':<25} {'Hit Rate':<12} {'Std':<8} {'Latency':<12} {'Std':<8}")
print("-"*70)
for method in methods:
    method_data = df[df['Method'] == method]
    hr = method_data['Hit_Rate'].mean()
    lat = method_data['Avg_Latency'].mean()
    hr_std = method_data['Hit_Rate'].std()
    lat_std = method_data['Avg_Latency'].std()
    print(f"{method:<25} {hr:.4f}     {hr_std:.4f}  {lat:.2f}        {lat_std:.2f}")

# ============= EWMA Smoothed Trend =============
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

span = 30
ax1 = axes2[0]
for method in methods:
    method_data = df[df['Method'] == method].sort_values('Interval')
    hit_rates = method_data['Hit_Rate'].values
    ewm = pd.Series(hit_rates).ewm(span=span).mean()
    ax1.plot(ewm, label=method_name_map.get(method, method), color=colors.get(method, '#718096'), linewidth=2)

ax1.set_xlabel('Interval')
ax1.set_ylabel('Hit Rate (EWMA)')
ax1.legend(loc='lower right', bbox_to_anchor=(1.05, 0.5), fontsize=10)
ax1.grid(axis='y', linestyle='--', alpha=0.7)

ax2 = axes2[1]
for method in methods:
    method_data = df[df['Method'] == method].sort_values('Interval')
    latencies = method_data['Avg_Latency'].values
    ewm = pd.Series(latencies).ewm(span=span).mean()
    ax2.plot(ewm, label=method_name_map.get(method, method), color=colors.get(method, '#718096'), linewidth=2)

ax2.set_xlabel('Interval')
ax2.set_ylabel('Average Latency (ms)')
ax2.legend(loc='upper right', bbox_to_anchor=(1.05, 0.5), fontsize=10)
ax2.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('results/time_series_ewma_100.png', dpi=150, bbox_inches='tight')
plt.savefig('results/time_series_ewma_100.pdf', bbox_inches='tight')
