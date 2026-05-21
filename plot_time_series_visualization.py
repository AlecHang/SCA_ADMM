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
    'Q_learning': '#3182ce'
}

window_size = 15

csv_path = 'results/compare_results_100.csv'
df = pd.read_csv(csv_path)

methods = ['SCA_ADMM', 'SCA_neighborhood_search', 'Q_learning']
intervals = df['Interval'].unique()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ============= Hit Rate Time Series =============
ax1 = axes[0, 0]
for method in methods:
    method_data = df[df['Method'] == method].sort_values('Interval')
    intervals_arr = method_data['Interval'].values
    hit_rates = method_data['Hit_Rate'].values

    smoothed = pd.Series(hit_rates).rolling(window=window_size, min_periods=1).mean()

    ax1.plot(intervals_arr, hit_rates, alpha=0.2, color=colors[method], linewidth=0.5)
    ax1.plot(intervals_arr, smoothed, label=method, color=colors[method], linewidth=2)

ax1.set_xlabel('Interval')
ax1.set_ylabel('Hit Rate')
ax1.legend(loc='lower right')
ax1.set_ylim(0, 0.5)
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# ============= Latency Time Series =============
ax2 = axes[0, 1]
for method in methods:
    method_data = df[df['Method'] == method].sort_values('Interval')
    intervals_arr = method_data['Interval'].values
    latencies = method_data['Avg_Latency'].values

    smoothed = pd.Series(latencies).rolling(window=window_size, min_periods=1).mean()

    ax2.plot(intervals_arr, latencies, alpha=0.2, color=colors[method], linewidth=0.5)
    ax2.plot(intervals_arr, smoothed, label=method, color=colors[method], linewidth=2)

ax2.set_xlabel('Interval')
ax2.set_ylabel('Average Latency (ms)')
ax2.legend(loc='upper right')
ax2.grid(axis='y', linestyle='--', alpha=0.7)

# ============= Segmented Hit Rate (Confidence Interval) =============
ax3 = axes[1, 0]
n_segments = 15
segment_size = len(intervals) // n_segments

for method in methods:
    method_data = df[df['Method'] == method].sort_values('Interval')
    hit_rates = method_data['Hit_Rate'].values

    segment_means = []
    segment_stds = []
    segment_positions = []

    for i in range(n_segments):
        start = i * segment_size
        end = (i + 1) * segment_size if i < n_segments - 1 else len(hit_rates)
        segment_means.append(np.mean(hit_rates[start:end]))
        segment_stds.append(np.std(hit_rates[start:end]))
        segment_positions.append(start + segment_size // 2)

    segment_means = np.array(segment_means)
    segment_stds = np.array(segment_stds)

    ax3.errorbar(segment_positions, segment_means, yerr=segment_stds,
                 marker='s', capsize=5, label=method, color=colors[method], linewidth=2, markersize=8)

ax3.set_xlabel('Interval')
ax3.set_ylabel('Hit Rate')
ax3.legend(loc='lower right')
ax3.grid(axis='y', linestyle='--', alpha=0.7)

# ============= Segmented Latency (Confidence Interval) =============
ax4 = axes[1, 1]
for method in methods:
    method_data = df[df['Method'] == method].sort_values('Interval')
    latencies = method_data['Avg_Latency'].values

    segment_means = []
    segment_stds = []
    segment_positions = []

    for i in range(n_segments):
        start = i * segment_size
        end = (i + 1) * segment_size if i < n_segments - 1 else len(latencies)
        segment_means.append(np.mean(latencies[start:end]))
        segment_stds.append(np.std(latencies[start:end]))
        segment_positions.append(start + segment_size // 2)

    segment_means = np.array(segment_means)
    segment_stds = np.array(segment_stds)

    ax4.errorbar(segment_positions, segment_means, yerr=segment_stds,
                 marker='s', capsize=5, label=method, color=colors[method], linewidth=2, markersize=8)

ax4.set_xlabel('Interval')
ax4.set_ylabel('Average Latency (ms)')
ax4.legend(loc='upper right')
ax4.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('results/time_series_visualization_200.png', dpi=150, bbox_inches='tight')
plt.savefig('results/time_series_visualization_200.pdf', bbox_inches='tight')
print(f"Saved to results/time_series_visualization_200.png and .pdf")

# ============= Summary Statistics =============
print("\n" + "="*60)
print("Summary Statistics (Full Period)")
print("="*60)
for method in methods:
    method_data = df[df['Method'] == method]
    hr = method_data['Hit_Rate'].mean()
    lat = method_data['Avg_Latency'].mean()
    hr_std = method_data['Hit_Rate'].std()
    lat_std = method_data['Avg_Latency'].std()
    print(f"{method:25s}  Hit Rate: {hr:.4f} ± {hr_std:.4f}  Latency: {lat:.2f} ± {lat_std:.2f}")

# ============= EWMA Smoothed Trend =============
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

span = 30
ax1 = axes2[0]
for method in methods:
    method_data = df[df['Method'] == method].sort_values('Interval')
    hit_rates = method_data['Hit_Rate'].values
    ewm = pd.Series(hit_rates).ewm(span=span).mean()
    ax1.plot(ewm, label=method, color=colors[method], linewidth=2.5)

ax1.set_xlabel('Interval')
ax1.set_ylabel('Hit Rate (EWMA)')
ax1.set_title(f'Exponentially Weighted Moving Average (span={span})')
ax1.legend()
ax1.grid(axis='y', linestyle='--', alpha=0.7)

ax2 = axes2[1]
for method in methods:
    method_data = df[df['Method'] == method].sort_values('Interval')
    latencies = method_data['Avg_Latency'].values
    ewm = pd.Series(latencies).ewm(span=span).mean()
    ax2.plot(ewm, label=method, color=colors[method], linewidth=2.5)

ax2.set_xlabel('Interval')
ax2.set_ylabel('Average Latency (ms)')
ax2.set_title(f'Exponentially Weighted Moving Average (span={span})')
ax2.legend()
ax2.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('results/time_series_ewma_200.png', dpi=150, bbox_inches='tight')
plt.savefig('results/time_series_ewma_200.pdf', bbox_inches='tight')
print(f"\nSaved EWMA plot to results/time_series_ewma_200.png and .pdf")