import pandas as pd
import matplotlib.pyplot as plt
import os

# 读取CSV文件
csv_path = 'results/interval_statistics_GEANT_time200_ratio0.4_cap40_20260428_171519.csv'

# 检查文件是否存在
if not os.path.exists(csv_path):
    print(f"错误：文件 {csv_path} 不存在")
    exit(1)

# 读取数据
df = pd.read_csv(csv_path)

# 确保interval和hit_rate列是数值类型
df['interval'] = pd.to_numeric(df['interval'], errors='coerce')
df['hit_rate'] = pd.to_numeric(df['hit_rate'], errors='coerce')

# 移除无效数据
df = df.dropna()

# 获取所有方法名称
methods = df['method'].unique()

# 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 6))

# 定义颜色和标记样式
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']

# 为每个方法绘制折线
for i, method in enumerate(methods):
    method_data = df[df['method'] == method]
    ax.plot(method_data['interval'], method_data['hit_rate'], 
            label=method, 
            color=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            markersize=3,
            linewidth=1.5,
            alpha=0.8)

# 设置图表标题和标签
ax.set_title('Cache Hit Rate Comparison Over Time', fontsize=14, fontweight='bold')
ax.set_xlabel('Time Interval', fontsize=12)
ax.set_ylabel('Hit Rate', fontsize=12)

# 设置坐标轴范围
ax.set_xlim(1, df['interval'].max())
ax.set_ylim(bottom=0)

# 添加图例
ax.legend(title='Methods', bbox_to_anchor=(1.05, 1), loc='upper left')

# 添加网格
ax.grid(True, linestyle='--', alpha=0.7)

# 自动调整布局
plt.tight_layout()

# 保存图像
output_path = 'results/hit_rate_comparison.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"图像已保存到 {output_path}")

# 显示图像
plt.show()