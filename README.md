SCA_ADMM



```
Cache-Allocation-Project-enhanced/
├── Simulations/
│   ├── config.yml          # 配置文件
│   ├── simulation_code.py  # 主要模拟代码
│   ├── Auxiliary_functions.py  # 辅助函数
│   ├── run.sh              # 运行脚本
│   ├── results/            # 结果保存目录
│   └── figures/            # 图表保存目录
└── README.md               # 项目说明
```

## 配置说明

主要配置文件为 `Simulations/config.yml`，包含以下部分：

### 基本模拟参数
- `simulation_time`：模拟时间（秒）
- `cache_capacity`：缓存容量
- `bin_size`：每个bin的大小
- `nb_videos`：每个CP的视频数量
- `request_rate`：每秒请求数
- `interval_size`：间隔大小（秒）
- `delta`：分配修改的步长
- `method`：优化方法（SARSA, Q_learning, R_learning）
- `D`：delta的系数集合

### 服务提供商参数
- `count`：服务提供商数量
- `probabilities`：每个SP的请求概率
- `cacheability`：每个SP的可缓存比例
- `zipf_alphas`：每个SP的Zipf分布参数


### 节点间时延配置
- `enabled`：是否启用网络时延模拟
- `nodes`：节点数量
- `latency_matrix`：节点间时延矩阵（毫秒）
- `bandwidth`：带宽（Mbps）

## 使用方法

1. **修改配置**：根据需要编辑 `config.yml` 文件
2. **运行模拟**：
   ```bash
   cd Simulations
   python simulation_code.py
   ```
   或使用运行脚本：
   ```bash
   cd Simulations
   ./run.sh
   ```

3. **查看结果**：
   - 结果数据保存在 `results/` 目录
   - 图表保存在 `figures/` 目录

## 输出说明

- **分配历史**：`allocations_*.txt` 文件记录了每次迭代的缓存分配情况
- **结果数据**：`results_*.csv` 文件包含成本和时延数据
- **成本图表**：`cost_evolution_*.png` 显示成本随时间的变化
- **时延图表**：`latency_evolution_*.png` 显示时延随时间的变化（仅当启用网络模拟时）

## 缓存分配方法

1. **平均分配** (`equal_allocation`): 将缓存容量平均分配给所有服务提供商。
2. **最佳分配** (`best_allocation`): 根据历史请求数据计算最佳缓存分配。
3. **SCA_ADMM** (`SCA_ADMM`): 使用SCA-ADMM算法进行缓存分配优化。
4. **基于请求概率的分配** (`proportional_allocation`): 根据服务提供商的请求概率进行比例分配。

### 基于请求概率的分配方法

基于请求概率的分配方法是一种简单而有效的缓存分配策略，它根据服务提供商的请求概率来分配缓存空间。具体步骤如下：

1. 计算每个服务提供商的请求概率占比。
2. 根据请求概率占比，将每个缓存节点的容量按比例分配给各个服务提供商。
3. 处理分配过程中的余数，确保总分配量等于缓存节点的容量。
4. 确保所有分配都是非负的。

这种方法的优点是计算简单，执行速度快，并且能够根据服务提供商的请求频率自动调整分配比例。

## 示例配置

默认配置模拟了一个包含3个服务提供商和3个节点的场景，节点间有时延差异。您可以根据需要修改配置文件以模拟不同的场景。

## 依赖项

- Python 3.6+
- numpy
- pandas
- matplotlib
- pyyaml
