# 增强版缓存资源分配项目

这是一个基于 model-free 方法的缓存资源分配项目，具有以下增强功能：

## 主要特性

1. **配置驱动**：通过 `config.yml` 文件集中管理所有参数，无需修改代码即可调整模拟设置
2. **节点间时延模拟**：支持配置不同节点之间的网络时延，实现更真实的网络环境模拟
3. **模块化设计**：代码结构清晰，易于理解和扩展
4. **详细的结果输出**：生成成本和时延的图表，方便分析
5. **经验回放**：支持强化学习的经验回放机制，提高学习效率

## 项目结构

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

### 强化学习参数
- `gamma`：折扣因子
- `alpha`：学习率
- `epsilon`：探索率
- `epsilon_decay`：是否使用epsilon衰减
- `alpha_scheduling`：是否使用学习率调度
- `activate_memory`：是否使用经验回放

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

## 技术特点

1. **基于强化学习**：使用 Q-learning 或 SARSA 算法优化缓存分配
2. **网络时延模拟**：支持配置节点间的时延矩阵，模拟真实网络环境
3. **自适应参数**：支持 epsilon 衰减和学习率调度，提高学习效率
4. **经验回放**：使用经验回放机制，加速学习过程
5. **模块化设计**：代码结构清晰，易于扩展和修改

## 缓存分配方法

1. **平均分配** (`equal_allocation`): 将缓存容量平均分配给所有服务提供商。
2. **最佳分配** (`best_allocation`): 根据历史请求数据计算最佳缓存分配。
3. **SCA-ADMM** (`SCA_ADMM`): 使用SCA-ADMM算法进行缓存分配优化。
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

## 注意事项

- 缓存容量和视频数量应根据实际情况设置，避免内存不足
- 模拟时间和请求率会影响运行时间，建议先使用较小的值进行测试
- 节点数量应与服务提供商数量相匹配，以获得最佳模拟效果
