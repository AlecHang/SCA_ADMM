# 参数扫描实验系统 - 并行化框架使用指南

## 概述

本系统为 `Cache-Allocation-Project-enhanced` 项目中的参数扫描实验提供了高效的并行化运行机制。通过多进程并行执行，可以显著提升实验速度，特别是在需要运行大量参数组合时。

## 主要特性

### 1. 核心功能
- **自动任务分配**: 根据参数空间自动生成和分配实验任务
- **配置文件隔离**: 每个并行任务使用独立的配置文件，避免数据冲突
- **进度监控**: 实时显示执行进度、预计剩余时间和任务状态
- **结果收集**: 自动收集和整合所有并行任务的执行结果
- **错误处理**: 完善的异常处理和错误报告机制

### 2. 性能优势
- **多进程并行**: 充分利用多核CPU资源
- **灵活的并行度**: 可根据硬件环境调整并行进程数
- **资源优化**: 智能的任务调度和资源管理
- **时间节省**: 相比串行执行，可节省50%-80%的执行时间

## 快速开始

### 基本使用

1. **运行参数扫描实验**:
   ```bash
   cd Cache-Allocation-Project-enhanced/Simulations
   python parameter_sweep_experiments.py
   ```

2. **选择执行模式**:
   - 串行模式：逐个执行实验，适合调试
   - 并行模式：同时执行多个实验，显著提升速度

3. **配置并行度**:
   - 系统会自动检测CPU核心数
   - 建议并行进程数不超过CPU核心数
   - 可根据实际情况调整

### 交互式操作

运行程序后，按照提示进行选择：

```
请选择拓扑类型:
1. GEANT
2. TISCALI
3. 两种拓扑都执行

请输入拓扑选择 (1-3): 1

请选择执行模式:
1. 串行模式（逐个执行实验，适合调试）
2. 并行模式（同时执行多个实验，显著提升速度）

请输入执行模式选择 (1-2): 2

当前系统CPU核心数: 8
请输入最大并行进程数 (默认=8，建议不超过CPU核心数): 4

请选择扫描类型:
1. 单缓存容量扫描 [10, 20, 30, 40, 50, 60, 70, 80]
2. 缓存节点比例扫描 [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
3. 模拟时间扫描 [25, 50, 75, 100, 125, 150, 175, 200]
4. 全部扫描

请输入扫描选择 (1-4): 1
```

## 详细说明

### 并行化框架架构

#### 1. 核心类

**ExperimentTask** - 实验任务数据类
```python
@dataclass
class ExperimentTask:
    task_id: str              # 任务唯一标识
    parameter_name: str       # 参数名称
    parameter_value: Any      # 参数值
    topology_type: Optional[str]  # 拓扑类型
    repeat_count: int         # 重复实验次数
    config_file: str          # 配置文件路径
    priority: int = 0         # 任务优先级
```

**TaskResult** - 任务结果数据类
```python
@dataclass
class TaskResult:
    task_id: str              # 任务唯一标识
    success: bool             # 是否成功
    parameter_name: str       # 参数名称
    parameter_value: Any      # 参数值
    topology_type: Optional[str]  # 拓扑类型
    execution_time: float     # 执行时间
    error_message: Optional[str] = None  # 错误信息
    result_file: Optional[str] = None   # 结果文件路径
```

**ParallelExperimentManager** - 并行实验管理器
- 管理任务队列和执行
- 控制进程池和资源分配
- 监控进度和收集结果
- 处理错误和异常

#### 2. 关键机制

**配置文件隔离**
- 每个并行任务创建独立的配置文件
- 避免多个任务同时修改同一配置文件
- 确保任务之间的数据独立性
- 执行完成后自动清理临时文件

**进程池管理**
- 使用 `ProcessPoolExecutor` 管理进程池
- 支持动态调整并行度
- 自动处理进程创建和销毁
- 确保资源正确释放

**进度监控**
- 实时显示进度条
- 计算预计剩余时间
- 显示已完成/总任务数
- 提供详细的执行统计

**结果收集**
- 自动收集所有任务结果
- 按参数值和方法分类整理
- 生成详细的统计报告
- 支持结果文件自动查找

### API 使用

#### 生成任务列表
```python
from parameter_sweep_experiments import generate_sweep_tasks

tasks = generate_sweep_tasks(
    config_file='config.yml',
    parameter_name='single_cache_capacity',
    parameter_values=[10, 20, 30, 40],
    topology_type='GEANT',
    N=1
)
```

#### 执行并行任务
```python
from parameter_sweep_experiments import ParallelExperimentManager

manager = ParallelExperimentManager(
    max_workers=4,        # 最大并行进程数
    enable_progress=True   # 启用进度显示
)

results = manager.execute_tasks(tasks)
```

#### 运行参数扫描
```python
from parameter_sweep_experiments import single_cache_capacity_sweep_parallel

single_cache_capacity_sweep_parallel(
    config_file='config.yml',
    topology_type='GEANT',
    N=1,                    # 重复实验次数
    max_workers=4,          # 最大并行进程数
    enable_progress=True    # 启用进度显示
)
```

### 参数扫描类型

#### 1. 单缓存容量扫描
- 参数范围: [10, 20, 30, 40, 50, 60, 70, 80]
- 测试不同缓存容量对性能的影响
- 函数: `single_cache_capacity_sweep_parallel()`

#### 2. 缓存节点比例扫描
- 参数范围: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
- 测试不同缓存节点比例对性能的影响
- 函数: `cache_nodes_ratio_sweep_parallel()`

#### 3. 模拟时间扫描
- 参数范围: [25, 50, 75, 100, 125, 150, 175, 200]
- 测试不同模拟时间对性能的影响
- 函数: `simulation_time_sweep_parallel()`

#### 4. 全部扫描
- 依次执行上述三种扫描
- 函数: `run_all_sweeps_parallel()`

## 性能优化建议

### 1. 并行度选择
- **推荐**: 使用CPU核心数的50%-75%
- **保守**: 使用CPU核心数的50%
- **激进**: 使用CPU核心数的100%（可能影响系统响应）

### 2. 内存管理
- 每个进程需要独立的内存空间
- 建议预留足够的系统内存
- 监控内存使用情况，避免OOM

### 3. 磁盘I/O
- 大量并行任务可能导致磁盘I/O瓶颈
- 考虑使用SSD提升性能
- 避免在机械硬盘上运行大规模并行任务

### 4. 任务调度
- 优先执行高优先级任务
- 合理安排任务执行顺序
- 避免长时间运行的任务阻塞其他任务

## 错误处理

### 常见错误及解决方案

#### 1. 配置文件冲突
**错误**: 多个任务同时修改配置文件
**解决**: 使用配置文件隔离机制（已自动实现）

#### 2. 内存不足
**错误**: 系统内存不足，任务失败
**解决**: 减少并行进程数或增加系统内存

#### 3. 进程超时
**错误**: 某些任务执行时间过长
**解决**: 增加超时时间或优化实验参数

#### 4. 结果文件丢失
**错误**: 无法找到生成的结果文件
**解决**: 检查结果目录权限和磁盘空间

## 测试和验证

### 运行测试套件
```bash
python test_parallel_sweep.py
```

测试内容包括:
- 任务生成功能
- 并行管理器初始化
- 配置文件隔离机制
- 小规模并行扫描

### 验证结果
1. 检查结果目录是否生成正确的CSV文件
2. 验证文件名是否包含正确的参数信息
3. 确认结果数据的完整性和准确性
4. 对比串行和并行执行的结果一致性

## 最佳实践

### 1. 开发阶段
- 使用串行模式进行调试
- 逐步增加并行度进行测试
- 验证结果的正确性

### 2. 生产环境
- 使用并行模式提升效率
- 监控系统资源使用情况
- 定期清理临时文件

### 3. 大规模实验
- 分批执行大量任务
- 使用日志记录执行过程
- 定期备份实验结果

### 4. 结果管理
- 建立清晰的结果文件命名规范
- 定期整理和归档实验结果
- 使用版本控制管理配置文件

## 技术细节

### 实现原理

1. **任务生成**
   - 根据参数空间生成任务列表
   - 每个任务包含完整的实验信息
   - 支持任务优先级设置

2. **配置隔离**
   - 为每个任务创建临时配置文件
   - 修改特定参数值
   - 确保任务之间的独立性

3. **进程管理**
   - 使用 `ProcessPoolExecutor` 创建进程池
   - 动态分配任务给空闲进程
   - 自动处理进程异常

4. **进度监控**
   - 实时更新进度条
   - 计算预计剩余时间
   - 显示详细统计信息

5. **结果收集**
   - 自动收集任务结果
   - 按参数分类整理
   - 生成统计报告

### 依赖项
- Python 3.7+
- multiprocessing
- concurrent.futures
- dataclasses
- pathlib
- yaml
- subprocess

## 扩展和定制

### 添加新的参数扫描类型

1. 定义参数范围
2. 创建任务生成函数
3. 实现并行扫描函数
4. 集成到主程序

### 自定义进度显示

继承 `ParallelExperimentManager` 并重写 `_update_progress()` 方法

### 添加新的结果分析

扩展 `analyze_sweep_results()` 函数以支持新的分析需求

## 故障排除

### 调试技巧
1. 使用串行模式重现问题
2. 检查临时配置文件内容
3. 查看详细的错误日志
4. 验证结果文件格式

### 日志分析
- 检查执行统计摘要
- 分析失败任务的原因
- 对比成功和失败任务的差异

## 总结

本并行化框架为参数扫描实验提供了高效、可靠、易用的解决方案。通过合理配置和使用，可以显著提升实验效率，同时保证结果的准确性和可重复性。

## 联系和支持

如有问题或建议，请通过以下方式联系：
- 提交Issue到项目仓库
- 查看项目文档和FAQ
- 参与社区讨论

---

**版本**: 1.0.0  
**最后更新**: 2024-04-28  
**作者**: Cache-Allocation-Project Team
