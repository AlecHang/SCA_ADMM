import os
import yaml
import random
import statistics
import csv
import subprocess
import sys
from datetime import datetime
import time
import shutil
import multiprocessing as mp
from multiprocessing import Pool, Manager, Queue, Lock
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import threading
import uuid

@dataclass
class ExperimentTask:
    """实验任务数据类"""
    task_id: str
    parameter_name: str
    parameter_value: Any
    topology_type: Optional[str]
    repeat_count: int
    config_file: str
    priority: int = 0

@dataclass
class TaskResult:
    """任务结果数据类"""
    task_id: str
    success: bool
    parameter_name: str
    parameter_value: Any
    topology_type: Optional[str]
    execution_time: float
    error_message: Optional[str] = None
    result_file: Optional[str] = None

class ParallelExperimentManager:
    """并行实验管理器 - 负责管理并行实验的执行和结果收集"""
    
    def __init__(self, max_workers: Optional[int] = None, enable_progress: bool = True):
        """
        初始化并行实验管理器
        
        Args:
            max_workers: 最大并行工作进程数，None表示使用CPU核心数
            enable_progress: 是否启用进度显示
        """
        self.max_workers = max_workers or mp.cpu_count()
        self.enable_progress = enable_progress
        self.task_queue: Queue = None
        self.result_queue: Queue = None
        self.lock: Lock = None
        self.completed_tasks: int = 0
        self.total_tasks: int = 0
        self.start_time: float = 0
        self.results: List[TaskResult] = []
        self.config_file_locks: Dict[str, Lock] = {}
        
    def _create_isolated_config(self, original_config: str, task: ExperimentTask) -> str:
        """
        为每个任务创建隔离的配置文件
        
        Args:
            original_config: 原始配置文件路径
            task: 实验任务
            
        Returns:
            隔离配置文件的路径（绝对路径）
        """
        # 使用绝对路径确保子进程能够找到文件
        original_config_path = Path(original_config).resolve()
        
        # 创建临时配置文件目录
        config_dir = original_config_path.parent / "temp_configs"
        config_dir.mkdir(exist_ok=True)
        
        # 生成唯一的配置文件名（避免使用可能有问题的字符）
        safe_task_id = task.task_id.replace(':', '_').replace('\\', '_').replace('/', '_')
        unique_id = f"{safe_task_id}_{uuid.uuid4().hex[:8]}"
        isolated_config = config_dir / f"config_{unique_id}.yml"
        
        # 复制原始配置文件
        shutil.copy2(str(original_config_path), str(isolated_config))
        
        # 修改配置文件
        self._modify_config_file(str(isolated_config), task.parameter_name, 
                                  task.parameter_value, task.topology_type)
        
        return str(isolated_config)
    
    def _modify_config_file(self, config_file: str, parameter_name: str, 
                           parameter_value: Any, topology_type: Optional[str] = None):
        """修改配置文件中的特定参数"""
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if parameter_name == 'single_cache_capacity':
            config['simulation'][parameter_name] = int(parameter_value)
        elif parameter_name == 'cache_nodes_ratio':
            config['topology'][parameter_name] = float(parameter_value)
        elif parameter_name == 'simulation_time':
            config['simulation']['time'] = int(parameter_value)
        elif parameter_name == 'topology_type':
            config['topology']['type'] = parameter_value
        else:
            raise ValueError(f"不支持的参数名: {parameter_name}")
        
        if topology_type:
            config['topology']['type'] = topology_type
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
    
    def _execute_single_task(self, task: ExperimentTask) -> TaskResult:
        """
        执行单个实验任务（在独立进程中运行）
        
        Args:
            task: 实验任务
            
        Returns:
            任务结果
        """
        start_time = time.time()
        
        try:
            # 创建隔离的配置文件
            isolated_config = self._create_isolated_config(task.config_file, task)
            
            # 执行实验（传递隔离的配置文件路径作为第一个参数）
            # 使用绝对路径确保工作目录正确
            config_dir = str(Path(task.config_file).resolve().parent)
            result = subprocess.run([
                sys.executable, 'repeated_experiments.py', 
                isolated_config,  # 传递隔离配置文件路径
                task.parameter_name, str(task.parameter_value), str(task.repeat_count)
            ], capture_output=True, text=True, cwd=config_dir)
            
            execution_time = time.time() - start_time
            
            if result.returncode == 0:
                # 查找生成的结果文件
                result_file = self._find_result_file(task)
                return TaskResult(
                    task_id=task.task_id,
                    success=True,
                    parameter_name=task.parameter_name,
                    parameter_value=task.parameter_value,
                    topology_type=task.topology_type,
                    execution_time=execution_time,
                    result_file=result_file
                )
            else:
                return TaskResult(
                    task_id=task.task_id,
                    success=False,
                    parameter_name=task.parameter_name,
                    parameter_value=task.parameter_value,
                    topology_type=task.topology_type,
                    execution_time=execution_time,
                    error_message=result.stderr
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            return TaskResult(
                task_id=task.task_id,
                success=False,
                parameter_name=task.parameter_name,
                parameter_value=task.parameter_value,
                topology_type=task.topology_type,
                execution_time=execution_time,
                error_message=str(e)
            )
    
    def _find_result_file(self, task: ExperimentTask) -> Optional[str]:
        """查找任务对应的结果文件"""
        results_dir = Path(task.config_file).parent.parent / "results"
        
        if not results_dir.exists():
            return None
        
        # 根据任务参数构建文件名模式
        param_prefix = {
            'single_cache_capacity': 'cap',
            'cache_nodes_ratio': 'ratio',
            'simulation_time': 'time'
        }.get(task.parameter_name, '')
        
        # 查找匹配的结果文件
        for file in results_dir.glob(f"repeated_experiments_*_{param_prefix}{task.parameter_value}_N{task.repeat_count}_*.csv"):
            if task.topology_type and task.topology_type in file.name:
                return str(file)
            elif not task.topology_type:
                return str(file)
        
        return None
    
    def _update_progress(self):
        """更新进度显示"""
        if not self.enable_progress:
            return
        
        elapsed = time.time() - self.start_time
        progress = self.completed_tasks / self.total_tasks if self.total_tasks > 0 else 0
        
        # 计算预计剩余时间
        if progress > 0:
            eta = elapsed / progress - elapsed
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))
        else:
            eta_str = "--:--:--"
        
        # 显示进度条
        bar_length = 40
        filled = int(bar_length * progress)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        print(f"\r进度: [{bar}] {progress*100:.1f}% ({self.completed_tasks}/{self.total_tasks}) "
              f"已用: {time.strftime('%H:%M:%S', time.gmtime(elapsed))} "
              f"预计剩余: {eta_str}", end='', flush=True)
    
    def _worker_callback(self, result: TaskResult):
        """任务完成回调函数"""
        self.results.append(result)
        self.completed_tasks += 1
        self._update_progress()
    
    def execute_tasks(self, tasks: List[ExperimentTask]) -> List[TaskResult]:
        """
        并行执行实验任务
        
        Args:
            tasks: 实验任务列表
            
        Returns:
            所有任务的结果列表
        """
        self.total_tasks = len(tasks)
        self.completed_tasks = 0
        self.start_time = time.time()
        self.results = []
        
        if self.total_tasks == 0:
            print("没有任务需要执行")
            return []
        
        print(f"\n开始并行执行 {self.total_tasks} 个实验任务")
        print(f"最大并行进程数: {self.max_workers}")
        print(f"{'='*80}")
        
        # 使用进程池并行执行任务
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_task = {
                executor.submit(self._execute_single_task, task): task 
                for task in tasks
            }
            
            # 等待所有任务完成
            for future in as_completed(future_to_task):
                try:
                    result = future.result()
                    self._worker_callback(result)
                except Exception as e:
                    task = future_to_task[future]
                    print(f"\n任务 {task.task_id} 执行失败: {e}")
        
        # 完成后换行
        if self.enable_progress:
            print()
        
        # 打印执行统计
        self._print_execution_summary()
        
        return self.results
    
    def _print_execution_summary(self):
        """打印执行统计摘要"""
        total_time = time.time() - self.start_time
        successful = sum(1 for r in self.results if r.success)
        failed = len(self.results) - successful
        
        print(f"\n{'='*80}")
        print("执行统计摘要")
        print(f"{'='*80}")
        print(f"总任务数: {len(self.results)}")
        print(f"成功: {successful}")
        print(f"失败: {failed}")
        print(f"总执行时间: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")
        print(f"平均任务时间: {total_time/len(self.results):.2f}秒" if self.results else "N/A")
        print(f"{'='*80}\n")

def backup_config(config_file):
    """
    备份配置文件
    返回备份文件的路径
    """
    backup_file = config_file + '.backup'
    shutil.copy2(config_file, backup_file)
    print(f"配置文件已备份: {backup_file}")
    return backup_file

def restore_config(config_file, backup_file):
    """
    从备份文件恢复配置文件
    """
    shutil.copy2(backup_file, config_file)
    print(f"配置文件已恢复: {config_file}")

def modify_config(config_file, parameter_name, parameter_value, topology_type=None):
    """
    修改配置文件中的特定参数
    """
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 根据参数名修改对应的配置项
    if parameter_name == 'single_cache_capacity':
        config['simulation'][parameter_name] = parameter_value
    elif parameter_name == 'cache_nodes_ratio':
        config['topology'][parameter_name] = parameter_value
    elif parameter_name == 'simulation_time':
        config['simulation']['time'] = parameter_value  # 修正：使用'time'而不是'simulation_time'
    elif parameter_name == 'topology_type':
        config['topology']['type'] = parameter_value
    else:
        raise ValueError(f"不支持的参数名: {parameter_name}")
    
    # 如果指定了拓扑类型，也更新拓扑类型
    if topology_type:
        config['topology']['type'] = topology_type
    
    # 保存修改后的配置
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
    
    print(f"配置已更新: {parameter_name} = {parameter_value}")
    if topology_type:
        print(f"拓扑类型已更新: {topology_type}")

def run_repeated_experiments(config_file, parameter_name, parameter_value, topology_type=None, N=10):
    """
    运行指定参数下的重复实验
    """

    
    try:
        # 修改配置文件
        modify_config(config_file, parameter_name, parameter_value, topology_type)
        
        # 在运行重复实验之前，打印关心的参数信息
        print(f"\n开始运行参数 {parameter_name}={parameter_value} 的重复实验...")
        
        # 读取当前配置以打印所有关心的参数
        with open(config_file, 'r', encoding='utf-8') as f:
            current_config = yaml.safe_load(f)
        
        # 打印关心的参数
        print("当前实验参数配置:")
        print(f"  拓扑类型: {current_config['topology']['type']}")
        print(f"  模拟时间: {current_config['simulation']['time']}秒")
        print(f"  单个缓存容量: {current_config['simulation']['single_cache_capacity']}")
        print(f"  缓存节点比例: {current_config['topology']['cache_nodes_ratio']}")
        print(f"  服务提供商数量: {current_config['providers']['count']}")
        print(f"  重复实验次数: {N}")
        
        # 使用subprocess运行repeated_experiments.py，并传递参数名、参数值和重复实验次数N
        # 确保repeated_experiments.py使用传入的参数值更新配置
        result = subprocess.run([
            sys.executable, 'repeated_experiments.py', parameter_name, str(parameter_value), str(N)
        ], capture_output=True, text=True, cwd=os.path.dirname(config_file))
        
        if result.returncode == 0:
            print(f"参数 {parameter_name}={parameter_value} 的实验完成")
            if topology_type:
                print(f"拓扑类型 {topology_type} 的实验完成")
            return True
        else:
            print(f"参数 {parameter_name}={parameter_value} 的实验失败")
            print(f"错误信息: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"运行实验时发生错误: {e}")
        return False

def analyze_sweep_results(results_dir, parameter_name, parameter_values, topology_type=None):
    """
    分析参数扫描实验结果
    支持新的子文件夹结构：results/{topology_type}/{scan_type}/
    """
    # 收集所有相关的结果文件
    result_files = []
    
    # 根据扫描类型确定子文件夹名
    scan_type_name = {
        'single_cache_capacity': 'single_cache_capacity',
        'cache_nodes_ratio': 'cache_nodes_ratio',
        'simulation_time': 'simulation_time'
    }.get(parameter_name, 'other')
    
    # 构建完整的结果目录路径（支持新的子文件夹结构）
    if topology_type:
        full_results_dir = os.path.join(results_dir, topology_type, scan_type_name)
    else:
        full_results_dir = results_dir
    
    # 检查目录是否存在
    if not os.path.exists(full_results_dir):
        # 如果新结构不存在，尝试旧结构（直接在results目录下）
        full_results_dir = results_dir
    
    # 遍历目录查找结果文件
    for file in os.listdir(full_results_dir):
        if file.startswith('repeated_experiments') and file.endswith('.csv'):
            # 检查文件名是否包含参数信息和拓扑类型
            file_matches = True
            
            # 检查拓扑类型
            if topology_type and topology_type not in file:
                file_matches = False
            
            # 检查参数值（适应新的文件名格式）
            if file_matches:
                for value in parameter_values:
                    # 适应新的文件名格式：包含time信息的文件名
                    # 新的格式：repeated_experiments_GEANT_time100_ratio0.4_cap30_N10_...
                    
                    # 检查参数值是否在文件名中（跳过time信息部分）
                    if parameter_name == 'single_cache_capacity':
                        # 查找cap{value}模式
                        if f"cap{value}" in file:
                            result_files.append(os.path.join(full_results_dir, file))
                            break
                    elif parameter_name == 'cache_nodes_ratio':
                        # 查找ratio{value}模式
                        if f"ratio{value}" in file:
                            result_files.append(os.path.join(full_results_dir, file))
                            break
                    elif parameter_name == 'simulation_time':
                        # 查找time{value}模式
                        if f"time{value}" in file:
                            result_files.append(os.path.join(full_results_dir, file))
                            break
                    else:
                        # 通用匹配
                        if str(value) in file:
                            result_files.append(os.path.join(full_results_dir, file))
                            break
    
    if not result_files:
        print(f"未找到参数 {parameter_name} 扫描的结果文件")
        if topology_type:
            print(f"拓扑类型: {topology_type}")
        return
    
    # 分析每个参数值的结果
    sweep_results = {}
    
    for result_file in result_files:
        with open(result_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            # 从文件名中提取参数值
            filename = os.path.basename(result_file)
            param_value = None
            for value in parameter_values:
                if f"{parameter_name[0:3]}{value}" in filename or str(value) in filename:
                    param_value = value
                    break
            
            if param_value is None:
                continue
            
            if param_value not in sweep_results:
                sweep_results[param_value] = {}
            
            # 读取每个方法的结果
            for row in reader:
                method = row['method']
                if method not in sweep_results[param_value]:
                    sweep_results[param_value][method] = {
                        'hit_rates': [],
                        'latencies': []
                    }
                
                sweep_results[param_value][method]['hit_rates'].append(float(row['hit_avg']))
                sweep_results[param_value][method]['latencies'].append(float(row['lat_avg']))
    
    # 打印参数扫描分析结果
    print(f"\n{'='*100}")
    print(f"参数扫描实验结果分析 - {parameter_name}")
    if topology_type:
        print(f"拓扑类型: {topology_type}")
    print(f"{'='*100}")
    
    # 获取所有方法名
    methods = set()
    for param_value, method_data in sweep_results.items():
        methods.update(method_data.keys())
    
    methods = sorted(list(methods))
    
    # 为每个方法打印结果
    for method in methods:
        print(f"\n方法: {method}")
        print(f"{parameter_name:<15} {'命中率':<15} {'时延':<15}")
        print("-" * 45)
        
        for param_value in sorted(sweep_results.keys()):
            if method in sweep_results[param_value]:
                data = sweep_results[param_value][method]
                hit_avg = statistics.mean(data['hit_rates']) if data['hit_rates'] else 0.0
                lat_avg = statistics.mean(data['latencies']) if data['latencies'] else 0.0
                print(f"{param_value:<15} {hit_avg:<15.4f} {lat_avg:<15.4f}")
    
    print(f"\n{'='*100}")

def generate_sweep_tasks(config_file: str, parameter_name: str, parameter_values: List[Any], 
                         topology_type: Optional[str], N: int) -> List[ExperimentTask]:
    """
    生成参数扫描的任务列表
    
    Args:
        config_file: 配置文件路径
        parameter_name: 参数名称
        parameter_values: 参数值列表
        topology_type: 拓扑类型
        N: 重复实验次数
        
    Returns:
        任务列表
    """
    tasks = []
    for i, value in enumerate(parameter_values):
        task_id = f"{parameter_name}_{value}_{i}"
        task = ExperimentTask(
            task_id=task_id,
            parameter_name=parameter_name,
            parameter_value=value,
            topology_type=topology_type,
            repeat_count=N,
            config_file=config_file,
            priority=i
        )
        tasks.append(task)
    return tasks

def single_cache_capacity_sweep_parallel(config_file, topology_type=None, N=10, 
                                          max_workers=None, enable_progress=True):
    """
    单缓存容量参数扫描（并行版本）
    
    Args:
        config_file: 配置文件路径
        topology_type: 拓扑类型
        N: 每个参数值的重复实验次数
        max_workers: 最大并行进程数
        enable_progress: 是否启用进度显示
    """
    # 备份原始配置文件
    backup_file = backup_config(config_file)
    
    try:
        print("\n" + "="*80)
        print("单缓存容量参数扫描实验（并行模式）")
        if topology_type:
            print(f"拓扑类型: {topology_type}")
        print("="*80)
        
        capacities = [10, 20, 30, 40, 50, 60, 70, 80]
        
        # 生成任务列表
        tasks = generate_sweep_tasks(config_file, 'single_cache_capacity', 
                                      capacities, topology_type, N)
        
        # 创建并行管理器并执行任务
        manager = ParallelExperimentManager(max_workers=max_workers, 
                                           enable_progress=enable_progress)
        results = manager.execute_tasks(tasks)
        
        # 检查是否有失败的任务
        failed_tasks = [r for r in results if not r.success]
        if failed_tasks:
            print(f"\n警告: {len(failed_tasks)} 个任务失败")
            for task in failed_tasks:
                print(f"  - {task.task_id}: {task.error_message}")
        
        # 分析结果
        results_dir = os.path.join(os.path.dirname(config_file), '..', 'results')
        analyze_sweep_results(results_dir, 'single_cache_capacity', capacities, topology_type)
    
    finally:
        # 恢复原始配置文件
        restore_config(config_file, backup_file)
        if os.path.exists(backup_file):
            os.remove(backup_file)
            print(f"备份文件已删除: {backup_file}")
        
        # 清理临时配置文件
        temp_config_dir = Path(config_file).parent / "temp_configs"
        if temp_config_dir.exists():
            shutil.rmtree(temp_config_dir)
            print(f"临时配置文件已清理")

def cache_nodes_ratio_sweep_parallel(config_file, topology_type=None, N=10, 
                                      max_workers=None, enable_progress=True):
    """
    缓存节点比例参数扫描（并行版本）
    
    Args:
        config_file: 配置文件路径
        topology_type: 拓扑类型
        N: 每个参数值的重复实验次数
        max_workers: 最大并行进程数
        enable_progress: 是否启用进度显示
    """
    # 备份原始配置文件
    backup_file = backup_config(config_file)
    
    try:
        print("\n" + "="*80)
        print("缓存节点比例参数扫描实验（并行模式）")
        if topology_type:
            print(f"拓扑类型: {topology_type}")
        print("="*80)
        
        ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        
        # 生成任务列表
        tasks = generate_sweep_tasks(config_file, 'cache_nodes_ratio', 
                                      ratios, topology_type, N)
        
        # 创建并行管理器并执行任务
        manager = ParallelExperimentManager(max_workers=max_workers, 
                                           enable_progress=enable_progress)
        results = manager.execute_tasks(tasks)
        
        # 检查是否有失败的任务
        failed_tasks = [r for r in results if not r.success]
        if failed_tasks:
            print(f"\n警告: {len(failed_tasks)} 个任务失败")
            for task in failed_tasks:
                print(f"  - {task.task_id}: {task.error_message}")
        
        # 分析结果
        results_dir = os.path.join(os.path.dirname(config_file), '..', 'results')
        analyze_sweep_results(results_dir, 'cache_nodes_ratio', ratios, topology_type)
    
    finally:
        # 恢复原始配置文件
        restore_config(config_file, backup_file)
        if os.path.exists(backup_file):
            os.remove(backup_file)
            print(f"备份文件已删除: {backup_file}")
        
        # 清理临时配置文件
        temp_config_dir = Path(config_file).parent / "temp_configs"
        if temp_config_dir.exists():
            shutil.rmtree(temp_config_dir)
            print(f"临时配置文件已清理")

def simulation_time_sweep_parallel(config_file, topology_type=None, N=10, 
                                    max_workers=None, enable_progress=True):
    """
    模拟时间参数扫描（并行版本）
    
    Args:
        config_file: 配置文件路径
        topology_type: 拓扑类型
        N: 每个参数值的重复实验次数
        max_workers: 最大并行进程数
        enable_progress: 是否启用进度显示
    """
    # 备份原始配置文件
    backup_file = backup_config(config_file)
    
    try:
        print("\n" + "="*80)
        print("模拟时间参数扫描实验（并行模式）")
        if topology_type:
            print(f"拓扑类型: {topology_type}")
        print("="*80)
        
        times = [25, 50, 75, 100, 125, 150, 175, 200]
        
        # 生成任务列表
        tasks = generate_sweep_tasks(config_file, 'simulation_time', 
                                      times, topology_type, N)
        
        # 创建并行管理器并执行任务
        manager = ParallelExperimentManager(max_workers=max_workers, 
                                           enable_progress=enable_progress)
        results = manager.execute_tasks(tasks)
        
        # 检查是否有失败的任务
        failed_tasks = [r for r in results if not r.success]
        if failed_tasks:
            print(f"\n警告: {len(failed_tasks)} 个任务失败")
            for task in failed_tasks:
                print(f"  - {task.task_id}: {task.error_message}")
        
        # 分析结果
        results_dir = os.path.join(os.path.dirname(config_file), '..', 'results')
        analyze_sweep_results(results_dir, 'simulation_time', times, topology_type)
    
    finally:
        # 恢复原始配置文件
        restore_config(config_file, backup_file)
        if os.path.exists(backup_file):
            os.remove(backup_file)
            print(f"备份文件已删除: {backup_file}")
        
        # 清理临时配置文件
        temp_config_dir = Path(config_file).parent / "temp_configs"
        if temp_config_dir.exists():
            shutil.rmtree(temp_config_dir)
            print(f"临时配置文件已清理")

def run_all_sweeps_parallel(config_file, topology_type, N=10, max_workers=None, enable_progress=True):
    """
    为指定拓扑类型运行所有参数扫描（并行版本）
    
    Args:
        config_file: 配置文件路径
        topology_type: 拓扑类型
        N: 每个参数值的重复实验次数
        max_workers: 最大并行进程数
        enable_progress: 是否启用进度显示
    """
    print(f"\n{'#'*100}")
    print(f"开始为拓扑类型 {topology_type} 执行所有参数扫描（并行模式）")
    print(f"{'#'*100}")
    
    # 运行三项参数扫描
    single_cache_capacity_sweep_parallel(config_file, topology_type, N, max_workers, enable_progress)
    cache_nodes_ratio_sweep_parallel(config_file, topology_type, N, max_workers, enable_progress)
    #simulation_time_sweep_parallel(config_file, topology_type, N, max_workers, enable_progress)
    
    print(f"\n{'#'*100}")
    print(f"拓扑类型 {topology_type} 的所有参数扫描完成")
    print(f"{'#'*100}")

def single_cache_capacity_sweep(config_file, topology_type=None, N=10):
    """
    单缓存容量参数扫描
    """
    # 备份原始配置文件（整个扫描过程）
    backup_file = backup_config(config_file)
    
    try:
        print("\n" + "="*80)
        print("单缓存容量参数扫描实验")
        if topology_type:
            print(f"拓扑类型: {topology_type}")
        print("="*80)
        
        capacities = [10, 20, 30, 40, 50, 60, 70, 80]
        
        for capacity in capacities:
            success = run_repeated_experiments(config_file, 'single_cache_capacity', capacity, topology_type, N)
            if not success:
                print(f"容量 {capacity} 的实验失败，跳过后续实验")
                break
            
            # 添加延迟以避免配置冲突
            time.sleep(2)
        
        # 分析结果
        results_dir = os.path.join(os.path.dirname(config_file), '..', 'results')
        analyze_sweep_results(results_dir, 'single_cache_capacity', capacities, topology_type)
    
    finally:
        # 恢复原始配置文件（整个扫描过程）
        restore_config(config_file, backup_file)
        # 删除备份文件
        if os.path.exists(backup_file):
            os.remove(backup_file)
            print(f"备份文件已删除: {backup_file}")

def cache_nodes_ratio_sweep(config_file, topology_type=None, N=10):
    """
    缓存节点比例参数扫描
    """
    # 备份原始配置文件（整个扫描过程）
    backup_file = backup_config(config_file)
    
    try:
        print("\n" + "="*80)
        print("缓存节点比例参数扫描实验")
        if topology_type:
            print(f"拓扑类型: {topology_type}")
        print("="*80)
        
        ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        
        for ratio in ratios:
            success = run_repeated_experiments(config_file, 'cache_nodes_ratio', ratio, topology_type, N)
            if not success:
                print(f"比例 {ratio} 的实验失败，跳过后续实验")
                break
            
            # 添加延迟以避免配置冲突
            time.sleep(2)
        
        # 分析结果
        results_dir = os.path.join(os.path.dirname(config_file), '..', 'results')
        analyze_sweep_results(results_dir, 'cache_nodes_ratio', ratios, topology_type)
    
    finally:
        # 恢复原始配置文件（整个扫描过程）
        restore_config(config_file, backup_file)
        # 删除备份文件
        if os.path.exists(backup_file):
            os.remove(backup_file)
            print(f"备份文件已删除: {backup_file}")

def simulation_time_sweep(config_file, topology_type=None, N=10):
    """
    模拟时间参数扫描
    """
    # 备份原始配置文件（整个扫描过程）
    backup_file = backup_config(config_file)
    
    try:
        print("\n" + "="*80)
        print("模拟时间参数扫描实验")
        if topology_type:
            print(f"拓扑类型: {topology_type}")
        print("="*80)
        
        times = [25, 50, 75, 100, 125, 150, 175, 200]
        
        for sim_time in times:
            success = run_repeated_experiments(config_file, 'simulation_time', sim_time, topology_type, N)
            if not success:
                print(f"时间 {sim_time} 的实验失败，跳过后续实验")
                break
            
            # 添加延迟以避免配置冲突
            time.sleep(2)
        
        # 分析结果
        results_dir = os.path.join(os.path.dirname(config_file), '..', 'results')
        analyze_sweep_results(results_dir, 'simulation_time', times, topology_type)
    
    finally:
        # 恢复原始配置文件（整个扫描过程）
        restore_config(config_file, backup_file)
        # 删除备份文件
        if os.path.exists(backup_file):
            os.remove(backup_file)
            print(f"备份文件已删除: {backup_file}")

def run_all_sweeps_for_topology(config_file, topology_type, N=10):
    """
    为指定拓扑类型运行所有参数扫描
    """
    print(f"\n{'#'*100}")
    print(f"开始为拓扑类型 {topology_type} 执行所有参数扫描")
    print(f"{'#'*100}")
    
    # 运行三项参数扫描
    single_cache_capacity_sweep(config_file, topology_type, N)
    cache_nodes_ratio_sweep(config_file, topology_type, N)
    simulation_time_sweep(config_file, topology_type, N)
    
    print(f"\n{'#'*100}")
    print(f"拓扑类型 {topology_type} 的所有参数扫描完成")
    print(f"{'#'*100}")

def main():
    """
    主函数
    """
    # 配置文件路径
    config_file = os.path.join(os.path.dirname(__file__), 'config.yml')
    
    if not os.path.exists(config_file):
        print(f"配置文件不存在: {config_file}")
        return
    
    # 实验参数
    N = 10  # 每个参数值的重复实验次数
    
    print("参数扫描实验系统")
    print("="*50)
    
    # 用户选择拓扑类型
    print("请选择拓扑类型:")
    print("1. GEANT")
    print("2. TISCALI")
    print("3. 两种拓扑都执行")
    
    topology_choice = input("请输入拓扑选择 (1-3): ").strip()
    
    if topology_choice not in ['1', '2', '3']:
        print("无效的拓扑选择")
        return
    
    # 用户选择执行模式
    print("\n请选择执行模式:")
    print("1. 串行模式（逐个执行实验，适合调试）")
    print("2. 并行模式（同时执行多个实验，显著提升速度）")
    
    mode_choice = input("请输入执行模式选择 (1-2): ").strip()
    
    if mode_choice not in ['1', '2']:
        print("无效的执行模式选择")
        return
    
    # 如果选择并行模式，询问并行度
    max_workers = None
    if mode_choice == '2':
        print(f"\n当前系统CPU核心数: {mp.cpu_count()}")
        workers_input = input(f"请输入最大并行进程数 (默认={mp.cpu_count()}，建议不超过CPU核心数): ").strip()
        if workers_input:
            try:
                max_workers = int(workers_input)
                if max_workers <= 0:
                    print("并行进程数必须大于0，将使用默认值")
                    max_workers = None
            except ValueError:
                print("无效的输入，将使用默认值")
                max_workers = None
    
    # 用户选择扫描类型
    print("\n请选择扫描类型:")
    print("1. 单缓存容量扫描 [10, 20, 30, 40, 50, 60, 70, 80]")
    print("2. 缓存节点比例扫描 [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]")
    print("3. 模拟时间扫描 [25, 50, 75, 100, 125, 150, 175, 200]")
    print("4. 全部扫描")
    
    sweep_choice = input("请输入扫描选择 (1-4): ").strip()
    
    if sweep_choice not in ['1', '2', '3', '4']:
        print("无效的扫描选择")
        return
    
    try:
        # 根据拓扑选择和执行模式执行相应的实验
        if mode_choice == '1':
            # 串行模式
            if topology_choice == '1':
                # GEANT拓扑
                if sweep_choice == '1':
                    single_cache_capacity_sweep(config_file, 'GEANT', N)
                elif sweep_choice == '2':
                    cache_nodes_ratio_sweep(config_file, 'GEANT', N)
                elif sweep_choice == '3':
                    simulation_time_sweep(config_file, 'GEANT', N)
                elif sweep_choice == '4':
                    run_all_sweeps_for_topology(config_file, 'GEANT', N)
                    
            elif topology_choice == '2':
                # TISCALI拓扑
                if sweep_choice == '1':
                    single_cache_capacity_sweep(config_file, 'TISCALI', N)
                elif sweep_choice == '2':
                    cache_nodes_ratio_sweep(config_file, 'TISCALI', N)
                elif sweep_choice == '3':
                    simulation_time_sweep(config_file, 'TISCALI', N)
                elif sweep_choice == '4':
                    run_all_sweeps_for_topology(config_file, 'TISCALI', N)
                    
            elif topology_choice == '3':
                # 两种拓扑都执行
                if sweep_choice == '1':
                    single_cache_capacity_sweep(config_file, 'GEANT', N)
                    single_cache_capacity_sweep(config_file, 'TISCALI', N)
                elif sweep_choice == '2':
                    cache_nodes_ratio_sweep(config_file, 'GEANT', N)
                    cache_nodes_ratio_sweep(config_file, 'TISCALI', N)
                elif sweep_choice == '3':
                    simulation_time_sweep(config_file, 'GEANT', N)
                    simulation_time_sweep(config_file, 'TISCALI', N)
                elif sweep_choice == '4':
                    run_all_sweeps_for_topology(config_file, 'GEANT', N)
                    run_all_sweeps_for_topology(config_file, 'TISCALI', N)
        
        else:
            # 并行模式
            if topology_choice == '1':
                # GEANT拓扑
                if sweep_choice == '1':
                    single_cache_capacity_sweep_parallel(config_file, 'GEANT', N, max_workers)
                elif sweep_choice == '2':
                    cache_nodes_ratio_sweep_parallel(config_file, 'GEANT', N, max_workers)
                elif sweep_choice == '3':
                    simulation_time_sweep_parallel(config_file, 'GEANT', N, max_workers)
                elif sweep_choice == '4':
                    run_all_sweeps_parallel(config_file, 'GEANT', N, max_workers)
                    
            elif topology_choice == '2':
                # TISCALI拓扑
                if sweep_choice == '1':
                    single_cache_capacity_sweep_parallel(config_file, 'TISCALI', N, max_workers)
                elif sweep_choice == '2':
                    cache_nodes_ratio_sweep_parallel(config_file, 'TISCALI', N, max_workers)
                elif sweep_choice == '3':
                    simulation_time_sweep_parallel(config_file, 'TISCALI', N, max_workers)
                elif sweep_choice == '4':
                    run_all_sweeps_parallel(config_file, 'TISCALI', N, max_workers)
                    
            elif topology_choice == '3':
                # 两种拓扑都执行
                if sweep_choice == '1':
                    single_cache_capacity_sweep_parallel(config_file, 'GEANT', N, max_workers)
                    single_cache_capacity_sweep_parallel(config_file, 'TISCALI', N, max_workers)
                elif sweep_choice == '2':
                    cache_nodes_ratio_sweep_parallel(config_file, 'GEANT', N, max_workers)
                    cache_nodes_ratio_sweep_parallel(config_file, 'TISCALI', N, max_workers)
                elif sweep_choice == '3':
                    simulation_time_sweep_parallel(config_file, 'GEANT', N, max_workers)
                    simulation_time_sweep_parallel(config_file, 'TISCALI', N, max_workers)
                elif sweep_choice == '4':
                    run_all_sweeps_parallel(config_file, 'GEANT', N, max_workers)
                    run_all_sweeps_parallel(config_file, 'TISCALI', N, max_workers)
            
    except KeyboardInterrupt:
        print("\n实验被用户中断")
    except Exception as e:
        print(f"实验执行出错: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n参数扫描实验完成")

if __name__ == "__main__":
    main()