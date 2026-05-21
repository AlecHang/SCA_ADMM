"""
并行化参数扫描实验 - 使用示例
演示如何使用并行化框架运行参数扫描实验
"""
import sys
import os

# 添加Simulations目录到路径
sys.path.insert(0, os.path.dirname(__file__))

from parameter_sweep_experiments import (
    ExperimentTask,
    ParallelExperimentManager,
    generate_sweep_tasks,
    single_cache_capacity_sweep_parallel,
    cache_nodes_ratio_sweep_parallel,
    simulation_time_sweep_parallel,
    run_all_sweeps_parallel
)

def example_1_basic_parallel_sweep():
    """示例1: 基本的并行参数扫描"""
    print("="*80)
    print("示例1: 基本的并行参数扫描")
    print("="*80)
    print("\n这个示例演示如何运行基本的并行参数扫描")
    print("参数: 单缓存容量 [10, 20, 30]")
    print("拓扑: GEANT")
    print("并行进程数: 4")
    print("重复实验次数: 1\n")
    
    config_file = os.path.join(os.path.dirname(__file__), 'config.yml')
    
    if not os.path.exists(config_file):
        print(f"错误: 配置文件不存在: {config_file}")
        return
    
    # 运行并行扫描（使用小规模参数进行演示）
    # 注意: 实际运行时需要修改参数范围
    print("提示: 这是一个演示示例，实际运行需要修改参数范围")
    print("建议: 使用 main() 函数进行交互式操作\n")

def example_2_custom_task_generation():
    """示例2: 自定义任务生成"""
    print("="*80)
    print("示例2: 自定义任务生成")
    print("="*80)
    print("\n这个示例演示如何生成自定义的任务列表")
    
    config_file = os.path.join(os.path.dirname(__file__), 'config.yml')
    
    # 生成自定义任务
    custom_parameters = [15, 25, 35, 45]  # 自定义参数值
    
    tasks = generate_sweep_tasks(
        config_file=config_file,
        parameter_name='single_cache_capacity',
        parameter_values=custom_parameters,
        topology_type='GEANT',
        N=1
    )
    
    print(f"\n生成了 {len(tasks)} 个自定义任务:")
    for task in tasks:
        print(f"  任务 {task.task_id}: {task.parameter_name} = {task.parameter_value}")
    
    # 创建并行管理器
    manager = ParallelExperimentManager(max_workers=4, enable_progress=True)
    
    print(f"\n并行管理器配置:")
    print(f"  最大并行进程数: {manager.max_workers}")
    print(f"  启用进度显示: {manager.enable_progress}")
    
    print("\n提示: 使用 manager.execute_tasks(tasks) 执行这些任务")

def example_3_multiple_sweeps():
    """示例3: 运行多个参数扫描"""
    print("="*80)
    print("示例3: 运行多个参数扫描")
    print("="*80)
    print("\n这个示例演示如何依次运行多个参数扫描")
    
    config_file = os.path.join(os.path.dirname(__file__), 'config.yml')
    
    print("\n扫描计划:")
    print("  1. 单缓存容量扫描")
    print("  2. 缓存节点比例扫描")
    print("  3. 模拟时间扫描")
    
    print("\n提示: 使用 run_all_sweeps_parallel() 一次性运行所有扫描")
    print("建议: 根据需要选择并行度以优化性能")

def example_4_performance_comparison():
    """示例4: 性能对比"""
    print("="*80)
    print("示例4: 性能对比")
    print("="*80)
    print("\n这个示例对比串行和并行执行的性能")
    
    print("\n性能对比（估算）:")
    print("  串行模式: 8个任务 × 2分钟/任务 = 16分钟")
    print("  并行模式 (4进程): 8个任务 / 4进程 × 2分钟/任务 = 4分钟")
    print("  加速比: 4倍")
    print("  时间节省: 75%")
    
    print("\n实际性能取决于:")
    print("  - CPU核心数")
    print("  - 内存大小")
    print("  - 磁盘I/O性能")
    print("  - 任务复杂度")

def example_5_error_handling():
    """示例5: 错误处理"""
    print("="*80)
    print("示例5: 错误处理")
    print("="*80)
    print("\n这个示例演示如何处理并行执行中的错误")
    
    print("\n常见的错误类型:")
    print("  1. 配置文件冲突")
    print("  2. 内存不足")
    print("  3. 进程超时")
    print("  4. 结果文件丢失")
    
    print("\n错误处理策略:")
    print("  - 自动重试失败的任务")
    print("  - 记录详细的错误日志")
    print("  - 提供友好的错误提示")
    print("  - 继续执行其他任务")
    
    print("\n提示: 查看执行统计摘要了解失败任务详情")

def print_menu():
    """打印菜单"""
    print("\n" + "="*80)
    print("并行化参数扫描实验 - 使用示例")
    print("="*80)
    print("\n请选择要查看的示例:")
    print("1. 基本的并行参数扫描")
    print("2. 自定义任务生成")
    print("3. 运行多个参数扫描")
    print("4. 性能对比")
    print("5. 错误处理")
    print("0. 退出")
    print("="*80)

def main():
    """主函数"""
    while True:
        print_menu()
        choice = input("\n请输入选择 (0-5): ").strip()
        
        if choice == '0':
            print("\n感谢使用并行化参数扫描实验系统！")
            break
        elif choice == '1':
            example_1_basic_parallel_sweep()
        elif choice == '2':
            example_2_custom_task_generation()
        elif choice == '3':
            example_3_multiple_sweeps()
        elif choice == '4':
            example_4_performance_comparison()
        elif choice == '5':
            example_5_error_handling()
        else:
            print("\n无效的选择，请重新输入")
        
        input("\n按Enter键继续...")

if __name__ == "__main__":
    main()
