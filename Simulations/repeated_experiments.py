import os
import yaml
import random
import statistics
import csv
import time
import multiprocessing
from datetime import datetime
from tqdm import tqdm
import simulation_code as sc

def run_method_parallel(args):
    """
    在独立进程中运行单个方法（用于并行执行）
    args: (method_name, random_seed_value, config_dict)
    """
    method_name, random_seed_value, config_dict = args
    
    # 设置随机种子
    random.seed(random_seed_value)
    
    try:
        # 导入模块（每个进程需要独立导入）
        import simulation_code as sc
        
        # 设置配置
        sc.config = config_dict
        sc.topology_type = config_dict['topology']['type']
        sc.simulation_time = config_dict['simulation']['time']
        sc.single_cache_capacity = config_dict['simulation']['single_cache_capacity']
        sc.nSP = config_dict['providers']['count']
        
        # 初始化环境
        sc.init()
        
        # 运行单个方法（传入列表形式，使用对比模式）
        result_dict = sc.main([method_name])
        
        # 从字典中提取该方法的结果
        if method_name in result_dict:
            result = result_dict[method_name]
            print(f"进程内调试: method_name={method_name}, result类型={type(result)}, result长度={len(result) if hasattr(result, '__len__') else 'N/A'}")
            return method_name, result
        else:
            print(f"方法 {method_name} 的结果不在返回字典中")
            return method_name, None
    except Exception as e:
        print(f"方法 {method_name} 执行失败: {e}")
        import traceback
        print(traceback.format_exc())
        return method_name, None

def run_experiment(method):
    """
    运行单次实验（串行执行多个方法）
    注意：调用前需要确保sc.init()已经被调用过
    """
    # 获取配置信息
    topology_type = sc.topology_type
    cache_nodes_ratio = sc.config['topology']['cache_nodes_ratio']
    # 使用sc.single_cache_capacity而不是从config字典获取，因为init()后可能被重新设置
    single_cache_capacity = sc.single_cache_capacity
    simulation_time = sc.config['simulation']['time']  # 获取模拟时间

    # 使用串行方式运行多个方法（保持与原始代码相同的行为）
    print(f"\n运行 {len(method)} 个方法...")
    
    # 运行所有方法（使用对比模式）
    try:
        results = sc.main(method)
        return topology_type, cache_nodes_ratio, single_cache_capacity, simulation_time, results
    except Exception as e:
        print(f"实验执行失败: {e}")
        import traceback
        print(traceback.format_exc())
        return None, None, None, None, None

def analyze_results(experiment_results, N, scan_type=None):
    """
    分析实验结果
    
    Args:
        scan_type: 扫描类型，用于确定结果保存的子文件夹
    """
    if not experiment_results:
        print("没有成功的实验结果")
        return

    # 整理结果
    method_results = {}
    topology_type = None
    cache_nodes_ratio = None
    single_cache_capacity = None

    for exp in experiment_results:
        topo, ratio, cap, time_val, results = exp
        if not topo or not ratio or not results:
            continue

        topology_type = topo
        cache_nodes_ratio = ratio
        single_cache_capacity = cap
        simulation_time = time_val
        
        for method_name, result in results.items():
            if method_name not in method_results:
                method_results[method_name] = {
                    'hit_rates': [],
                    'latencies': [],
                    'costs': []
                }
            
            # 计算该方法的平均命中率、时延和成本
            if len(result) >= 5:
                # 检查方法的返回值格式
                if isinstance(result, tuple) and len(result) == 6:
                    # 返回6个值的方法：(allocation, L_total_cost, L_nominal_cost, L_first_cost, L_best_cost, L_avg_latency)
                    # 这些方法已经计算了每个interval的成本和时延列表
                    # 直接计算列表的平均值，避免双重平均
                    if len(result[1]) > 0:  # L_total_cost列表
                        hit_rate = 1 - statistics.mean(result[1])  # 整个实验期间的平均命中率
                        cost = statistics.mean(result[1])  # 整个实验期间的平均成本
                    else:
                        hit_rate = 0.0
                        cost = 1.0
                    
                    if len(result[5]) > 0:  # L_avg_latency列表
                        latency = statistics.mean(result[5])  # 整个实验期间的平均时延
                    else:
                        latency = 0.0
                elif isinstance(result, list) and len(result) == 5:
                    # 返回5个值的方法：[L_total_cost, L_nominal_cost, L_first_cost, L_best_cost, L_avg_latency]
                    # 这些是强化学习方法，已经计算了成本时延列表
                    if len(result[0]) > 0:  # L_total_cost列表
                        hit_rate = 1 - statistics.mean(result[0])  # 整个实验期间的平均命中率
                        cost = statistics.mean(result[0])  # 整个实验期间的平均成本
                    else:
                        hit_rate = 0.0
                        cost = 1.0
                    
                    if len(result[4]) > 0:  # L_avg_latency列表
                        latency = statistics.mean(result[4])  # 整个实验期间的平均时延
                    else:
                        latency = 0.0
                else:
                    # 单方法模式或其他格式
                    # 根据simulation_code.py中的结果格式确定索引位置
                    
                    # 检查是否为返回6个值的方法
                    six_value_methods = ['best_allocation', 'equal_allocation', 'cooperative_best_allocation', 
                                        'proportional_allocation', 'SCA_ADMM', 'global_opt_allocation', 
                                        'SCA_neighborhood_search', 'manual_allocation', 'Q_learning']
                    
                    if method_name in six_value_methods:
                        # 对于返回6个值的方法，时延在索引5：L_avg_latency
                        latency_index = 5
                    else:
                        # 对于其他方法，时延在索引4：avg_latency
                        latency_index = 4
                    
                    if isinstance(result[1], (list, tuple)) and len(result[1]) > 0:
                        # 包含列表的情况
                        hit_rate = 1 - statistics.mean(result[1]) if len(result[1]) > 0 else 0.0
                        cost = statistics.mean(result[1]) if len(result[1]) > 0 else 1.0
                        latency = statistics.mean(result[latency_index]) if len(result) > latency_index and isinstance(result[latency_index], (list, tuple)) and len(result[latency_index]) > 0 else 0.0
                    else:
                        # 单个数值的情况
                        hit_rate = 1 - result[1] if isinstance(result[1], (int, float)) else 0.0
                        cost = result[1] if isinstance(result[1], (int, float)) else 1.0
                        latency = result[latency_index] if len(result) > latency_index and isinstance(result[latency_index], (int, float)) else 0.0
                
                method_results[method_name]['hit_rates'].append(hit_rate)
                method_results[method_name]['latencies'].append(latency)
                method_results[method_name]['costs'].append(cost)
    
    # 打印分析结果
    print(f"\n{'='*80}")
    print("实验结果分析")
    print(f"{'='*80}")
    print(f"拓扑: {topology_type}")
    print(f"缓存节点比例: {cache_nodes_ratio}")
    print(f"重复实验次数: {N}")
    print(f"{'='*80}")
    print(f"{'方法':<25} {'命中率最小值':<15} {'命中率最大值':<15} {'命中率平均值':<15} {'时延最小值':<15} {'时延最大值':<15} {'时延平均值':<15}")
    print(f"{'='*105}")
    
    # 准备保存结果
    save_data = []
    
    for method_name, data in method_results.items():
        if data['hit_rates']:
            hit_min = min(data['hit_rates'])
            hit_max = max(data['hit_rates'])
            hit_avg = statistics.mean(data['hit_rates'])
            
            lat_min = min(data['latencies'])
            lat_max = max(data['latencies'])
            lat_avg = statistics.mean(data['latencies'])
            
            print(f"{method_name:<25} {hit_min:<15.4f} {hit_max:<15.4f} {hit_avg:<15.4f} {lat_min:<15.4f} {lat_max:<15.4f} {lat_avg:<15.4f}")
            
            # 添加到保存数据
            save_data.append({
                'method': method_name,
                'topology': topology_type,
                'cache_nodes_ratio': cache_nodes_ratio,
                'single_cache_capacity': single_cache_capacity,
                'hit_min': hit_min,
                'hit_max': hit_max,
                'hit_avg': hit_avg,
                'lat_min': lat_min,
                'lat_max': lat_max,
                'lat_avg': lat_avg
            })
    
    print(f"{'='*105}")

    # 保存结果
    if save_data:
        save_results(save_data, N, topology_type, cache_nodes_ratio, single_cache_capacity, simulation_time, scan_type)

def save_results(data, N, topology_type, cache_nodes_ratio, single_cache_capacity, simulation_time, scan_type=None):
    """
    保存实验结果到CSV文件
    根据拓扑类型和扫描类型创建子文件夹结构
    
    Args:
        scan_type: 扫描类型，可选值: 'single_cache_capacity', 'cache_nodes_ratio', 'simulation_time'
    """
    # 创建结果目录结构: results/{topology_type}/{scan_type}/
    base_results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    
    # 根据扫描类型确定子文件夹名
    scan_type_name = {
        'single_cache_capacity': 'single_cache_capacity',
        'cache_nodes_ratio': 'cache_nodes_ratio',
        'simulation_time': 'simulation_time'
    }.get(scan_type, 'other')
    
    # 创建完整的结果目录路径
    results_dir = os.path.join(base_results_dir, topology_type, scan_type_name)
    os.makedirs(results_dir, exist_ok=True)

    # 生成文件名（包含所有实验参数信息）
    import uuid
    unique_id = str(uuid.uuid4())[:8]  # 使用UUID确保唯一性
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"repeated_experiments_{topology_type}_time{simulation_time}_ratio{cache_nodes_ratio}_cap{single_cache_capacity}_N{N}_{timestamp}_{unique_id}.csv"
    filepath = os.path.join(results_dir, filename)

    # 写入CSV
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['method', 'topology', 'cache_nodes_ratio', 'single_cache_capacity', 'hit_min', 'hit_max', 'hit_avg', 'lat_min', 'lat_max', 'lat_avg']
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()
        for row in data:
            writer.writerow(row)
    
    print(f"\n实验结果已保存到: {filepath}")

def main():
    """
    主函数
    """
    import sys
    
    # 检查命令行参数
    parameter_name = None
    parameter_value = None
    repeat_count = 10  # 默认重复实验次数
    config_path = None  # 可选的配置文件路径
    
    # 解析命令行参数
    # 格式: python repeated_experiments.py [config_file] [parameter_name] [parameter_value] [repeat_count]
    # 或: python repeated_experiments.py [parameter_name] [parameter_value] [repeat_count]
    args = sys.argv[1:]
    
    # 检查第一个参数是否是配置文件路径
    if args and os.path.isfile(args[0]):
        config_path = args[0]
        args = args[1:]
    
    # 解析剩余参数
    if len(args) > 2:
        parameter_name = args[0]
        parameter_value = args[1]
        repeat_count = int(args[2])
        print(f"检测到参数: {parameter_name} = {parameter_value}, 重复实验次数: {repeat_count}")
    elif len(args) > 1:
        parameter_name = args[0]
        parameter_value = args[1]
        print(f"检测到参数: {parameter_name} = {parameter_value} (使用默认重复实验次数: {repeat_count})")
    elif len(args) > 0:
        parameter_value = args[0]
        print(f"检测到参数值: {parameter_value} (参数名未知, 使用默认重复实验次数: {repeat_count})")
    
    # 如果没有指定配置文件路径，使用默认路径
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), 'config.yml')
    
    # 读取配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 如果配置加载失败，抛出错误
    if config is None:
        raise ValueError(f"配置文件加载失败: {config_path}")
    
    # 如果传入了参数名和参数值，更新配置
    if parameter_name and parameter_value:
        print(f"使用传入的参数更新配置: {parameter_name} = {parameter_value}")
        
        # 根据参数名更新对应的配置项
        if parameter_name == 'single_cache_capacity':
            config['simulation'][parameter_name] = int(parameter_value)
        elif parameter_name == 'cache_nodes_ratio':
            config['topology'][parameter_name] = float(parameter_value)
        elif parameter_name == 'simulation_time':
            config['simulation']['time'] = int(parameter_value)  # 注意：配置中使用的是'time'而不是'simulation_time'
        
        print(f"配置已更新: {parameter_name} = {parameter_value}")
    
    # 设置随机种子不固定
    config['rl']['fixed_seed'] = False
    
    # 实验参数
    N = repeat_count  # 使用传入的重复实验次数
    print(f"开始运行 {N} 次重复实验...")
    
    method = ['SCA_neighborhood_search', 'SCA_ADMM', 'global_opt_allocation', 'best_allocation', 'equal_allocation', 'proportional_allocation', 'cooperative_best_allocation', 'Q_learning']
    
    # 存储实验结果
    experiment_results = []
    
    # 运行N次实验，使用tqdm进度条
    print(f"\n开始运行 {N} 次重复实验...")
    print("使用tqdm进度条显示实验进度和预计完成时间")
    
    # 记录开始时间
    start_time = time.time()
    
    # 只读取一次配置文件，避免循环中文件被删除的问题
    with open(config_path, 'r', encoding='utf-8') as f:
        base_config = yaml.safe_load(f)
    
    # 如果传入了参数名和参数值，预先更新配置
    if parameter_name and parameter_value:
        if parameter_name == 'single_cache_capacity':
            base_config['simulation'][parameter_name] = int(parameter_value)
        elif parameter_name == 'cache_nodes_ratio':
            base_config['topology'][parameter_name] = float(parameter_value)
        elif parameter_name == 'simulation_time':
            base_config['simulation']['time'] = int(parameter_value)
    
    # 设置自定义配置文件路径，让 sc.init() 使用隔离的配置文件
    sc.custom_config_path = config_path
    
    # 创建tqdm进度条
    with tqdm(total=N, desc="重复实验进度", unit="实验") as pbar:
        for i in range(N):
            # 记录当前实验开始时间
            experiment_start_time = time.time()
            
            # 生成不同的随机种子
            random_seed = random.randint(1, 10000)
            random.seed(random_seed)
            
            # 使用内存中的配置副本，避免文件读取问题
            current_config = base_config.copy()
            
            # 将修改后的配置写回配置文件（因为sc.init()会调用load_config()重新加载）
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(current_config, f, default_flow_style=False)
            
            # 更新全局配置
            sc.config = current_config
            sc.topology_type = current_config['topology']['type']
            sc.simulation_time = current_config['simulation']['time']
            sc.fixed_seed = current_config['rl']['fixed_seed']
            
            # 更新拓扑相关参数
            sc.topology_params = current_config['topology']['parameters']
            sc.cache_nodes_count = current_config['topology'].get('cache_nodes_ratio', current_config['topology'].get('cache_nodes', 10))
            
            # 从providers配置获取SP节点数量
            sc.nSP = current_config['providers']['count']
            sc.sp_nodes_count = sc.nSP
            
            # 路由器节点数量默认为None，由拓扑管理器自动计算
            sc.router_nodes_count = None
            
            # 重新初始化拓扑和视频目录
            sc.topology_manager = None
            sc.cacheable_content = None
            sc.init()
            
            # 运行实验
            result = run_experiment(method)
            
            if result:
                topology_type, cache_nodes_ratio, single_cache_capacity, simulation_time, results = result
            else:
                topology_type, cache_nodes_ratio, single_cache_capacity, simulation_time, results = None, None, None, None, None

            if topology_type and cache_nodes_ratio and results:
                experiment_results.append((topology_type, cache_nodes_ratio, single_cache_capacity, simulation_time, results))
                
                # 计算当前实验耗时
                experiment_time = time.time() - experiment_start_time
                
                # 更新进度条描述，显示实验统计信息
                elapsed_time = time.time() - start_time
                avg_time_per_experiment = elapsed_time / (i + 1)
                estimated_total_time = avg_time_per_experiment * N
                remaining_time = estimated_total_time - elapsed_time
                
                # 格式化时间显示
                elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
                remaining_str = time.strftime("%H:%M:%S", time.gmtime(remaining_time))
                
                pbar.set_description(f"实验进度 (已用: {elapsed_str}, 预计剩余: {remaining_str})")
                pbar.set_postfix({
                    "当前实验耗时": f"{experiment_time:.1f}s",
                    "平均耗时": f"{avg_time_per_experiment:.1f}s",
                    "随机种子": random_seed
                })
                
            else:
                print(f"实验 {i+1} 失败")
            
            # 更新进度条
            pbar.update(1)
    
    # 计算总耗时
    total_time = time.time() - start_time
    total_time_str = time.strftime("%H:%M:%S", time.gmtime(total_time))
    print(f"\n所有实验完成！总耗时: {total_time_str}")
    
    # 分析结果
    if experiment_results:
        analyze_results(experiment_results, N, scan_type=parameter_name)
    else:
        print("没有成功的实验结果")

if __name__ == "__main__":
    main()