#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版缓存资源分配模拟代码
支持从配置文件读取参数，支持节点间时延模拟
Q-learning 部分已按新设计重构：奖励 = 命中率变化量，动作 = 单位缓存移动
"""

import random as rd
import numpy as np
import Auxiliary_functions as af
from copy import deepcopy
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import math
import os
import time

from topology_manager import TopologyManager

# 全局配置文件路径，可由外部设置以支持并行执行
custom_config_path = None

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("PyTorch未安装，DQN方法将不可用。请运行: pip install torch")

SCA_ADMM_RHO = None
SCA_ADMM_TAU = None


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = []
        self.capacity = capacity
        self.position = 0
        self.size = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        if self.size < batch_size:
            batch_size = self.size
        return rd.sample(self.buffer[:self.size], batch_size)
    
    def __len__(self):
        return self.size


class DQNNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, action_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

# 全局变量
config = None
list_alpha = None
SP_proba = None
video_nb_list = None
conss_zipf = None
nSP = None
single_cache_capacity = None
nb_videos = None
gamma = None
epsilon = None
alpha_de_sarsa = None
DEBUG = False
should_i_simulate = True
activate_memory = True
epsilon_decay = True
alpha_scheduling = True
simulation_time = None
cacheability = None
cacheable_content = None  # 存储每个SP的可缓存内容列表
fixed_seed = None  # 是否固定随机种子，默认为true

# 在线学习相关变量
node_request_history = {}  # 每个节点的请求历史
cache_node_neighbors = {}  # 缓存节点的邻居列表

# 网络参数
network_enabled = None
nodes = 0
bandwidth = 0

# 拓扑参数
topology_type = None
topology_params = None
cache_nodes_count = None
sp_nodes_count = None
router_nodes_count = None
topology_manager = None

# 输出参数
save_allocations = True
save_results = True
debug_mode = False
results_dir = "results"
figures_dir = "figures"
cooperative_caching = False  # 是否启用协同缓存

def debug_print(*args, **kwargs):
    """仅在debug模式开启时打印"""
    if debug_mode:
        print(*args, **kwargs)

# delta相关参数
min_delta = 10  # 最小delta值
delta_decay = 0.99  # delta衰减率


# 协同缓存参数
user_to_cache_latency = 5  # 用户到达缓存节点的时延（毫秒）
cache_to_cache_latency = 10  # 缓存节点到邻居节点的时延（毫秒）
cache_to_sp_latency = 30  # 未命中时到达SP的时延增加（毫秒）

def load_config():
    """
    从配置文件加载参数
    """
    global config, list_alpha, SP_proba, video_nb_list, conss_zipf, nSP
    global single_cache_capacity, nb_videos, gamma, epsilon, alpha_de_sarsa
    global simulation_time, cacheability
    global topology_type, topology_params, cache_nodes_count, sp_nodes_count, router_nodes_count
    global save_allocations, save_results, debug_mode, results_dir, figures_dir
    global fixed_seed
    global min_delta, delta_decay
    global cooperative_caching, user_to_cache_latency, cache_to_cache_latency, cache_to_sp_latency

    
    # 使用绝对路径加载配置文件
    import os
    
    # 如果设置了自定义配置文件路径，则使用它（用于并行执行）
    global custom_config_path
    if custom_config_path is not None and os.path.exists(custom_config_path):
        config_path = custom_config_path
    else:
        config_path = os.path.join(os.path.dirname(__file__), 'config.yml')
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 基本模拟参数
    sim_config = config['simulation']
    simulation_time = sim_config['time']
    single_cache_capacity = sim_config['single_cache_capacity']
    nb_videos = sim_config['nb_videos']
    # delta相关参数
    min_delta = sim_config.get('min_delta', 10)
    delta_decay = sim_config.get('delta_decay', 0.99)
    
    # 服务提供商参数
    prov_config = config['providers']
    nSP = prov_config['count']
    SP_proba = prov_config['probabilities']
    cacheability = prov_config['cacheability']
    list_alpha = prov_config['zipf_alphas']
    
    # 强化学习参数
    rl_config = config['rl']
    gamma = rl_config['gamma']
    epsilon = rl_config['epsilon']
    fixed_seed = rl_config.get('fixed_seed', True)
    
    # 拓扑参数
    topo_config = config['topology']
    topology_type = topo_config['type']
    topology_params = topo_config['parameters']
    cache_nodes_count = topo_config.get('cache_nodes_ratio', topo_config.get('cache_nodes', 10))
    # SP节点数目使用providers的count参数
    sp_nodes_count = nSP
    # 检查配置文件中是否指定了 router_nodes
    router_nodes_count = topo_config.get('router_nodes', None)
    
    # 输出参数
    out_config = config['output']
    save_allocations = out_config['save_allocations']
    save_results = out_config['save_results']
    debug_mode = out_config.get('debug', False)
    results_dir = out_config['results_dir']
    figures_dir = out_config['figures_dir']
    # 网络模拟参数
    global network_enabled
    network_enabled = out_config.get('network_enabled', False)
    
    # 协同缓存配置
    global cooperative_caching, user_to_cache_latency, cache_to_cache_latency, cache_to_sp_latency
    coop_config = config.get('cooperative_caching', {})
    cooperative_caching = coop_config.get('enabled', False)
    user_to_cache_latency = coop_config.get('user_to_cache_latency', 5)
    cache_to_cache_latency = coop_config.get('cache_to_cache_latency', 10)
    cache_to_sp_latency = coop_config.get('cache_to_sp_latency', 30)

    
    # 创建输出目录
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    # 计算衍生参数
    video_nb_list = [nb_videos for i in range(nSP)]
    conss_zipf = [af.zipf_norm(list_alpha[i], video_nb_list[i]) for i in range(nSP)]

    if debug_mode:
        print("配置加载完成:")
        print(f"  模拟时间: {simulation_time}秒")
        print(f"  单个缓存节点容量: {single_cache_capacity}")
        print(f"  服务提供商数量: {nSP}")
        print(f"  拓扑类型: {topology_type}")
        if isinstance(cache_nodes_count, float) and 0 < cache_nodes_count <= 1:
            print(f"  缓存节点比例: {cache_nodes_count} (将在创建拓扑后根据节点总数计算实际数量)")
        else:
            print(f"  缓存节点数目: {cache_nodes_count}")
        print(f"  SP节点数目: {sp_nodes_count}")
        if router_nodes_count is not None:
            print(f"  路由器节点数目: {router_nodes_count}")
        else:
            print(f"  路由器节点数目: 剩余所有节点")

def init(seed=None):
    """
    初始化全局变量，确保完全重置所有状态
    用于保证实验的可重复性和独立性
    
    参数:
        seed: 可选的随机种子。如果提供，将使用该种子初始化随机数生成器，
              确保拓扑创建的可重复性。如果为None，使用配置文件中的设置。
    """
    # 重置所有可能影响实验的全局变量
    global config, list_alpha, SP_proba, video_nb_list, conss_zipf
    global nSP, single_cache_capacity, nb_videos, gamma, epsilon, alpha_de_sarsa
    global DEBUG, should_i_simulate, activate_memory, epsilon_decay, alpha_scheduling
    global simulation_time, cacheability, cacheable_content, fixed_seed
    global node_request_history, cache_node_neighbors
    global network_enabled, nodes, bandwidth
    global topology_type, topology_params, cache_nodes_count, sp_nodes_count
    global router_nodes_count, topology_manager
    global save_allocations, save_results, debug_mode, results_dir, figures_dir
    global cooperative_caching, user_to_cache_latency, cache_to_cache_latency
    global cache_to_sp_latency, min_delta, delta_decay
    global SCA_ADMM_RHO, SCA_ADMM_TAU
    global previous_sp_proba, previous_sp_video_scores

    # 重置核心全局变量
    config = None
    list_alpha = None
    SP_proba = None
    video_nb_list = None
    conss_zipf = None
    nSP = None
    single_cache_capacity = None
    nb_videos = None
    gamma = None
    epsilon = None
    alpha_de_sarsa = None
    DEBUG = False
    should_i_simulate = True
    activate_memory = True
    epsilon_decay = True
    alpha_scheduling = True
    simulation_time = None
    cacheability = None
    cacheable_content = None  # 这很重要，确保重新生成可缓存内容
    fixed_seed = None

    # 重置在线学习相关变量
    node_request_history = {}
    cache_node_neighbors = {}
    previous_sp_proba = {}  # 添加这个变量的重置
    previous_sp_video_scores = {}  # 添加这个变量的重置

    # 重置网络参数
    network_enabled = None
    nodes = 0
    bandwidth = 0

    # 重置拓扑参数
    topology_type = None
    topology_params = None
    cache_nodes_count = None
    sp_nodes_count = None
    router_nodes_count = None
    topology_manager = None

    # 重置输出参数
    save_allocations = True
    save_results = True
    debug_mode = False
    results_dir = "results"
    figures_dir = "figures"
    cooperative_caching = False

    # 重置协同缓存参数
    user_to_cache_latency = 5
    cache_to_cache_latency = 10
    cache_to_sp_latency = 30

    # 重置delta相关参数
    min_delta = 10
    delta_decay = 0.99

    # 保存SCA_ADMM参数（如果已设置则保留）
    saved_rho = SCA_ADMM_RHO
    saved_tau = SCA_ADMM_TAU

    # 加载配置并重新初始化
    load_config()

    # 恢复SCA_ADMM参数（如果之前已设置）
    if saved_rho is not None:
        SCA_ADMM_RHO = saved_rho
    if saved_tau is not None:
        SCA_ADMM_TAU = saved_tau

    # 设置随机种子
    if seed is not None:
        # 使用外部提供的种子（用于重复实验）
        rd.seed(seed)
        np.random.seed(seed)
        if debug_mode:
            print(f"随机种子设置: 外部指定 ({seed})")
    elif fixed_seed:
        rd.seed(3231)  # 设置固定随机种子以保证可重复性
        np.random.seed(3231)
        if debug_mode:
            print(f"随机种子设置: 固定 (3231)")
    else:
        # 不固定随机种子，使用系统时间作为种子
        rd.seed()
        np.random.seed()
        if debug_mode:
            print(f"随机种子设置: 不固定 (系统时间)")

    # 初始化拓扑管理器
    topology_manager = TopologyManager(
        topology_type,
        topology_params,
        cache_nodes_count,
        sp_nodes_count,
        router_nodes_count
    )
    topology_manager.create_topology()

    if debug_mode:
        print(f"拓扑创建完成，节点总数: {len(topology_manager.graph.nodes())}")
        print(f"缓存节点: {topology_manager.get_cache_nodes()}")
        print(f"SP节点: {topology_manager.get_sp_nodes()}")
        print(f"路由器节点: {topology_manager.get_router_nodes()}")
        print(f"接收器节点: {topology_manager.get_receiver_nodes()}")

        # 打印每个缓存节点的邻居缓存节点列表
        print("\n=== 缓存节点邻居列表 ===")
        cache_nodes = topology_manager.get_cache_nodes()
        for node in cache_nodes:
            all_neighbors = topology_manager.get_neighbors(node)
            neighbor_cache_nodes = [n for n in all_neighbors if n in cache_nodes]
            print(f"节点 {node} 的邻居缓存节点: {neighbor_cache_nodes}")

    # 初始化视频目录和可缓存内容
    if debug_mode:
        print("\n=== 初始化视频目录 ===")
    initial_videos_proba = catalog()
    if debug_mode:
        print(f"视频目录初始化完成，可缓存内容长度: {len(cacheable_content) if cacheable_content else 0}")
        for SP in range(nSP):
            print(f"SP {SP} 可缓存内容数量: {len(cacheable_content[SP]) if cacheable_content and SP < len(cacheable_content) else 0}")

def zipf_distribution(alpha, nb_videos, norm):
    """
    创建遵循Zipf定律的视频请求概率分布
    """
    probabilites_pi = np.zeros(nb_videos)
    for i in range(1, nb_videos+1):
        pi = (1.0/i**alpha) * (1.0/norm)
        probabilites_pi[i-1] = pi
    return probabilites_pi

def catalog():
    """
    创建视频目录
    """
    global cacheable_content, videos_proba
    
    # 如果cacheable_content已经生成，直接返回
    if cacheable_content is not None:
        return videos_proba
    
    videos = np.zeros((nSP, nb_videos))
    
    # 为每个SP生成可缓存内容列表
    cacheable_content = []
    for SP in range(nSP):
        # 生成视频概率分布
        videos[SP] = zipf_distribution(list_alpha[SP], video_nb_list[SP], conss_zipf[SP])
        
        # 生成可缓存内容列表
        # 根据可缓存比例随机选择可缓存的视频
        cacheable = []
        for video_id in range(nb_videos):
            # 根据可缓存比例决定该视频是否可缓存
            if rd.random() <= cacheability[SP]:
                cacheable.append(video_id)
        cacheable_content.append(cacheable)
    
    # 打印可缓存内容信息
    #for SP in range(nSP):
    #    print(f"SP {SP} 可缓存内容数量: {len(cacheable_content[SP])}, 比例: {len(cacheable_content[SP])/nb_videos:.2f}")
    
    # 保存生成的视频概率分布
    videos_proba = videos
    return videos


def record_request(node, sp, video):
    """
    记录节点的请求历史
    """
    global node_request_history
    if node not in node_request_history:
        node_request_history[node] = {sp: {} for sp in range(nSP)}
    if sp not in node_request_history[node]:
        node_request_history[node][sp] = {}
    if video not in node_request_history[node][sp]:
        node_request_history[node][sp][video] = 0
    node_request_history[node][sp][video] += 1


def estimate_video_proba(node, sp, video, window_size=1000):
    """
    增强的视频概率估计，结合短期和长期历史数据
    使用滑动窗口或均匀分布作为初始估计
    """
    global node_request_history
    
    if node not in node_request_history or sp not in node_request_history[node]:
        return 1.0 / nb_videos  # 均匀分布作为初始估计
    
    history = node_request_history[node][sp]
    total_requests = sum(history.values())
    if total_requests == 0:
        return 1.0 / nb_videos  # 均匀分布
    
    # 短期概率（最近请求）
    short_term_prob = history.get(video, 0) / min(total_requests, window_size)
    
    # 长期概率（历史模式）
    long_term_prob = short_term_prob  # 默认为短期概率
    
    # 尝试使用node_local_states中的历史数据
    node_local_states = None
    try:
        node_local_states = globals().get('node_local_states')
        if node_local_states and node in node_local_states:
            node_history = node_local_states[node]['request_history']
            if 'video_requests' in node_history and sp in node_history['video_requests'] and video in node_history['video_requests'][sp]:
                # 计算历史平均概率
                historical_counts = node_history['video_requests'][sp].get(video, 0)
                historical_total = sum(node_history['video_requests'][sp].values())
                if historical_total > 0:
                    long_term_prob = historical_counts / historical_total
    except (NameError, KeyError):
        pass
    
    # 综合短期和长期概率
    alpha = 0.7  # 短期权重
    combined_prob = alpha * short_term_prob + (1 - alpha) * long_term_prob
    
    # 尝试更新历史数据
    try:
        node_local_states = globals().get('node_local_states')
        if node_local_states and node in node_local_states:
            node_history = node_local_states[node]['request_history']
            if 'video_requests' not in node_history:
                node_history['video_requests'] = {}
            if sp not in node_history['video_requests']:
                node_history['video_requests'][sp] = {}
            if video not in node_history['video_requests'][sp]:
                node_history['video_requests'][sp][video] = 0
            node_history['video_requests'][sp][video] += history.get(video, 0)
    except (NameError, KeyError):
        pass
    
    return combined_prob


def estimate_sp_proba(node, sp, window_size=1000, smoothing=0.7):
    """
    增强的SP概率估计，结合短期和长期历史数据
    使用节点本地的请求历史，不使用全局SP_proba
    """
    global node_request_history, cache_node_neighbors, previous_sp_proba
    
    # 检查节点是否有本地历史数据
    if node in node_request_history:
        # 统计该节点对每个SP的请求总数
        sp_request_counts = {}
        total_requests = 0
        for sp_idx in range(nSP):
            if sp_idx in node_request_history[node]:
                count = sum(node_request_history[node][sp_idx].values())
                sp_request_counts[sp_idx] = count
                total_requests += count
        
        if total_requests > 0:
            # 短期概率（最近请求）
            current_proba = sp_request_counts.get(sp, 0) / min(total_requests, window_size)
            
            # 长期概率（历史模式）
            long_term_proba = current_proba  # 默认为短期概率
            
            # 尝试使用node_local_states中的历史数据
            node_local_states = None
            try:
                node_local_states = globals().get('node_local_states')
                if node_local_states and node in node_local_states:
                    history = node_local_states[node]['request_history']
                    if 'sp_requests' in history and sp in history['sp_requests']:
                        # 计算历史平均概率
                        historical_counts = history['sp_requests'].get(sp, 0)
                        historical_total = sum(history['sp_requests'].values())
                        if historical_total > 0:
                            long_term_proba = historical_counts / historical_total
            except (NameError, KeyError):
                pass
            
            # 综合短期和长期概率
            alpha = 0.7  # 短期权重
            combined_proba = alpha * current_proba + (1 - alpha) * long_term_proba
            
            # 平滑处理
            if 'previous_sp_proba' in globals() and node in previous_sp_proba and sp in previous_sp_proba[node]:
                combined_proba = smoothing * combined_proba + (1 - smoothing) * previous_sp_proba[node][sp]
            
            # 保存当前概率用于下一次平滑
            if 'previous_sp_proba' in globals():
                if node not in previous_sp_proba:
                    previous_sp_proba[node] = {}
                previous_sp_proba[node][sp] = combined_proba
            
            # 尝试更新历史数据
            try:
                node_local_states = globals().get('node_local_states')
                if node_local_states and node in node_local_states:
                    if 'sp_requests' not in node_local_states[node]['request_history']:
                        node_local_states[node]['request_history']['sp_requests'] = {}
                    if int(sp) not in node_local_states[node]['request_history']['sp_requests']:
                        node_local_states[node]['request_history']['sp_requests'][int(sp)] = 0
                    node_local_states[node]['request_history']['sp_requests'][int(sp)] += sp_request_counts.get(sp, 0)
            except (NameError, KeyError):
                pass
            
            return combined_proba
    
    # 冷启动策略1：使用邻居的SP概率作为参考
    neighbors = cache_node_neighbors.get(node, [])
    neighbor_sp_counts = {}
    neighbor_total = 0
    for neighbor in neighbors:
        if neighbor in node_request_history:
            for sp_idx in range(nSP):
                if sp_idx in node_request_history[neighbor]:
                    count = sum(node_request_history[neighbor][sp_idx].values())
                    neighbor_sp_counts[sp_idx] = neighbor_sp_counts.get(sp_idx, 0) + count
                    neighbor_total += count
    
    if neighbor_total > 0:
        return neighbor_sp_counts.get(sp, 0) / neighbor_total
    
    # 冷启动策略2：基于SP索引的启发式（靠前的SP可能更重要）
    # 这是一种无数据时的合理启发式
    sp_importance = 1.0 / (sp + 1)  # 指数衰减
    total_importance = sum(1.0 / (s + 1) for s in range(nSP))
    return sp_importance / total_importance


def request_creation(video_probability):
    """
    创建模拟请求
    """
    # 选择SP
    S = SP_proba[0]
    SP_choice = rd.random()
    selected_SP = 0
    
    while(SP_choice > S):
        selected_SP += 1
        S += SP_proba[selected_SP]
    
    # 直接选择视频
    video_choice = rd.random()
    S2 = 0
    selected_video = -1
    
    while(video_choice > S2):
        selected_video += 1
        S2 += video_probability[selected_SP][selected_video]
    
    return [selected_SP, selected_video]

def calculate_latency(source_node, dest_node):
    """
    计算节点间的时延
    考虑节点间的实际距离和拓扑结构
    """
    if not network_enabled:
        return 0
    
    # 使用拓扑管理器计算节点间的路径长度
    if topology_manager:
        path_length = topology_manager.get_path_length(source_node, dest_node)
        if path_length != float('inf'):
            # 基础时延 + 路径长度 * 单位距离时延
            base_latency = 1  # 基础处理时延
            per_hop_latency = 5  # 每跳时延
            return base_latency + path_length * per_hop_latency
    
    # 如果没有拓扑管理器，返回默认时延
    return 0

def decide_opt_alloc(distrib):
    """
    计算最优缓存分配
    支持多个缓存节点的情况
    """
    # 获取缓存节点列表
    cache_nodes = topology_manager.get_cache_nodes()
    num_cache_nodes = len(cache_nodes)

    # 如果没有缓存节点，返回空分配
    if num_cache_nodes == 0:
        return {}

    # 为每个缓存节点分配缓存空间
    allocation = {}

    # 计算每个缓存节点的缓存容量（每个节点使用固定的single_cache_capacity）
    node_cache_capacity = single_cache_capacity
    
    for i, node in enumerate(cache_nodes):
        # 为每个缓存节点创建一个列表，存储每个SP的缓存分配
        node_allocation = [0] * nSP
        # 计算该节点的实际缓存容量（每个节点都是固定的single_cache_capacity）
        actual_capacity = node_cache_capacity
        
        # 为该节点计算最优分配
        pointer_vec = [0] * nSP
        
        for slot in range(actual_capacity):
            bestSP = 0  # 初始化
            best_score = -1
            
            for currentSP in range(nSP):
                # 确保有可缓存的内容
                if cacheable_content and currentSP < len(cacheable_content):
                    cacheable_videos = cacheable_content[currentSP]
                    if pointer_vec[currentSP] < len(cacheable_videos):
                        # 获取当前可缓存视频的索引
                        video_idx = cacheable_videos[pointer_vec[currentSP]]
                        # 计算当前SP的得分
                        score = distrib[currentSP][video_idx] * SP_proba[currentSP]
                        if score > best_score:
                            best_score = score
                            bestSP = currentSP
            
            # 分配slot给最佳SP
            node_allocation[bestSP] += 1
            pointer_vec[bestSP] += 1
        
        allocation[node] = node_allocation
    
    return allocation

def decide_cooperative_opt_alloc(distrib):
    """
    计算考虑邻居协作的最优缓存分配
    支持多个缓存节点的情况
    """
    # 获取缓存节点列表
    cache_nodes = topology_manager.get_cache_nodes()
    num_cache_nodes = len(cache_nodes)

    # 如果没有缓存节点，返回空分配
    if num_cache_nodes == 0:
        return {}

    # 为每个缓存节点创建邻居节点列表
    cache_node_neighbors = {}
    for node in cache_nodes:
        all_neighbors = topology_manager.get_neighbors(node)
        neighbors = [n for n in all_neighbors if n in cache_nodes]
        cache_node_neighbors[node] = neighbors
    
    # 为每个缓存节点分配缓存空间
    allocation = {}
    
    # 计算每个缓存节点的缓存容量（每个节点使用固定的single_cache_capacity）
    node_cache_capacity = single_cache_capacity
    
    # 首先初始化所有节点的分配为0
    for node in cache_nodes:
        allocation[node] = [0] * nSP
    
    # 为每个缓存节点计算最优分配，考虑邻居节点的缓存
    for i, node in enumerate(cache_nodes):
        # 计算该节点的实际缓存容量（每个节点都是固定的single_cache_capacity）
        actual_capacity = node_cache_capacity
        
        # 为该节点计算最优分配
        pointer_vec = [0] * nSP
        
        for slot in range(actual_capacity):
            bestSP = 0  # 初始化
            best_score = -1
            best_video_idx = -1
            
            for currentSP in range(nSP):
                # 确保有可缓存的内容
                if cacheable_content and currentSP < len(cacheable_content):
                    cacheable_videos = cacheable_content[currentSP]
                    if pointer_vec[currentSP] < len(cacheable_videos):
                        # 获取当前可缓存视频的索引
                        video_idx = cacheable_videos[pointer_vec[currentSP]]
                        
                        # 检查邻居节点是否已经缓存了该视频
                        neighbor_has_video = False
                        for neighbor in cache_node_neighbors[node]:
                            if neighbor in allocation:
                                # 检查邻居节点是否有足够的缓存分配
                                if allocation[neighbor][currentSP] > video_idx:
                                    neighbor_has_video = True
                                    break
                        
                        # 计算当前SP的得分，考虑邻居节点的缓存情况
                        base_score = distrib[currentSP][video_idx] * SP_proba[currentSP]
                        
                        # 如果邻居节点已经缓存了该视频，降低得分
                        if neighbor_has_video:
                            score = base_score * 0.5  # 降低权重，避免重复缓存
                        else:
                            score = base_score  # 邻居节点未缓存，保持原始得分
                        
                        if score > best_score:
                            best_score = score
                            bestSP = currentSP
                            best_video_idx = video_idx
            
            # 分配slot给最佳SP
            if bestSP is not None and best_video_idx != -1:
                allocation[node][bestSP] += 1
                pointer_vec[bestSP] += 1
        
    return allocation

def decide_global_opt_alloc(distrib):
    """
    计算全局最优缓存分配
    利用全局信息实现更优的缓存分配策略
    结合cooperative_best_allocation的优点，同时利用全局信息进行优化
    """
    import random
    import math
    import time
    
    start_time = time.time()
    
    # 获取缓存节点列表
    cache_nodes = topology_manager.get_cache_nodes()
    num_cache_nodes = len(cache_nodes)

    # 如果没有缓存节点，返回空分配
    if num_cache_nodes == 0:
        return {}

    # 为每个缓存节点创建邻居节点列表
    cache_node_neighbors = {}
    for node in cache_nodes:
        all_neighbors = topology_manager.get_neighbors(node)
        neighbors = [n for n in all_neighbors if n in cache_nodes]
        cache_node_neighbors[node] = neighbors
    
    # 计算每个缓存节点的缓存容量（每个节点使用固定的single_cache_capacity）
    node_cache_capacity = single_cache_capacity
    
    # 初始化分配
    allocation = {}
    node_capacities = {}
    for i, node in enumerate(cache_nodes):
        # 每个节点的实际缓存容量都是固定的single_cache_capacity
        actual_capacity = node_cache_capacity
        node_capacities[node] = actual_capacity
        allocation[node] = [0] * nSP
    
    # 计算节点重要性（基于邻居数量、网络位置和中心性）
    node_importance = {}
    node_centrality = {}
    for node in cache_nodes:
        # 计算节点的度（邻居数量）
        degree = len(cache_node_neighbors[node])
        
        # 计算节点的中心性（基于邻居的重要性）
        centrality = 0
        for neighbor in cache_node_neighbors[node]:
            centrality += len(cache_node_neighbors.get(neighbor, []))
        node_centrality[node] = centrality
        
        # 节点重要性 = 度和中心性的综合
        node_importance[node] = degree + 0.5 * centrality
    
    # 归一化节点重要性
    max_importance = max(node_importance.values()) if node_importance else 1
    # 防止所有节点重要性都为0的情况
    if max_importance == 0:
        max_importance = 1
    for node in node_importance:
        node_importance[node] /= max_importance
    
    # 计算网络规模因子，用于动态调整权重
    # 调整网络规模因子计算，增加小网络中全局信息的权重
    network_size_factor = min(1.0, max(0.7, num_cache_nodes / 15.0))  # 网络规模越大，全局权重越高，小网络最低保持0.7
    
    # 构建视频到索引的映射表，优化查找性能
    video_to_index = {}
    for sp in range(nSP):
        if cacheable_content and sp < len(cacheable_content):
            cacheable_videos = cacheable_content[sp]
            video_to_index[sp] = {video: idx for idx, video in enumerate(cacheable_videos)}
    
    # 预计算每个视频的全局重要性（考虑节点重要性）
    video_global_importance = {}
    for sp in range(nSP):
        if cacheable_content and sp < len(cacheable_content):
            cacheable_videos = cacheable_content[sp]
            # 限制计算范围，提高效率
            max_videos = min(1000, len(cacheable_videos))  # 扩大计算范围，增加小网络中全局信息的价值
            for video_idx in cacheable_videos[:max_videos]:
                global_importance = 0
                for other_node in cache_nodes:
                    other_prob = distrib[sp][video_idx] if sp < len(distrib) and video_idx < len(distrib[sp]) else 0
                    # 考虑节点重要性和中心性
                    importance_weight = node_importance.get(other_node, 1)
                    max_centrality = max(node_centrality.values()) if node_centrality else 1
                    if max_centrality == 0:
                        max_centrality = 1
                    centrality_weight = node_centrality.get(other_node, 0) / max_centrality
                    # 增加小网络中全局信息的权重
                    global_importance += other_prob * SP_proba[sp] * (0.6 * importance_weight + 0.4 * centrality_weight) * (1 + 0.3 * (1 - network_size_factor))
                video_global_importance[(sp, video_idx)] = global_importance
    
    # 计算全局缓存分布情况
    def calculate_global_cache_distribution(allocation):
        """计算全局缓存分布情况"""
        # 计算每个视频的全局缓存频率
        video_cache_frequency = {}
        for node in cache_nodes:
            for sp in range(nSP):
                if cacheable_content and sp < len(cacheable_content):
                    cacheable_videos = cacheable_content[sp]
                    for video_idx in range(min(allocation[node][sp], 500)):  # 进一步扩大计算范围，增加小网络中全局信息的价值
                        video = cacheable_videos[video_idx]
                        key = (sp, video)
                        video_cache_frequency[key] = video_cache_frequency.get(key, 0) + 1
        
        # 计算每个视频的全局缓存覆盖率
        video_cache_coverage = {}
        for sp in range(nSP):
            if cacheable_content and sp < len(cacheable_content):
                cacheable_videos = cacheable_content[sp]
                # 限制计算范围，提高效率
                max_videos = min(1000, len(cacheable_videos))  # 扩大计算范围，增加小网络中全局信息的价值
                for video_idx in range(min(max_videos, len(cacheable_videos))):  # 扩大计算范围
                    coverage = 0
                    for node in cache_nodes:
                        # 检查节点是否缓存了该视频
                        if cacheable_content and sp < len(cacheable_content):
                            if video_idx < allocation[node][sp]:
                                # 计算节点对该视频的覆盖贡献，考虑节点重要性和中心性
                                node_prob = distrib[sp][video_idx] if sp < len(distrib) and video_idx < len(distrib[sp]) else 0
                                importance_weight = node_importance.get(node, 1)
                                max_centrality = max(node_centrality.values()) if node_centrality else 1
                                if max_centrality == 0:
                                    max_centrality = 1
                                centrality_weight = node_centrality.get(node, 0) / max_centrality
                                # 增加小网络中全局信息的权重
                                coverage += node_prob * (0.6 * importance_weight + 0.4 * centrality_weight) * (1 + 0.3 * (1 - network_size_factor))
                    video_cache_coverage[(sp, video_idx)] = coverage
        
        return video_cache_frequency, video_cache_coverage
    
    # 计算每个SP的全局重要性
    sp_global_importance = {}
    for sp in range(nSP):
        sp_importance = 0
        if cacheable_content and sp < len(cacheable_content):
            cacheable_videos = cacheable_content[sp]
            # 限制计算范围，提高效率
            max_videos = min(200, len(cacheable_videos))
            for video_idx in cacheable_videos[:max_videos]:
                sp_importance += video_global_importance.get((sp, video_idx), 0)
        sp_global_importance[sp] = sp_importance
    
    # 归一化SP全局重要性
    total_sp_importance = sum(sp_global_importance.values())
    sp_importance_ratio = {}
    for sp in range(nSP):
        sp_importance_ratio[sp] = sp_global_importance[sp] / total_sp_importance if total_sp_importance > 0 else 1.0 / nSP
    
    # 为每个缓存节点计算最优分配，考虑邻居节点的缓存和全局信息
    for i, node in enumerate(cache_nodes):
        # 计算该节点的实际缓存容量
        actual_capacity = node_capacities[node]
        
        # 为该节点计算最优分配
        pointer_vec = [0] * nSP
        
        for slot in range(actual_capacity):
            bestSP = 0  # 初始化
            best_score = -1
            best_video_offset = -1
            
            # 遍历所有SP，寻找最佳的SP和视频组合
            for currentSP in range(nSP):
                # 确保有可缓存的内容
                if cacheable_content and currentSP < len(cacheable_content):
                    cacheable_videos = cacheable_content[currentSP]
                    # 考虑更多的视频选项，不仅仅是当前指针位置的视频
                    max_video_offset = min(pointer_vec[currentSP] + 30, len(cacheable_videos))
                    for video_offset in range(pointer_vec[currentSP], max_video_offset):
                        video_idx = cacheable_videos[video_offset]
                        
                        # 检查邻居节点是否已经缓存了该视频
                        neighbor_has_video = False
                        for neighbor in cache_node_neighbors[node]:
                            if neighbor in allocation:
                                # 检查邻居节点是否有足够的缓存分配
                                if allocation[neighbor][currentSP] > video_offset:
                                    neighbor_has_video = True
                                    break
                        
                        # 计算当前SP的得分，考虑邻居节点的缓存情况和全局信息
                        # 本地得分
                        local_score = distrib[currentSP][video_idx] * SP_proba[currentSP]
                        
                        # 全局得分：使用预计算的全局重要性
                        global_score = video_global_importance.get((currentSP, video_idx), 0)
                        
                        # 动态调整权重，基于节点重要性、中心性和网络规模
                        node_imp = node_importance.get(node, 1)
                        max_centrality = max(node_centrality.values()) if node_centrality else 1
                        if max_centrality == 0:
                            max_centrality = 1
                        node_cen = node_centrality.get(node, 0) / max_centrality
                        # 网络规模越大，全局权重越高，但在小网络中也保持较高的全局权重
                        local_weight = (0.3 + 0.15 * (1 - node_imp)) * (1 - network_size_factor * 0.15)
                        global_weight = (0.5 + 0.1 * node_cen) * (1 + network_size_factor * 0.15)
                        # 确保权重和为1
                        total_weight = local_weight + global_weight
                        local_weight /= total_weight
                        global_weight /= total_weight
                        
                        # 综合得分：本地得分和全局得分的加权和
                        total_score = local_weight * local_score + global_weight * global_score
                        
                        # 如果邻居节点已经缓存了该视频，降低得分
                        if neighbor_has_video:
                            score = total_score * 0.05  # 进一步降低权重，避免重复缓存
                        else:
                            score = total_score  # 邻居节点未缓存，保持原始得分
                        
                        if score > best_score:
                            best_score = score
                            bestSP = currentSP
                            best_video_offset = video_offset
            
            # 分配slot给最佳SP和视频
            if bestSP is not None and best_video_offset != -1:
                # 直接使用找到的video_offset，避免index查找
                pointer_vec[bestSP] = best_video_offset + 1
                # 确保指针向量不超过缓存视频数量
                pointer_vec[bestSP] = min(pointer_vec[bestSP], len(cacheable_content[bestSP]) if cacheable_content and bestSP < len(cacheable_content) else pointer_vec[bestSP])
                allocation[node][bestSP] += 1
    
    # 多轮迭代优化
    max_iterations = 15  # 增加迭代次数
    for iteration in range(max_iterations):
        # 尝试优化分配
        improved = False
        
        # 计算当前全局缓存分布
        video_cache_frequency, video_cache_coverage = calculate_global_cache_distribution(allocation)
        
        # 计算全局缓存均衡性指标
        global_cache_balance = {}
        for sp in range(nSP):
            sp_allocation = sum(allocation[node][sp] for node in cache_nodes)
            global_cache_balance[sp] = sp_allocation / sum(sp_global_importance.values()) if sum(sp_global_importance.values()) > 0 else 0
        
        for node in cache_nodes:
            actual_capacity = node_capacities[node]
            current_allocation = sum(allocation[node])
            
            # 检查是否有可以调整的空间
            if current_allocation == actual_capacity:
                # 计算每个视频的得分
                video_scores = {}
                for sp in range(nSP):
                    if cacheable_content and sp < len(cacheable_content):
                        cacheable_videos = cacheable_content[sp]
                        for video_idx in range(min(allocation[node][sp], 300)):  # 扩大计算范围
                            video = cacheable_videos[video_idx]
                            video_key = (sp, video)
                            # 计算视频的得分
                            local_score = distrib[sp][video] * SP_proba[sp]
                            global_score = video_global_importance.get(video_key, 0)
                            
                            # 全局缓存频率得分：优先缓存缓存频率低的视频
                            cache_frequency = video_cache_frequency.get(video_key, 0)
                            frequency_score = 1.0 / (1 + cache_frequency)  # 缓存频率越低，得分越高
                            
                            # 全局缓存覆盖率得分：优先缓存覆盖率低的视频
                            coverage = video_cache_coverage.get(video_key, 0)
                            coverage_score = 1.0 / (1 + coverage)  # 覆盖率越低，得分越高
                            
                            # 缓存均衡性得分：优先缓存分配比例低的SP的视频
                            balance_score = 1.0 / (1 + abs(global_cache_balance.get(sp, 0) - sp_importance_ratio.get(sp, 0)))
                            
                            # 动态调整权重，基于节点重要性、中心性和网络规模
                            node_imp = node_importance.get(node, 1)
                            max_centrality = max(node_centrality.values()) if node_centrality else 1
                            if max_centrality == 0:
                                max_centrality = 1
                            node_cen = node_centrality.get(node, 0) / max_centrality
                            # 网络规模越大，全局权重越高，但在小网络中也保持较高的全局权重
                            local_weight = (0.2 + 0.15 * (1 - node_imp)) * (1 - network_size_factor * 0.15)
                            global_weight = (0.4 + 0.1 * node_cen) * (1 + network_size_factor * 0.15)
                            frequency_weight = 0.1
                            coverage_weight = 0.1
                            balance_weight = 0.1  # 增加缓存均衡性权重
                            
                            # 确保权重和为1
                            total_weight = local_weight + global_weight + frequency_weight + coverage_weight + balance_weight
                            local_weight /= total_weight
                            global_weight /= total_weight
                            frequency_weight /= total_weight
                            coverage_weight /= total_weight
                            balance_weight /= total_weight
                            
                            # 综合得分：本地得分、全局得分、频率得分、覆盖率得分和均衡性得分的加权和
                            total_score = local_weight * local_score + global_weight * global_score + frequency_weight * frequency_score + coverage_weight * coverage_score + balance_weight * balance_score
                            video_scores[video_key] = total_score
                
                # 按得分排序视频
                sorted_videos = sorted(video_scores.items(), key=lambda x: x[1], reverse=True)
                
                # 尝试将低得分视频的缓存空间转移到高得分视频
                low_score_videos = sorted(video_scores.items(), key=lambda x: x[1])[:30]  # 增加低得分视频调整数量
                for (sp1, video1), score1 in low_score_videos:
                    if allocation[node][sp1] <= 0:
                        continue
                    
                    for (sp2, video2), score2 in sorted_videos:
                        if sp1 == sp2 and video1 == video2:
                            continue
                        
                        # 检查邻居在sp2的缓存情况
                        neighbor_sp2_allocation = 0
                        for neighbor in cache_node_neighbors[node]:
                            neighbor_sp2_allocation += allocation[neighbor][sp2]
                        
                        # 检查视频2的全局缓存情况
                        video2_key = (sp2, video2)
                        video2_frequency = video_cache_frequency.get(video2_key, 0)
                        video2_coverage = video_cache_coverage.get(video2_key, 0)
                        
                        # 如果邻居在sp2的缓存较少，视频2得分更高，且全局缓存频率较低，尝试转移
                        if neighbor_sp2_allocation < allocation[node][sp2] + 1 and score2 > score1 * 1.01 and video2_frequency < 3 and video2_coverage < 0.5:
                            # 使用预构建的查找表获取视频2的位置
                            if sp2 in video_to_index and video2 in video_to_index[sp2]:
                                video2_position = video_to_index[sp2][video2]
                                # 确保视频2位置不超过当前分配
                                if video2_position < allocation[node][sp2] + 1:
                                    allocation[node][sp1] -= 1
                                    allocation[node][sp2] += 1
                                    improved = True
                                    break
                    if improved:
                        break
        
        # 全局级优化：跨节点缓存协调
        if iteration % 2 == 1:  # 每两轮迭代进行一次全局优化
            # 识别过度缓存和缓存不足的视频
            video_cache_frequency, video_cache_coverage = calculate_global_cache_distribution(allocation)
            over_cached = [k for k, v in video_cache_frequency.items() if v > 3]
            under_cached = sorted(video_global_importance.items(), key=lambda x: x[1], reverse=True)[:150]  # 增加缓存不足视频数量
            under_cached = [k for k, v in under_cached if video_cache_frequency.get(k, 0) < 2 and video_cache_coverage.get(k, 0) < 0.3]
            
            # 尝试将过度缓存的视频空间转移到缓存不足的视频
            for over_key in over_cached:
                sp1, video1 = over_key
                # 找到缓存了该视频的节点
                for node in cache_nodes:
                    if allocation[node][sp1] > 0:
                        # 使用预构建的查找表获取视频1的位置
                        if sp1 in video_to_index and video1 in video_to_index[sp1]:
                            video_position = video_to_index[sp1][video1]
                            if video_position < allocation[node][sp1]:
                                # 寻找缓存不足的视频
                                for under_key in under_cached:
                                    sp2, video2 = under_key
                                    # 使用预构建的查找表检查视频2
                                    if sp2 in video_to_index and video2 in video_to_index[sp2]:
                                        # 检查邻居缓存情况
                                        neighbor_has_video = False
                                        for neighbor in cache_node_neighbors[node]:
                                            if neighbor in allocation:
                                                if allocation[neighbor][sp2] > video_to_index[sp2][video2]:
                                                    neighbor_has_video = True
                                                    break
                                        if not neighbor_has_video:
                                            # 转移缓存空间
                                            allocation[node][sp1] -= 1
                                            allocation[node][sp2] += 1
                                            improved = True
                                            break
                    if improved:
                        break
        
        if not improved:
            break
    
    end_time = time.time()
    print(f"全局最优缓存分配方法执行时间: {end_time - start_time:.2f}秒")
    
    return allocation

def find_epsilon(time):
    A = 0.5  # 提高初始探索率
    B = 0.15
    C = 0.003  # 降低衰减速度
    
    standardized_time = (time - B * simulation_time) / (A * simulation_time)
    cosh = np.cosh(math.exp(-standardized_time))
    epsilon = 0.9 - (0.8 / cosh + (time * C / simulation_time))
    return max(epsilon, 0.15)  # 提高最小探索率

def find_N(time):
    """
    计算随时间变化的小批量大小N
    """
    D = 0.15
    E = 0.3
    F = 0.7
    
    standardized_time = (time - D * simulation_time) / (E * simulation_time)
    cosh = np.cosh(math.exp(-standardized_time))
    N = round((100 / cosh + (time * F / simulation_time)))
    return N

def evaluate_cost(allocation, first_alloc, best_alloc, requests_nb, video_probabi, fixed_requests=None, debug_output=False):
    """
    评估不同分配的成本
    支持多个缓存节点的情况
    
    参数:
    allocation: 当前分配
    first_alloc: 初始分配
    best_alloc: 最佳分配
    requests_nb: 请求数量
    video_probabi: 视频概率分布
    fixed_requests: 固定的请求序列，如果为None则随机生成
    debug_output: 是否输出调试信息
    """
    cost = 0
    b_cost = 0
    f_cost = 0
    total_latency = 0

    # 为每个缓存节点计算单独的命中率
    cache_node_hits = {}
    cache_nodes = topology_manager.get_cache_nodes()

    # 如果没有缓存节点，返回最大成本（全部未命中）
    if len(cache_nodes) == 0:
        # 没有缓存节点时，所有请求都未命中
        avg_latency = user_to_cache_latency + cache_to_sp_latency
        cache_node_hit_rates = {}
        return (1.0, 1.0, 1.0, avg_latency, cache_node_hit_rates)

    for node in cache_nodes:
        cache_node_hits[node] = 0
    
    # 为每个缓存节点创建邻居节点列表（根据拓扑确定）
    cache_node_neighbors = {}
    for node in cache_nodes:
        # 使用拓扑管理器的get_neighbors方法获取真正的邻居
        all_neighbors = topology_manager.get_neighbors(node)
        # 过滤出也是缓存节点的邻居
        neighbors = [n for n in all_neighbors if n in cache_nodes]
        cache_node_neighbors[node] = neighbors
    
    # 获取SP节点和接收器节点
    sp_nodes = topology_manager.get_sp_nodes()
    receiver_nodes = topology_manager.get_receiver_nodes()
    router_nodes = topology_manager.get_router_nodes()

    # 生成或使用固定的请求序列
    requests = []
    source_nodes = []

    if fixed_requests:
        # 检查fixed_requests的类型，兼容两种格式
        if isinstance(fixed_requests, dict) and 'requests' in fixed_requests:
            # 格式1: 字典格式，包含'requests'和'source_nodes'
            requests = fixed_requests['requests']
            source_nodes = fixed_requests.get('source_nodes', [])
        elif isinstance(fixed_requests, list):
            # 格式2: 列表格式，只包含请求元组 (SP, video_id)
            requests = fixed_requests
            # 为每个请求生成源节点
            for request in requests:
                # 请求源可以是路由器节点或缓存节点（代表用户终端）
                possible_sources = []
                if router_nodes:
                    possible_sources.extend(router_nodes)
                if cache_nodes:
                    possible_sources.extend(cache_nodes)
                
                if possible_sources:
                    source_node = rd.choice(possible_sources)
                else:
                    source_node = None
                source_nodes.append(source_node)
        else:
            # 未知格式，使用默认处理
            requests = []
            source_nodes = []
    else:
        # 生成随机请求序列
        for r in range(requests_nb):
            request = request_creation(video_probabi)
            requests.append(request)

            # 请求源可以是路由器节点或缓存节点（代表用户终端）
            possible_sources = []
            if router_nodes:
                possible_sources.extend(router_nodes)
            if cache_nodes:
                possible_sources.extend(cache_nodes)

            if possible_sources:
                source_node = rd.choice(possible_sources)
            else:
                source_node = None
            source_nodes.append(source_node)
    
    # 处理请求
    processed_requests = 0
    
    # 输出调试信息：请求列表
    if debug_output:
        print(f"\n=== 当前Interval请求列表 ===")
        print(f"请求数量: {min(requests_nb, len(requests))}")
        for r in range(min(requests_nb, len(requests))):
            request = requests[r]
            source_node = source_nodes[r]
            if source_node is not None:
                print(f"  请求{r+1}: SP{request[0]}, 视频ID{request[1]}, 接收节点{source_node}")
    
    for r in range(min(requests_nb, len(requests))):
        request = requests[r]
        source_node = source_nodes[r]
        
        # 跳过无效的源节点
        if source_node is None:
            continue
            
        SP_of_the_video_requested = request[0]
        video_id = request[1]

        processed_requests += 1

        # 找到最近的缓存节点
        closest_cache = None
        min_latency = float('inf')

        for cache_node in cache_nodes:
            latency = topology_manager.get_latency(source_node, cache_node)
            if latency < min_latency:
                min_latency = latency
                closest_cache = cache_node

        # 检查最近的缓存节点是否有缓存
        cache_hit = False
        hit_node = None
        # 时延 = 用户到接入节点的固定时延(user_to_cache_latency) + 接入节点到缓存节点的链路时延
        latency = user_to_cache_latency + min_latency
        
        if closest_cache in allocation:
            # 首先检查该视频是否可缓存
            if cacheable_content and SP_of_the_video_requested < len(cacheable_content):
                cacheable_videos = cacheable_content[SP_of_the_video_requested]
                # 检查视频是否在可缓存列表中
                if video_id in cacheable_videos:
                    # 获取视频在可缓存列表中的索引
                    video_idx = cacheable_videos.index(video_id)
                    # 检查缓存分配是否足够 - 统一使用 > video_idx 条件
                    if allocation[closest_cache][SP_of_the_video_requested] > video_idx:  # 统一边界：分配必须大于视频索引
                        cache_hit = True
                        hit_node = closest_cache
                        
                        # 输出缓存命中详细信息
                        if debug_output:
                            # 获取该节点缓存的所有内容
                            cached_content = []
                            if cacheable_content and SP_of_the_video_requested < len(cacheable_content):
                                cacheable_videos = cacheable_content[SP_of_the_video_requested]
                                alloc_size = allocation[closest_cache][SP_of_the_video_requested]
                                # 确保alloc_size是整数
                                alloc_size_int = int(alloc_size) if isinstance(alloc_size, (int, float)) else 0
                                # 只显示实际缓存的内容，不超过alloc_size_int
                                cached_content = cacheable_videos[:alloc_size_int]
                            
                            print(f"  [缓存命中] 请求{r+1}: SP{SP_of_the_video_requested}, 视频ID{video_id}, 接收节点{source_node}")
                            print(f"          命中节点: {hit_node}, 视频索引: {video_idx}")
                            print(f"          节点{hit_node}缓存内容(SP{SP_of_the_video_requested}): {cached_content[:10]}{'...' if len(cached_content) > 10 else ''}")
                            print(f"          实际分配空间: {alloc_size_int}, 可缓存视频总数: {len(cacheable_videos) if cacheable_content and SP_of_the_video_requested < len(cacheable_content) else 0}")
        
        # 如果缓存未命中，且启用了协同缓存，尝试邻居节点
        if not cache_hit and cooperative_caching:
            # 遍历邻居节点
            for neighbor_node in cache_node_neighbors[closest_cache]:
                if neighbor_node in allocation:
                    # 首先检查该视频是否可缓存
                    if cacheable_content and SP_of_the_video_requested < len(cacheable_content):
                        cacheable_videos = cacheable_content[SP_of_the_video_requested]
                        # 检查视频是否在可缓存列表中
                        if video_id in cacheable_videos:
                            # 获取视频在可缓存列表中的索引
                            video_idx = cacheable_videos.index(video_id)
                            # 检查缓存分配是否足够
                            if allocation[neighbor_node][SP_of_the_video_requested] > video_idx:
                                cache_hit = True
                                hit_node = neighbor_node
                                # 增加缓存节点到邻居节点的时延
                                latency += cache_to_cache_latency
                                
                                # 输出协同缓存命中详细信息
                                if debug_output:
                                    # 获取该节点缓存的所有内容
                                    cached_content = []
                                    if cacheable_content and SP_of_the_video_requested < len(cacheable_content):
                                        cacheable_videos = cacheable_content[SP_of_the_video_requested]
                                        alloc_size = allocation[neighbor_node][SP_of_the_video_requested]
                                        # 确保alloc_size是整数
                                        alloc_size_int = int(alloc_size) if isinstance(alloc_size, (int, float)) else 0
                                        # 只显示实际缓存的内容，不超过alloc_size_int
                                        cached_content = cacheable_videos[:alloc_size_int]
                                    
                                    print(f"  [协同缓存命中] 请求{r+1}: SP{SP_of_the_video_requested}, 视频ID{video_id}, 接收节点{source_node}")
                                    print(f"              命中节点: {hit_node}, 视频索引: {video_idx}")
                                    print(f"              节点{hit_node}缓存内容(SP{SP_of_the_video_requested}): {cached_content[:10]}{'...' if len(cached_content) > 10 else ''}")
                                    print(f"              实际分配空间: {alloc_size_int}, 可缓存视频总数: {len(cacheable_videos) if cacheable_content and SP_of_the_video_requested < len(cacheable_content) else 0}")
                                break
        
        # 如果所有缓存节点都未命中，从SP节点获取
        if not cache_hit:
            cost += 1
            # 增加未命中时到达SP的时延
            # 1. 添加从最近缓存节点到SP节点的链路时延
            sp_node = sp_nodes[SP_of_the_video_requested] if sp_nodes and SP_of_the_video_requested < len(sp_nodes) else None
            if sp_node and closest_cache:
                sp_latency = topology_manager.get_latency(closest_cache, sp_node)
                latency += sp_latency
            # 2. 添加固定的SP访问时延
            latency += cache_to_sp_latency
        else:
            # 记录命中的缓存节点
            if hit_node:
                cache_node_hits[hit_node] += 1
                # 记录请求历史，用于在线学习
                record_request(hit_node, SP_of_the_video_requested, video_id)
        
        # 累加时延
        total_latency += latency
        
        # 计算最佳分配和初始分配的成本
        if isinstance(best_alloc, dict):
            # 检查最佳分配是否命中
            best_cache_hit = False
            for cache_node in cache_nodes:
                if cache_node in best_alloc:
                    # 首先检查该视频是否可缓存
                    if cacheable_content and SP_of_the_video_requested < len(cacheable_content):
                        cacheable_videos = cacheable_content[SP_of_the_video_requested]
                        # 检查视频是否在可缓存列表中
                        if video_id in cacheable_videos:
                            # 获取视频在可缓存列表中的索引
                            video_idx = cacheable_videos.index(video_id)
                            # 检查缓存分配是否足够
                            if best_alloc[cache_node][SP_of_the_video_requested] > video_idx:
                                best_cache_hit = True
                                break
            if not best_cache_hit:
                b_cost += 1
        else:
            # 兼容旧格式
            # 首先检查该视频是否可缓存
            if cacheable_content and SP_of_the_video_requested < len(cacheable_content):
                cacheable_videos = cacheable_content[SP_of_the_video_requested]
                # 检查视频是否在可缓存列表中
                if video_id in cacheable_videos:
                    # 获取视频在可缓存列表中的索引
                    video_idx = cacheable_videos.index(video_id)
                    # 检查缓存分配是否足够
                    b_allocated_cache_space = best_alloc[SP_of_the_video_requested]
                    if b_allocated_cache_space <= video_idx:
                        b_cost += 1
            else:
                # 如果不可缓存，直接计入成本
                b_cost += 1
        
        if isinstance(first_alloc, dict):
            # 检查初始分配是否命中
            first_cache_hit = False
            for cache_node in cache_nodes:
                if cache_node in first_alloc:
                    # 首先检查该视频是否可缓存
                    if cacheable_content and SP_of_the_video_requested < len(cacheable_content):
                        cacheable_videos = cacheable_content[SP_of_the_video_requested]
                        # 检查视频是否在可缓存列表中
                        if video_id in cacheable_videos:
                            # 获取视频在可缓存列表中的索引
                            video_idx = cacheable_videos.index(video_id)
                            # 检查缓存分配是否足够
                            if first_alloc[cache_node][SP_of_the_video_requested] > video_idx:
                                first_cache_hit = True
                                break
            if not first_cache_hit:
                f_cost += 1
        else:
            # 兼容旧格式
            # 首先检查该视频是否可缓存
            if cacheable_content and SP_of_the_video_requested < len(cacheable_content):
                cacheable_videos = cacheable_content[SP_of_the_video_requested]
                # 检查视频是否在可缓存列表中
                if video_id in cacheable_videos:
                    # 获取视频在可缓存列表中的索引
                    video_idx = cacheable_videos.index(video_id)
                    # 检查缓存分配是否足够
                    f_allocated_cache_space = first_alloc[SP_of_the_video_requested]
                    if f_allocated_cache_space <= video_idx:
                        f_cost += 1
            else:
                # 如果不可缓存，直接计入成本
                f_cost += 1
    
    # 归一化成本（使用实际处理的请求数量）
    actual_requests = processed_requests if processed_requests > 0 else 1
    cost = cost / actual_requests
    b_cost = b_cost / actual_requests
    f_cost = f_cost / actual_requests
    avg_latency = total_latency / actual_requests if network_enabled else 0
    
    # 计算每个缓存节点的命中率（使用实际处理的请求数量）
    cache_node_hit_rates = {}
    for node in cache_nodes:
        cache_node_hit_rates[node] = cache_node_hits[node] / actual_requests
    
    return (cost, f_cost, b_cost, avg_latency, cache_node_hit_rates)

def states_nSP(capacity, numberSP, delta2):
    """
    生成所有可能的状态（缓存分配）
    修复状态生成问题，确保包含更多可能的状态
    """
    MAX_STATES = 1000000  # 最大状态数量限制，避免内存溢出
    state_count = 0  # 状态计数器
    
    def generate_states(cap, sp, delta):
        """
        递归生成状态
        """
        nonlocal state_count
        states = []
        
        if sp == 1:
            if state_count < MAX_STATES:
                states.append([cap])
                state_count += 1
            return states
        elif sp == 2:
            # 生成所有可能的分配组合（不使用delta限制）
            for j in range(cap + 1):
                if state_count >= MAX_STATES:
                    break
                states.append([j, cap - j])
                state_count += 1
            return states
        else:
            # 生成所有可能的分配组合（不使用delta限制）
            for i in range(cap + 1):
                if state_count >= MAX_STATES:
                    break
                remaining_cap = cap - i
                other_states = generate_states(remaining_cap, sp - 1, delta)
                for state in other_states:
                    if state_count >= MAX_STATES:
                        break
                    new_state = state.copy()
                    new_state.append(i)
                    states.append(new_state)
                    state_count += 1
            return states
    
    # 生成状态（使用较小的delta值来生成更多状态）
    effective_delta = max(1, min(delta2, 3))  # 使用较小的delta值
    all_states = generate_states(capacity, numberSP, effective_delta)
    
    # 打印状态空间大小
    #print(f"状态空间大小: {len(all_states)}")
    
    # 确保最佳分配在状态空间中
    # 注意：不重新调用catalog()函数，以保持cacheable_content和最佳分配的一致性
    # 使用全局变量videos_proba
    global videos_proba
    best_allocation = decide_opt_alloc(videos_proba)
    
    for node, alloc in best_allocation.items():
        # 四舍五入到最近的delta2倍数
        rounded_alloc = [round(a / effective_delta) * effective_delta for a in alloc]
        # 调整总和为capacity
        total = sum(rounded_alloc)
        if total != capacity:
            diff = capacity - total
            # 调整最大的分配以保持总和正确
            max_idx = rounded_alloc.index(max(rounded_alloc))
            rounded_alloc[max_idx] += diff
        # 确保所有值非负
        rounded_alloc = [max(0, a) for a in rounded_alloc]
        # 如果四舍五入后的分配不在状态空间中，添加它
        if rounded_alloc not in all_states and state_count < MAX_STATES:
            all_states.append(rounded_alloc)
            state_count += 1
            print(f"添加四舍五入后的最佳分配到状态空间: {rounded_alloc}")
    
    return all_states

def get_state_index(alloc, delta, states):
    """
    获取给定分配在状态列表中的索引
    """
    # 使用原始顺序的分配，保留SP的顺序信息
    original_alloc = tuple(alloc)
    for state_index, state in enumerate(states):
        # 直接比较原始顺序的状态
        if tuple(state) == original_alloc:
            return state_index
    # 如果没有找到，返回-1
    return -1

def take_action(allocation, epsilon, D, delta, Q, state_index):
    """
    执行动作（单缓存节点版本）
    """
    old_allocation = deepcopy(allocation)
    D_size = len(D)
    alea = rd.random()
    coeff_ind = rd.randint(0, D_size-1)
    coeff = D[coeff_ind]
    
    if alea <= epsilon:  # epsilon-greedy策略
        action = rd.randint(0, nSP**2 - 1)
        action_plus = action // nSP
        action_minus = action % nSP
        if action_plus < len(allocation) and action_minus < len(allocation):
            allocation[action_plus] += coeff * delta
            allocation[action_minus] -= coeff * delta
        else:
            return (action, action_minus, action_plus, old_allocation)
    else:
        # 确保 q_values 是一维数组
        q_values = Q[:, state_index]
        q_values = q_values.flatten()[:nSP**2]  # 只取前 nSP**2 个元素
        (best_score, best_actions) = af.find_max_list(q_values)
        # 确保 action 在有效范围内
        if best_actions:
            # 只选择前 nSP**2 范围内的动作
            valid_actions = [a for a in best_actions if a < nSP**2]
            if valid_actions:
                action = rd.choice(valid_actions)
            else:
                action = rd.randint(0, nSP**2 - 1)
        else:
            action = rd.randint(0, nSP**2 - 1)
        action_plus = action // nSP
        action_minus = action % nSP
        if action_plus < len(allocation) and action_minus < len(allocation):
            allocation[action_plus] += coeff * delta
            allocation[action_minus] -= coeff * delta
        else:
            return (action, action_minus, action_plus, old_allocation)
    
    # 检查分配是否有效
    for SP_cache in allocation:
        if SP_cache < 0 or SP_cache > single_cache_capacity:
            allocation = old_allocation
            Q[action][state_index] = 0
    
    return (action, action_minus, action_plus, allocation)

def take_action_multi_cache(allocation, epsilon, D, delta, Q, state_index, cache_nodes, nSP, states, state_to_index):
    """
    执行多缓存节点的动作
    为每个缓存节点选择一个动作，在不同SP之间转移缓存空间
    修复delta值过大的问题，使用更小的动作步长
    """
    old_allocation = deepcopy(allocation)
    D_size = len(D)
    
    # 为每个缓存节点选择动作
    actions = {}
    action_plus = {}
    action_minus = {}
    
    # 调整delta值，避免过大导致无效动作
    adjusted_delta = min(delta, 2)  # 限制delta最大为2
    
    for node in cache_nodes:
        # 获取该节点的当前状态索引
        node_state_index = state_index[node]
        
        # 为每个节点独立选择系数和随机数
        alea = rd.random()
        coeff_ind = rd.randint(0, D_size-1) if D_size > 0 else 0
        coeff = D[coeff_ind] if D_size > 0 else 1
        
        # 进一步调整动作步长，确保动作有效
        action_step = min(coeff * adjusted_delta, 1)  # 最大步长为1
        
        if alea <= epsilon:  # epsilon-greedy策略（探索）
            # 随机选择一个动作
            action = rd.randint(0, nSP**2 - 1)
            action_plus_val = action // nSP
            action_minus_val = action % nSP
            
            # 跳过无效动作（plus和minus相同）
            if action_plus_val == action_minus_val:
                action_plus[node] = None
                action_minus[node] = None
                actions[node] = None
                continue
            
            # 尝试执行动作，如果无效则回滚
            temp_allocation = allocation[node].copy()
            temp_allocation[action_plus_val] += action_step
            temp_allocation[action_minus_val] -= action_step
            
            # 检查动作是否有效
            valid = True
            node_capacity = single_cache_capacity
            for sp in range(nSP):
                if temp_allocation[sp] < 0 or temp_allocation[sp] > node_capacity:
                    valid = False
                    break
            
            if valid:
                allocation[node] = temp_allocation
                action_plus[node] = action_plus_val
                action_minus[node] = action_minus_val
                actions[node] = action
            else:
                # 动作无效，保持原分配
                action_plus[node] = None
                action_minus[node] = None
                actions[node] = None
                
        else:  # 利用策略（基于Q表）
            # 基于Q表选择最佳动作
            q_values = Q[node][:, node_state_index].flatten()
            (best_score, best_actions) = af.find_max_list(q_values)
            
            if best_actions:
                # 随机选择一个最佳动作
                action = rd.choice(best_actions)
                action_plus_val = action // nSP
                action_minus_val = action % nSP
                
                # 跳过无效动作
                if action_plus_val == action_minus_val:
                    action_plus[node] = None
                    action_minus[node] = None
                    actions[node] = None
                    continue
                
                # 尝试执行动作，如果无效则回滚
                temp_allocation = allocation[node].copy()
                temp_allocation[action_plus_val] += action_step
                temp_allocation[action_minus_val] -= action_step
                
                # 检查动作是否有效
                valid = True
                node_capacity = single_cache_capacity
                for sp in range(nSP):
                    if temp_allocation[sp] < 0 or temp_allocation[sp] > node_capacity:
                        valid = False
                        break
                
                if valid:
                    allocation[node] = temp_allocation
                    action_plus[node] = action_plus_val
                    action_minus[node] = action_minus_val
                    actions[node] = action
                else:
                    # 动作无效，保持原分配
                    action_plus[node] = None
                    action_minus[node] = None
                    actions[node] = None
            else:
                # 没有最佳动作，随机选择一个
                action = rd.randint(0, nSP**2 - 1)
                action_plus_val = action // nSP
                action_minus_val = action % nSP
                
                # 跳过无效动作
                if action_plus_val == action_minus_val:
                    action_plus[node] = None
                    action_minus[node] = None
                    actions[node] = None
                    continue
                
                # 尝试执行动作
                temp_allocation = allocation[node].copy()
                temp_allocation[action_plus_val] += action_step
                temp_allocation[action_minus_val] -= action_step
                
                # 检查动作是否有效
                valid = True
                node_capacity = single_cache_capacity
                for sp in range(nSP):
                    if temp_allocation[sp] < 0 or temp_allocation[sp] > node_capacity:
                        valid = False
                        break
                
                if valid:
                    allocation[node] = temp_allocation
                    action_plus[node] = action_plus_val
                    action_minus[node] = action_minus_val
                    actions[node] = action
                else:
                    # 动作无效，保持原分配
                    action_plus[node] = None
                    action_minus[node] = None
                    actions[node] = None
    
    # 更新状态索引
    for node in cache_nodes:
        original_alloc = tuple(allocation[node])
        new_state_index = state_to_index[node].get(original_alloc, -1)
        if new_state_index != -1:
            state_index[node] = new_state_index
    
    return (actions, action_minus, action_plus, allocation)

def optimize_nSP(allocation, initial_videos_proba, best_allocation, request_rate, nb_interval, interval_size, gama, delta, D, method, fixed_requests=None, debug_interval=None):
    """
    优化缓存分配
    支持多个缓存节点的情况
    Q-learning 部分已按新设计实现：
    - 奖励 = 命中率变化量 (old_hit_rate - new_hit_rate) 即 (old_nominal_cost - new_nominal_cost)
    - 动作 = 单位缓存移动（coeff * delta，通常coeff=1）
    - ε-greedy 策略，合法动作检查
    
    参数:
    debug_interval: 要输出调试信息的interval编号，如果为None则禁用调试输出
    """
    # 声明全局变量
    global save_allocations, results_dir, epsilon_decay, alpha_scheduling, activate_memory
    global single_cache_capacity, nSP, topology_manager
    global cacheable_content, videos_proba, SP_proba
    
    # 如果方法是best_allocation，直接返回最佳分配
    if method == 'best_allocation':
        print("使用最佳分配方法")
        print(f"最佳分配: {best_allocation}")
        # 计算最佳分配的成本
        request_nb = int(interval_size * request_rate)
        
        # 为每个interval计算成本
        L_total_cost = []
        L_nominal_cost = []
        L_first_cost = []
        L_best_cost = []
        L_avg_latency = []
        
        for interval in range(nb_interval):
            # 使用当前interval的固定请求
            interval_fixed_requests = fixed_requests[interval] if fixed_requests and len(fixed_requests) > interval else None
            # 检查是否需要输出调试信息
            debug_output = (debug_interval is not None and interval == debug_interval)
            (nominal_cost, first_cost, best_cost, avg_latency, cache_node_hit_rates) = evaluate_cost(
                best_allocation, allocation, best_allocation, request_nb, initial_videos_proba, interval_fixed_requests, debug_output
            )
            
            L_total_cost.append(nominal_cost)
            L_nominal_cost.append(nominal_cost)
            L_first_cost.append(first_cost)
            L_best_cost.append(best_cost)
            L_avg_latency.append(avg_latency)
        
        print(f"最佳分配成本: {sum(L_nominal_cost) / len(L_nominal_cost):.4f}")
        print(f"最佳分配命中率: {cache_node_hit_rates}")
        
        # 准备输出文件
        print(f"保存分配历史: {save_allocations}")
        print(f"结果目录: {results_dir}")
        print(f"单个缓存节点容量: {single_cache_capacity}")
        print(f"请求率: {request_rate}")
        allocations_file = os.path.join(results_dir, f"allocations_single_cache_capacity{single_cache_capacity}_request_rate{request_rate}.txt")
        print(f"分配文件路径: {allocations_file}")
        f = open(allocations_file, "w") if save_allocations else None
        print(f"文件对象: {f}")
        
        # 保存最佳分配
        if save_allocations and f:
            f.write(str(best_allocation) + '\n')
        
        if f:
            f.close()
        
        # 返回最佳分配和成本列表
        return best_allocation, L_total_cost, L_nominal_cost, L_first_cost, L_best_cost, L_avg_latency
    
    # 如果方法是cooperative_best_allocation，使用考虑邻居协作的最佳分配方法
    elif method == 'cooperative_best_allocation':
        print("使用考虑邻居协作的最佳分配方法")
        # 计算考虑邻居协作的最佳分配
        cooperative_allocation = decide_cooperative_opt_alloc(initial_videos_proba)
        print(f"考虑邻居协作的最佳分配: {cooperative_allocation}")
        # 计算最佳分配的成本
        request_nb = int(interval_size * request_rate)
        
        # 为每个interval计算成本
        L_total_cost = []
        L_nominal_cost = []
        L_first_cost = []
        L_best_cost = []
        L_avg_latency = []
        
        for interval in range(nb_interval):
            # 使用当前interval的固定请求
            interval_fixed_requests = fixed_requests[interval] if fixed_requests and len(fixed_requests) > interval else None
            # 检查是否需要输出调试信息
            debug_output = (debug_interval is not None and interval == debug_interval)
            (nominal_cost, first_cost, best_cost, avg_latency, cache_node_hit_rates) = evaluate_cost(
                cooperative_allocation, allocation, best_allocation, request_nb, initial_videos_proba, interval_fixed_requests, debug_output
            )
            
            L_total_cost.append(nominal_cost)
            L_nominal_cost.append(nominal_cost)
            L_first_cost.append(first_cost)
            L_best_cost.append(best_cost)
            L_avg_latency.append(avg_latency)
        
        print(f"考虑邻居协作的最佳分配成本: {sum(L_nominal_cost) / len(L_nominal_cost):.4f}")
        print(f"考虑邻居协作的最佳分配命中率: {cache_node_hit_rates}")
        
        # 准备输出文件
        print(f"保存分配历史: {save_allocations}")
        print(f"结果目录: {results_dir}")
        print(f"单个缓存节点容量: {single_cache_capacity}")
        print(f"请求率: {request_rate}")
        allocations_file = os.path.join(results_dir, f"allocations_single_cache_capacity{single_cache_capacity}_request_rate{request_rate}.txt")
        print(f"分配文件路径: {allocations_file}")
        f = open(allocations_file, "w") if save_allocations else None
        print(f"文件对象: {f}")
        
        # 保存最佳分配
        if save_allocations and f:
            f.write(str(cooperative_allocation) + '\n')
        
        if f:
            f.close()
        
        # 返回最佳分配和成本列表
        return cooperative_allocation, L_total_cost, L_nominal_cost, L_first_cost, L_best_cost, L_avg_latency
    
    # 如果方法是global_opt_allocation，使用全局最优缓存分配方法
    elif method == 'global_opt_allocation':
        print("使用全局最优缓存分配方法")
        # 计算全局最优缓存分配
        global_allocation = decide_global_opt_alloc(initial_videos_proba)
        print(f"全局最优缓存分配: {global_allocation}")
        # 计算最佳分配的成本
        request_nb = int(interval_size * request_rate)
        
        # 为每个interval计算成本
        L_total_cost = []
        L_nominal_cost = []
        L_first_cost = []
        L_best_cost = []
        L_avg_latency = []
        
        for interval in range(nb_interval):
            # 使用当前interval的固定请求
            interval_fixed_requests = fixed_requests[interval] if fixed_requests and len(fixed_requests) > interval else None
            # 检查是否需要输出调试信息
            debug_output = (debug_interval is not None and interval == debug_interval)
            (nominal_cost, first_cost, best_cost, avg_latency, cache_node_hit_rates) = evaluate_cost(
                global_allocation, allocation, best_allocation, request_nb, initial_videos_proba, interval_fixed_requests, debug_output
            )
            
            L_total_cost.append(nominal_cost)
            L_nominal_cost.append(nominal_cost)
            L_first_cost.append(first_cost)
            L_best_cost.append(best_cost)
            L_avg_latency.append(avg_latency)
        
        print(f"全局最优缓存分配成本: {sum(L_nominal_cost) / len(L_nominal_cost):.4f}")
        print(f"全局最优缓存分配命中率: {cache_node_hit_rates}")
        
        # 准备输出文件
        print(f"保存分配历史: {save_allocations}")
        print(f"结果目录: {results_dir}")
        print(f"单个缓存节点容量: {single_cache_capacity}")
        print(f"请求率: {request_rate}")
        allocations_file = os.path.join(results_dir, f"allocations_single_cache_capacity{single_cache_capacity}_request_rate{request_rate}_global_opt.txt")
        print(f"分配文件路径: {allocations_file}")
        f = open(allocations_file, "w") if save_allocations else None
        print(f"文件对象: {f}")
        
        # 保存最佳分配
        if save_allocations and f:
            f.write(str(global_allocation) + '\n')
        
        if f:
            f.close()
        
        # 返回最佳分配和成本列表
        return global_allocation, L_total_cost, L_nominal_cost, L_first_cost, L_best_cost, L_avg_latency
    
    # 如果方法是equal_allocation，平均分配缓存空间
    elif method == 'equal_allocation':
        print("使用平均分配方法")

        # 获取缓存节点列表
        cache_nodes = topology_manager.get_cache_nodes()

        # 如果没有缓存节点，返回空分配
        if len(cache_nodes) == 0:
            print(f"平均分配: {{}}")
            avg_latency = user_to_cache_latency + cache_to_sp_latency
            L_total_cost = [1.0] * nb_interval
            L_nominal_cost = [1.0] * nb_interval
            L_first_cost = [1.0] * nb_interval
            L_best_cost = [1.0] * nb_interval
            L_avg_latency = [avg_latency] * nb_interval
            return {}, L_total_cost, L_nominal_cost, L_first_cost, L_best_cost, L_avg_latency

        # 为每个缓存节点平均分配缓存空间
        equal_allocation = {}
        node_cache_capacity = single_cache_capacity
        
        for i, node in enumerate(cache_nodes):
            # 每个节点的实际缓存容量都是固定的single_cache_capacity
            actual_capacity = node_cache_capacity
            # 平均分配给每个SP
            per_sp_capacity = actual_capacity // nSP
            extra = actual_capacity % nSP
            
            node_allocation = []
            for sp_idx in range(nSP):
                # 每个SP先分配平均数量
                node_allocation.append(per_sp_capacity)
                # 多余的缓存空间分配给第一个SP
                if sp_idx == 0:
                    node_allocation[0] += extra
            
            equal_allocation[node] = node_allocation
        
        print(f"平均分配: {equal_allocation}")
        
        # 计算平均分配的成本
        request_nb = int(interval_size * request_rate)
        
        # 为每个interval计算成本
        L_total_cost = []
        L_nominal_cost = []
        L_first_cost = []
        L_best_cost = []
        L_avg_latency = []
        
        for interval in range(nb_interval):
            # 使用当前interval的固定请求
            interval_fixed_requests = fixed_requests[interval] if fixed_requests and len(fixed_requests) > interval else None
            # 检查是否需要输出调试信息
            debug_output = (debug_interval is not None and interval == debug_interval)
            (nominal_cost, first_cost, best_cost, avg_latency, cache_node_hit_rates) = evaluate_cost(
                equal_allocation, allocation, best_allocation, request_nb, videos_proba, interval_fixed_requests, debug_output
            )
            
            L_total_cost.append(nominal_cost)
            L_nominal_cost.append(nominal_cost)
            L_first_cost.append(first_cost)
            L_best_cost.append(best_cost)
            L_avg_latency.append(avg_latency)
        
        print(f"平均分配成本: {sum(L_nominal_cost) / len(L_nominal_cost):.4f}")
        print(f"平均分配命中率: {cache_node_hit_rates}")
        
        # 返回成本列表
        return equal_allocation, L_total_cost, L_nominal_cost, L_first_cost, L_best_cost, L_avg_latency
    
    # 如果方法是SCA_ADMM，使用SCA-ADMM方法进行缓存分配
    elif method == 'SCA_ADMM':
        print("使用SCA-ADMM方法（视频级别）")

        # 获取缓存节点列表
        cache_nodes = topology_manager.get_cache_nodes()
        num_cache_nodes = len(cache_nodes)

        # 为每个缓存节点计算缓存容量（每个节点使用固定的single_cache_capacity）
        node_cache_capacity = single_cache_capacity

        # 初始化分配 - 对每个节点不同SP的均匀分配
        # 每个节点的缓存容量均匀分配给所有SP
        sca_admm_allocation = {}
        for i, node in enumerate(cache_nodes):
            # 每个节点的实际缓存容量都是固定的single_cache_capacity
            actual_capacity = node_cache_capacity
            
            # 均匀分配：每个SP获得相同的缓存容量
            base_allocation = actual_capacity // nSP
            remainder = actual_capacity % nSP
            
            node_allocation = []
            for sp in range(nSP):
                # 前remainder个SP多分配1个缓存单元
                if sp < remainder:
                    node_allocation.append(base_allocation + 1)
                else:
                    node_allocation.append(base_allocation)
            
            sca_admm_allocation[node] = node_allocation

        #print(f"初始分配: {sca_admm_allocation}")

        # SCA-ADMM参数（优化版本）
        max_sca_iterations_per_interval = 20  # 增加SCA迭代次数以确保充分收敛
        max_admm_iterations = 100  # 增加ADMM内循环迭代次数以确保充分收敛
        tau = SCA_ADMM_TAU if SCA_ADMM_TAU is not None else 0.05
        rho = SCA_ADMM_RHO if SCA_ADMM_RHO is not None else 0.2
        lambda_lasso = 0 #0.1  # 网络套索正则化参数（引入以促进邻居协同）
        
        # 时间平滑参数
        smoothing_alpha = 0.9  # 增强指数平滑系数
        max_change = 0.25  # 放宽变化限制
        convergence_threshold = 0.05  # 更严格的收敛阈值
        
        # 为每个缓存节点创建邻居列表
        cache_node_neighbors = {}
        for node in cache_nodes:
            all_neighbors = topology_manager.get_neighbors(node)
            neighbors = [n for n in all_neighbors if n in cache_nodes]
            cache_node_neighbors[node] = neighbors
        
        # 自适应参数调整机制
        def adaptive_parameter_adjustment(primal_residual, dual_residual, iteration, current_rho, current_tau):
            """
            基于残差自适应调整ADMM参数
            """
            new_rho = current_rho
            new_tau = current_tau
            
            # 残差比率监测
            if primal_residual > 0 and dual_residual > 0:
                residual_ratio = primal_residual / dual_residual
                
                # 自适应调整rho
                if residual_ratio > 10:  # 原始残差过大
                    new_rho = min(current_rho * 2.0, 1.0)  # 增大惩罚参数，上限1.0
                    #print(f"  自适应调整: 原始残差过大，rho从{current_rho:.3f}增加到{new_rho:.3f}")
                elif residual_ratio < 0.1:  # 对偶残差过大
                    new_rho = max(current_rho / 2.0, 0.05)  # 减小惩罚参数，下限0.05
                    #print(f"  自适应调整: 对偶残差过大，rho从{current_rho:.3f}减小到{new_rho:.3f}")
            
            # 基于迭代次数调整tau（逐渐减小邻近项）
            new_tau = current_tau * (0.95 ** min(iteration, 10))  # 最多衰减10次
            
            return new_rho, new_tau
        
        # 节点本地状态变量（在时间间隔之间保留）
        node_local_states = {}
        previous_sp_video_scores = {}
        previous_sp_proba = {}
        neighbor_history = {}
        
        # 初始化节点本地状态
        for node in cache_nodes:
            neighbors = cache_node_neighbors[node]
            node_local_states[node] = {
                'z': {},
                'v': {},
                'request_history': {
                    'sp_requests': {},  # 按SP分类的请求历史
                    'video_requests': {},  # 按视频分类的请求历史
                    'time_series': []  # 时间序列请求模式
                },
                'performance_history': []  # 性能历史
            }
            for neighbor in neighbors:
                node_local_states[node]['z'][neighbor] = sca_admm_allocation[node].copy()
                node_local_states[node]['v'][neighbor] = [0.0] * nSP
            previous_sp_video_scores[node] = [0.0] * nSP
            previous_sp_proba[node] = {}
        
        # 初始化邻居历史
        for node in cache_nodes:
            neighbor_history[node] = {}
            for neighbor in cache_node_neighbors[node]:
                neighbor_history[node][neighbor] = {
                    'allocation_history': [],  # 邻居分配历史
                    'hit_rate_history': [],  # 邻居命中率历史
                    'request_pattern': {}  # 邻居请求模式
                }
        
        # 准备输出文件
        allocations_file = os.path.join(results_dir, f"allocations_single_cache_capacity{single_cache_capacity}_request_rate{request_rate}_SCA_ADMM.txt")
        f = open(allocations_file, "w") if save_allocations else None
        
        # 初始化成本列表
        L_total_cost = []
        L_nominal_cost = []
        L_first_cost = []
        L_best_cost = []
        L_avg_latency = []
        
        # 计算一次初始分配和最佳分配的成本，作为固定值
        request_nb = int(interval_size * request_rate)
        # 使用第一个interval的固定请求
        interval_fixed_requests = fixed_requests[0] if fixed_requests and len(fixed_requests) > 0 else None
        (_, fixed_first_cost, fixed_best_cost, _, _) = evaluate_cost(
            best_allocation, sca_admm_allocation, best_allocation, request_nb, initial_videos_proba, interval_fixed_requests
        )
        
        # 在nb_interval次迭代中不断优化
        for interval in range(nb_interval):
            #print(f"\n=== 第 {interval+1}/{nb_interval} 个时间间隔 ===")
            
            # 在线学习：不预生成视频概率分布，而是基于历史请求数据
            # 清除缓存内容，强制重新生成可缓存内容列表
            current_videos_proba = catalog()  # 重新生成可缓存内容列表，使用返回值
            #print(f"  可缓存内容已更新，使用在线学习估计请求概率")
            
            # 计算当前分配的成本
            # 使用当前interval的固定请求
            interval_fixed_requests = fixed_requests[interval] if fixed_requests and len(fixed_requests) > interval else None
            # 检查是否需要输出调试信息
            debug_output = (debug_interval is not None and interval == debug_interval)
            (current_cost, _, _, current_latency, current_hit_rates) = evaluate_cost(
                sca_admm_allocation, sca_admm_allocation, best_allocation, request_nb, current_videos_proba, interval_fixed_requests, debug_output
            )
            #print(f"当前成本: {current_cost:.4f}")
            #print(f"当前命中率: {current_hit_rates}")
            
            # SCA外循环
            # 保存当前时间间隔的初始分配作为SCA的起始点
            current_interval_allocation = {}
            for node in sca_admm_allocation:
                current_interval_allocation[node] = sca_admm_allocation[node].copy()
            
            for sca_iter in range(max_sca_iterations_per_interval):
                #print(f"  SCA迭代 {sca_iter+1}/{max_sca_iterations_per_interval}")
                
                # 计算线性化系数（基于节点本地历史请求数据，不使用全局SP_proba）
                # 参考建模报告：对非凸的邻居乘积项进行线性化
                linear_coeffs = {}
                for node in cache_nodes:
                    coeffs = {}
                    neighbors = cache_node_neighbors[node]
                    x_current = current_interval_allocation[node]
                    
                    for sp in range(nSP):
                        # 节点本地的SP请求概率（基于历史数据估计）
                        node_sp_proba = estimate_sp_proba(node, sp)
                        
                        # 计算本地命中率 h_i(x_i[sp])
                        # 改进的命中率模型：h(x) = 1 - exp(-x / 50)，这样导数更大
                        h_i = 1.0 - math.exp(-x_current[sp] / 50.0)
                        dh_i = math.exp(-x_current[sp] / 50.0) / 50.0  # 导数
                        
                        # 计算邻居的乘积项 Π(1 - h_j)
                        product_term = 1.0
                        for neighbor in neighbors:
                            if neighbor in current_interval_allocation:
                                neighbor_alloc = current_interval_allocation[neighbor][sp]
                                h_j = 1.0 - math.exp(-neighbor_alloc / 50.0)
                                product_term *= (1 - h_j)
                        
                        # 增强信号强度的因子
                        signal_boost = 100.0
                        
                        # 根据建模报告：F_i(x_i) = -sum(lambda_{i,s} * h_i * product_term)
                        # 梯度 ∇F_i[x_i[sp]] = -lambda_{i,sp} * product_term * dh_i
                        # 其中 lambda_{i,sp} 是基于节点本地历史请求估计的权重
                        # 因此线性化系数 c_{i,sp} = -∇F_i = lambda_{i,sp} * product_term * dh_i
                        # 增加信号强度以加速收敛
                        coeffs[sp] = node_sp_proba * product_term * dh_i * signal_boost
                    linear_coeffs[node] = coeffs
                
                # ADMM内循环（使用节点本地状态）
                # 从节点本地状态构建z和v变量
                z = {}
                v = {}
                for node in cache_nodes:
                    z[node] = node_local_states[node]['z']
                    v[node] = node_local_states[node]['v']
                
                # 当前ADMM参数（初始值）
                current_rho = rho
                current_tau = tau
                
                for admm_iter in range(max_admm_iterations):
                    # Step 1: 更新x_i（节点本地）- 视频级别协同优化
                    # 参考cooperative_best_allocation的评分机制：
                    # score = video_proba * SP_proba * (1 if neighbor_has_video else 0.5)
                    new_allocation = {}

                    for node in cache_nodes:
                        neighbors = cache_node_neighbors[node]
                        has_neighbors = len(neighbors) > 0

                        # 计算邻居节点已缓存的视频信息
                        # neighbor_video_cache[sp] = 该SP在该节点缓存的前N个视频
                        neighbor_video_cache = {}
                        for neighbor in neighbors:
                            neighbor_video_cache[neighbor] = {}
                            if neighbor in sca_admm_allocation:
                                for sp in range(nSP):
                                    # 邻居分配给该SP的空间数量
                                    neighbor_sp_alloc = sca_admm_allocation[neighbor][sp]
                                    # 缓存的视频索引上限（按请求概率排序）
                                    neighbor_video_cache[neighbor][sp] = int(neighbor_sp_alloc)

                        # 构建凸优化问题
                        # 每个节点的实际缓存容量都是固定的single_cache_capacity
                        actual_capacity = node_cache_capacity

                        # 计算每个SP的视频级别得分（使用节点本地估计的SP概率）
                        # 参考建模报告：F_i(x_i) = -sum(lambda_{i,s} * h_i * product_term)
                        # 其中 lambda_{i,s} 使用节点本地的SP请求概率估计
                        sp_video_scores = []
                        for sp in range(nSP):
                            if cacheable_content and sp < len(cacheable_content):
                                cacheable_videos = cacheable_content[sp]
                                sp_score = 0.0
                                # 使用节点本地的SP请求概率（基于历史数据估计）
                                node_sp_proba = estimate_sp_proba(node, sp)
                                for video_idx, video in enumerate(cacheable_videos):
                                    if video_idx >= actual_capacity:
                                        break
                                    # 视频的请求概率 - 使用在线学习估计的概率
                                    video_proba = estimate_video_proba(node, sp, video)
                                    
                                    # 检查邻居是否已缓存该视频
                                    neighbor_has_video = False
                                    for neighbor in neighbors:
                                        if neighbor in neighbor_video_cache:
                                            if neighbor_video_cache[neighbor].get(int(sp), 0) > video_idx:
                                                neighbor_has_video = True
                                                break
                                    
                                    # 利用邻居历史信息增强缓存决策
                                    neighbor_benefit = 1.0
                                    try:
                                        if 'neighbor_history' in globals() and node in neighbor_history:
                                            for neighbor in neighbors:
                                                if neighbor in neighbor_history[node]:
                                                    # 检查邻居的热门视频
                                                    if 'request_pattern' in neighbor_history[node][neighbor]:
                                                        neighbor_pattern = neighbor_history[node][neighbor]['request_pattern']
                                                        if sp in neighbor_pattern and video in neighbor_pattern[sp]:
                                                            # 邻居也有类似的热门视频，降低缓存优先级
                                                            neighbor_benefit *= 0.7
                                    except NameError:
                                        pass
                                    
                                    # 邻居未缓存时得分为1，缓存时得分为0.5
                                    cache_benefit = 1.0 if not neighbor_has_video else 0.5
                                    cache_benefit *= neighbor_benefit
                                    
                                    # 长期收益估计
                                    long_term_benefit = 1.0
                                    if node in node_local_states:
                                        # 基于历史请求模式预测长期收益
                                        history = node_local_states[node]['request_history']
                                        if 'video_requests' in history and sp in history['video_requests']:
                                            video_history = history['video_requests'][sp]
                                            if video in video_history:
                                                # 视频有历史请求，增加长期收益
                                                video_count = video_history[video]
                                                total_count = sum(video_history.values())
                                                if total_count > 0:
                                                    long_term_benefit = 1.0 + (video_count / total_count) * 0.5
                                    
                                    # 使用节点本地的SP概率而非全局SP_proba
                                    sp_score += video_proba * node_sp_proba * cache_benefit * long_term_benefit
                                sp_video_scores.append(sp_score)
                            else:
                                sp_video_scores.append(0.0)

                        # 使用视频级别得分进行ADMM x_i更新
                        # 参考建模报告Step 1的解析解：
                        # x_i^{k+1} = argmin { F_i(x_i) + (rho|N(i)|/2)||x_i - bar_z||^2 + (tau/2)||x_i - x_i^t||^2 }
                        # 其中 F_i(x_i) = -sum(sp_video_scores[sp] * x_i[sp])（视频级别线性近似）
                        # 解析解：x_i[sp] = (sp_video_scores[sp] + rho|N(i)|*bar_z[sp] + tau*x_i^t[sp]) / (rho|N(i)| + tau)
                        if has_neighbors:
                            sum_z_minus_v = [0.0] * nSP
                            for neighbor in neighbors:
                                for sp in range(nSP):
                                    sum_z_minus_v[sp] += z[node][neighbor][sp] - v[node][neighbor][sp] / rho
                            avg_z_minus_v = [s / len(neighbors) for s in sum_z_minus_v]
                        else:
                            # 无邻居时，使用基于梯度的局部优化
                            # 梯度：∇F_i[x_i[sp]] = -sp_video_scores[sp]（因为F_i是线性函数）
                            # 使用梯度上升然后投影到可行域
                            avg_z_minus_v = [0.0] * nSP
                        
                        num_neighbors = len(neighbors) if neighbors else 1
                        x_i = sca_admm_allocation[node].copy()
                        
                        if has_neighbors:
                            # 正常的ADMM更新（有邻居）
                            for sp in range(nSP):
                                # 根据建模报告的解析解计算x_i[sp]
                                # F_i(x_i) = -sum(sp_video_scores[sp] * x_i[sp])
                                # 目标函数导数：-sp_video_scores[sp] + rho*|N(i)|*(x_i[sp] - avg_z_minus_v[sp]) + tau*(x_i[sp] - x_i^{t-1}[sp])
                                # 令导数为0，解得：
                                # x_i^{t} = (sp_video_scores[sp] + rho*|N(i)|*avg_z_minus_v[sp] + tau*x_i^{t-1}[sp]) / (rho*|N(i)| + tau)
                                # 其中 x_i^{t-1} 是上一次ADMM迭代的值（x_i变量的当前值）
                                numerator = sp_video_scores[sp] + current_rho * num_neighbors * avg_z_minus_v[sp] + current_tau * x_i[sp]
                                denominator = current_rho * num_neighbors + current_tau
                                new_value = numerator / denominator
                                
                                # 限制变化幅度
                                old_value = x_i[sp]
                                if new_value > old_value:
                                    new_value = min(old_value * (1 + max_change), new_value)
                                else:
                                    new_value = max(old_value * (1 - max_change), new_value)
                                
                                x_i[sp] = new_value
                        else:
                            # 无邻居时的基于梯度的局部优化
                            # 目标：最大化 sum(sp_video_scores[sp] * x_i[sp])
                            # 约束：sum(x_i) = actual_capacity, x_i >= 0
                            # 使用梯度上升结合投影梯度下降
                            
                            # 计算梯度
                            gradient = sp_video_scores.copy()
                            
                            # 梯度上升步长
                            gradient_step = 10.0
                            
                            # 梯度上升
                            for sp in range(nSP):
                                x_i[sp] += gradient_step * gradient[sp]
                            
                            # 投影到非负 orthant
                            for sp in range(nSP):
                                x_i[sp] = max(0, x_i[sp])
                            
                            # 投影到容量约束（使用欧几里得投影）
                            total = sum(x_i)
                            if total > 0:
                                scale = actual_capacity / total
                                for sp in range(nSP):
                                    x_i[sp] *= scale
                            
                            # 应用平滑限制，防止剧烈变化
                            old_x_i = sca_admm_allocation[node].copy()
                            for sp in range(nSP):
                                if x_i[sp] > old_x_i[sp]:
                                    x_i[sp] = min(old_x_i[sp] * (1 + max_change), x_i[sp])
                                else:
                                    x_i[sp] = max(old_x_i[sp] * (1 - max_change), x_i[sp])
                            
                            # 再次投影到容量约束
                            total = sum(x_i)
                            if total > 0:
                                scale = actual_capacity / total
                                for sp in range(nSP):
                                    x_i[sp] *= scale

                        # 投影到非负 orthant
                        for sp in range(nSP):
                            x_i[sp] = max(0, x_i[sp])

                        # 投影到容量约束
                        total = sum(x_i)
                        if total > 0:
                            scale = actual_capacity / total
                            for sp in range(nSP):
                                x_i[sp] *= scale

                        new_allocation[node] = x_i
                    
                    # Step 2: 更新z_ij（边本地）
                    for node in cache_nodes:
                        neighbors = cache_node_neighbors[node]
                        for neighbor in neighbors:
                            a = new_allocation[node].copy()
                            for sp in range(nSP):
                                a[sp] += v[node][neighbor][sp] / rho
                            b = new_allocation[neighbor].copy()
                            for sp in range(nSP):
                                b[sp] += v[neighbor][node][sp] / rho
                            
                            # 计算d = a - b
                            d = [a[sp] - b[sp] for sp in range(nSP)]
                            norm_d = np.linalg.norm(d)
                            
                            # L2范数近端算子
                            if norm_d > 0:
                                threshold = (2 * lambda_lasso) / rho
                                if norm_d > threshold:
                                    scale = 1 - threshold / norm_d
                                    d = [scale * x for x in d]
                                else:
                                    d = [0.0] * nSP
                            
                            # 更新z_ij和z_ji
                            m = [(a[sp] + b[sp]) / 2 for sp in range(nSP)]
                            z[node][neighbor] = [m[sp] + d[sp] / 2 for sp in range(nSP)]
                            z[neighbor][node] = [m[sp] - d[sp] / 2 for sp in range(nSP)]
                    
                    # Step 3: 更新拉格朗日乘子
                    for node in cache_nodes:
                        neighbors = cache_node_neighbors[node]
                        for neighbor in neighbors:
                            for sp in range(nSP):
                                v[node][neighbor][sp] += current_rho * (new_allocation[node][sp] - z[node][neighbor][sp])
                                v[neighbor][node][sp] += current_rho * (new_allocation[neighbor][sp] - z[neighbor][node][sp])
                    
                    # 检查收敛
                    primal_residual = 0.0
                    dual_residual = 0.0
                    
                    # 计算原始残差
                    for node in cache_nodes:
                        neighbors = cache_node_neighbors[node]
                        for neighbor in neighbors:
                            for sp in range(nSP):
                                primal_residual += (new_allocation[node][sp] - z[node][neighbor][sp])**2
                    primal_residual = np.sqrt(primal_residual)
                    
                    # 计算对偶残差
                    for node in cache_nodes:
                        neighbors = cache_node_neighbors[node]
                        for neighbor in neighbors:
                            for sp in range(nSP):
                                # 对偶残差 = rho * (z^{k+1} - z^k)
                                # 这里简化计算，使用v的变化作为对偶残差的代理
                                dual_residual += (v[node][neighbor][sp] - node_local_states[node]['v'].get(neighbor, [0]*nSP)[sp])**2
                    dual_residual = np.sqrt(dual_residual) * current_rho
                    
                    # 自适应参数调整（每5次迭代调整一次）
                    if admm_iter > 0 and admm_iter % 5 == 0:
                        current_rho, current_tau = adaptive_parameter_adjustment(
                            primal_residual, dual_residual, admm_iter, current_rho, current_tau
                        )
                    
                    # 检查分配变化收敛（时间间隔内部的ADMM迭代）
                    allocation_diff = 0.0
                    for node in cache_nodes:
                        for sp in range(nSP):
                            allocation_diff += abs(new_allocation[node][sp] - sca_admm_allocation[node][sp])
                    
                    if primal_residual < 1e-3 and dual_residual < 1e-3:
                        #print(f"  收敛于ADMM迭代 {admm_iter+1}, 原始残差: {primal_residual:.6f}, 对偶残差: {dual_residual:.6f}")
                        break
                
                # 更新SCA迭代的分配
                sca_admm_allocation = new_allocation
                
                # 更新节点本地状态
                for node in cache_nodes:
                    node_local_states[node]['z'] = z[node]
                    node_local_states[node]['v'] = v[node]
            
            # 四舍五入到整数前打印调试信息
            #print(f"  四舍五入前的分配(第一个节点): {list(sca_admm_allocation.values())[0]}")
            
            # 四舍五入到整数
            for node in cache_nodes:
                # 每个节点的实际缓存容量都是固定的single_cache_capacity
                actual_capacity = node_cache_capacity
                node_allocation = sca_admm_allocation[node]
                # 四舍五入到最近的整数
                rounded_allocation = [round(x) for x in node_allocation]
                # 调整总和为actual_capacity
                total = sum(rounded_allocation)
                if total != actual_capacity:
                    diff = actual_capacity - total
                    # 调整最大的分配
                    max_idx = rounded_allocation.index(max(rounded_allocation))
                    rounded_allocation[max_idx] += diff
                # 确保所有值非负
                rounded_allocation = [max(0, x) for x in rounded_allocation]
                sca_admm_allocation[node] = rounded_allocation
            
            #print(f"  调整后的分配: {sca_admm_allocation}")
            
            # 保存分配
            if save_allocations and f:
                f.write(str(sca_admm_allocation) + '\n')
            
            # 计算当前分配的成本
            # 使用当前interval的固定请求
            interval_fixed_requests = fixed_requests[interval] if fixed_requests and len(fixed_requests) > interval else None
            (nominal_cost, _, _, avg_latency, cache_node_hit_rates) = evaluate_cost(
                sca_admm_allocation, sca_admm_allocation, best_allocation, request_nb, videos_proba, interval_fixed_requests
            )
            #print(f"  成本: {nominal_cost:.4f}")
            #print(f"  命中率: {cache_node_hit_rates}")
            
            # 使用固定的初始成本和最佳成本
            first_cost = fixed_first_cost
            best_cost = fixed_best_cost
            
            # 记录成本
            total_cost = nominal_cost
            L_total_cost.append(total_cost)
            L_nominal_cost.append(nominal_cost)
            L_first_cost.append(first_cost)
            L_best_cost.append(best_cost)
            L_avg_latency.append(avg_latency)
            
            # 更新节点性能历史
            for node in cache_nodes:
                if node in node_local_states:
                    # 计算节点的命中率
                    node_hit_rate = current_hit_rates.get(node, 0.0)
                    # 更新性能历史
                    node_local_states[node]['performance_history'].append({
                        'hit_rate': node_hit_rate,
                        'cost': total_cost,
                        'allocation': sca_admm_allocation[node].copy(),
                        'interval': interval
                    })
            
            # 智能信息交换
            def exchange_intelligent_info(node, neighbors):
                """智能信息交换，共享关键历史信息"""
                # 收集当前节点的信息
                current_allocation = sca_admm_allocation[node]
                
                # 计算最近的命中率
                recent_hit_rate = 0.0
                if node in node_local_states and node_local_states[node]['performance_history']:
                    recent_performance = node_local_states[node]['performance_history'][-5:]  # 最近5个时间间隔
                    recent_hit_rate = sum(p['hit_rate'] for p in recent_performance) / len(recent_performance)
                
                # 获取热门视频
                top_videos = {}
                if node in node_request_history:
                    for sp in range(nSP):
                        if sp in node_request_history[node]:
                            video_counts = node_request_history[node][sp]
                            sorted_videos = sorted(video_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                            top_videos[sp] = [v[0] for v in sorted_videos]
                
                # 交换数据
                for neighbor in neighbors:
                    # 发送数据给邻居
                    exchange_data = {
                        'current_allocation': current_allocation,
                        'recent_hit_rate': recent_hit_rate,
                        'top_videos': top_videos
                    }
                    
                    # 更新邻居历史
                    if node in neighbor_history and neighbor in neighbor_history[node]:
                        neighbor_history[node][neighbor]['allocation_history'].append(current_allocation)
                        neighbor_history[node][neighbor]['hit_rate_history'].append(recent_hit_rate)
                        neighbor_history[node][neighbor]['request_pattern'] = top_videos
            
            # 执行信息交换
            for node in cache_nodes:
                neighbors = cache_node_neighbors[node]
                exchange_intelligent_info(node, neighbors)
        
        # 时间间隔之间的收敛检测
        if interval > 0:
            # 计算时间间隔之间的分配变化
            interval_allocation_diff = 0.0
            for node in cache_nodes:
                current_allocation = sca_admm_allocation[node]
                if 'previous_allocation' in node_local_states[node]:
                    previous_allocation = node_local_states[node]['previous_allocation']
                    for sp in range(nSP):
                        interval_allocation_diff += abs(current_allocation[sp] - previous_allocation[sp])
            
            # 检查时间间隔之间的收敛
            #if interval_allocation_diff < convergence_threshold:
                #print(f"\n=== 时间间隔间收敛 ===")
                #print(f"  时间间隔 {interval+1}, 变化: {interval_allocation_diff:.6f} < {convergence_threshold:.6f}")
                #print(f"  已收敛，继续运行剩余时间间隔")
            
            # 对每个节点的分配进行平滑
            for node in cache_nodes:
                current_allocation = sca_admm_allocation[node].copy()
                # 应用平滑
                if 'previous_allocation' in node_local_states[node]:
                    previous_allocation = node_local_states[node]['previous_allocation']
                    for sp in range(nSP):
                        sca_admm_allocation[node][sp] = smoothing_alpha * current_allocation[sp] + (1 - smoothing_alpha) * previous_allocation[sp]
                # 保存当前分配用于下一次收敛检测
                node_local_states[node]['previous_allocation'] = current_allocation.copy()
        
        if f:
            f.close()

        if debug_mode:
            print(f"\n最终SCA-ADMM分配: {sca_admm_allocation}")

        return sca_admm_allocation, L_total_cost, L_nominal_cost, L_first_cost, L_best_cost, L_avg_latency

    # 如果方法是SCA_neighborhood_search，使用SCA外循环+离散邻域搜索替代ADMM
    elif method == 'SCA_neighborhood_search':
        print("使用SCA_neighborhood_search方法（离散邻域搜索优化）")
        print("  特点：SCA外循环 + 离散邻域搜索替代ADMM")
        print("  在线学习：只依赖自身和邻居历史信息")
        print("  视频级别信息交互：模仿SCA_ADMM的视频级别协同")
        print("  优化方法：离散邻域搜索，适合整数规划问题")

        # 辅助函数：对单个节点的分配进行取整
        def round_allocation(node_allocation, actual_capacity):
            """
            将浮点数分配转换为整数分配，确保总和等于容量
            """
            integer_allocation = []
            # 第一步：四舍五入为整数
            for sp_alloc in node_allocation:
                if (sp_alloc is not None and 
                    not math.isnan(sp_alloc) and 
                    not math.isinf(sp_alloc) and 
                    sp_alloc >= 0 and 
                    sp_alloc <= 1e6):
                    integer_allocation.append(int(round(sp_alloc)))
                else:
                    integer_allocation.append(0)
            
            # 第二步：调整确保总和等于容量
            total_allocated = sum(integer_allocation)
            if total_allocated != actual_capacity:
                diff = actual_capacity - total_allocated
                if diff > 0:
                    # 需要增加分配
                    for i in range(diff):
                        min_idx = integer_allocation.index(min(integer_allocation))
                        integer_allocation[min_idx] += 1
                else:
                    # 需要减少分配
                    diff = abs(diff)
                    for i in range(diff):
                        max_idx = integer_allocation.index(max(integer_allocation))
                        if integer_allocation[max_idx] > 0:
                            integer_allocation[max_idx] -= 1
            
            return integer_allocation

        # 获取缓存节点列表
        cache_nodes = topology_manager.get_cache_nodes()
        num_cache_nodes = len(cache_nodes)
        
        # 初始化缓存分配（与SCA_ADMM相同的均匀分配策略）
        sca_neighborhood_allocation = {}
        for i, node in enumerate(cache_nodes):
            actual_capacity = single_cache_capacity
            
            # 均匀分配：每个SP获得相同的缓存容量（与SCA_ADMM保持一致）
            base_allocation = actual_capacity // nSP
            remainder = actual_capacity % nSP
            
            node_allocation = []
            for sp in range(nSP):
                # 前remainder个SP多分配1个缓存单元
                if sp < remainder:
                    node_allocation.append(base_allocation + 1)
                else:
                    node_allocation.append(base_allocation)
            
            sca_neighborhood_allocation[node] = node_allocation

        print(f"初始分配: {sca_neighborhood_allocation}")

        # 准备输出文件
        allocations_file = os.path.join(results_dir, f"allocations_single_cache_capacity{single_cache_capacity}_request_rate{request_rate}_SCA_neighborhood_search.txt")
        f = open(allocations_file, "w") if save_allocations else None

        # 计算一次初始分配和最佳分配的成本，作为固定值（与SCA_ADMM保持一致）
        request_nb = int(interval_size * request_rate)
        interval_fixed_requests = fixed_requests[0] if fixed_requests and len(fixed_requests) > 0 else None
        (_, fixed_first_cost, fixed_best_cost, _, _) = evaluate_cost(
            best_allocation, sca_neighborhood_allocation, best_allocation, request_nb, initial_videos_proba, interval_fixed_requests
        )

        # 离散邻域搜索参数（进一步优化版本）
        max_sca_iterations_per_interval = 20  # 与SCA_ADMM相同的SCA迭代次数
        max_neighborhood_search_iterations = 50  # 减少邻域搜索迭代次数（原为50）
        neighborhood_size = 3  # 邻域大小（保持原有设置）
        temperature = 0.3  # 模拟退火温度（用于接受劣解）
        cooling_rate = 0.95  # 温度衰减率
        
        # 提前终止参数
        no_improvement_threshold = 5  # 连续无改进次数阈值
        
        # 时间平滑参数（与SCA_ADMM保持一致）
        smoothing_alpha = 0.9  # 增强指数平滑系数
        
        # 彻底优化：避免视频级计算，使用SP级别成本估计
        # 不再预计算视频概率矩阵，改用SP级别成本函数
        
        # 为每个缓存节点创建邻居列表（与SCA_ADMM保持一致）
        cache_node_neighbors = {}
        for node in cache_nodes:
            all_neighbors = topology_manager.get_neighbors(node)
            neighbors = [n for n in all_neighbors if n in cache_nodes]
            cache_node_neighbors[node] = neighbors

        # 初始化节点状态（用于在线学习）
        node_local_states = {}
        for node in cache_nodes:
            neighbors = cache_node_neighbors[node]
            node_local_states[node] = {
                'request_history': {
                    'video_requests': {int(sp): {} for sp in range(nSP)},
                    'sp_requests': {int(sp): 0 for sp in range(nSP)}
                },
                'gradient_tracking': {
                    'local_gradient': [0.0] * nSP,
                    'tracked_gradient': [0.0] * nSP,
                    'neighbor_gradients': {neighbor: [0.0] * nSP for neighbor in neighbors}
                },
                'neighborhood_search': {
                    'current_cost': 0.0,
                    'best_allocation': sca_neighborhood_allocation[node].copy()
                },
                'previous_allocation': sca_neighborhood_allocation[node].copy()
            }
            
        #print(f"node_local_states: {node_local_states}")

        # 初始化成本记录列表（与SCA_ADMM保持一致）
        request_nb = int(interval_size * request_rate)  # 每个interval的请求数量
        L_total_cost = []
        L_nominal_cost = []
        L_first_cost = []
        L_best_cost = []
        L_avg_latency = []
        L_sca_iterations = []  # 记录每个interval的SCA迭代次数
        L_gradient_tracking_iterations = []  # 记录每个interval的梯度跟踪迭代次数

        # 使用对比模式中已生成的固定请求序列，确保公平比较
        # 如果外部没有传入fixed_requests，则使用默认的请求生成逻辑（包含源节点）
        if fixed_requests is None or len(fixed_requests) == 0:
            fixed_requests = []
            
            # 获取可能的源节点（与对比模式保持一致）
            sp_nodes = topology_manager.get_sp_nodes()
            receiver_nodes = topology_manager.get_receiver_nodes()
            router_nodes = topology_manager.get_router_nodes()
            cache_nodes_list = topology_manager.get_cache_nodes()
            
            for interval in range(nb_interval):
                interval_requests = {
                    'requests': [],
                    'source_nodes': []
                }
                
                for _ in range(request_nb):
                    # 随机选择SP
                    sp_idx = rd.choices(range(nSP), weights=SP_proba)[0]
                    
                    # 随机选择视频
                    if cacheable_content and sp_idx < len(cacheable_content):
                        cacheable_videos = cacheable_content[sp_idx]
                        video_idx = rd.randint(0, len(cacheable_videos) - 1)
                        video_id = cacheable_videos[video_idx]
                    else:
                        video_id = rd.randint(0, nb_videos - 1)
                    
                    # 随机选择源节点（与对比模式保持一致）
                    possible_sources = []
                    if router_nodes:
                        possible_sources.extend(router_nodes)
                    if cache_nodes_list:
                        possible_sources.extend(cache_nodes_list)
                    
                    if possible_sources:
                        source_node = rd.choice(possible_sources)
                    else:
                        source_node = None
                    
                    interval_requests['requests'].append((sp_idx, video_id))
                    interval_requests['source_nodes'].append(source_node)
                
                fixed_requests.append(interval_requests)
        
        #print(f"fixed_requests: {fixed_requests}")

        # SCA外循环（与SCA_ADMM相同的结构）
        for interval in range(nb_interval):
            current_videos_proba = catalog()
            interval_fixed_requests = fixed_requests[interval] if fixed_requests and len(fixed_requests) > interval else None
            # 检查是否需要输出调试信息
            debug_output = (debug_interval is not None and interval == debug_interval)
            (current_cost, _, _, current_latency, current_hit_rates) = evaluate_cost(
                sca_neighborhood_allocation, sca_neighborhood_allocation, best_allocation, request_nb, current_videos_proba, interval_fixed_requests, debug_output
            )

            # 保存当前时间间隔的初始分配
            current_interval_allocation = {}
            for node in sca_neighborhood_allocation:
                current_interval_allocation[node] = sca_neighborhood_allocation[node].copy()
            
            # SCA外循环（固定20次迭代，与SCA_ADMM一致）
            L_sca_iterations.append(max_sca_iterations_per_interval)
            total_neighborhood_search_iterations = 0
            
            for sca_iter in range(max_sca_iterations_per_interval):
                # 离散邻域搜索内循环（替代ADMM）
                neighborhood_search_converged = False
                previous_allocation = sca_neighborhood_allocation.copy()
                
                # 提前终止机制初始化
                no_improvement_count = 0
                best_cost_so_far = float('inf')
                
                for search_iter in range(max_neighborhood_search_iterations):
                    total_neighborhood_search_iterations += 1
                    new_allocation = {}
                    
                    # 残差监控：检查是否收敛
                    if search_iter > 0:
                        max_residual = 0.0
                        for node in cache_nodes:
                            if node in previous_allocation and node in sca_neighborhood_allocation:
                                for sp in range(nSP):
                                    residual = abs(sca_neighborhood_allocation[node][sp] - previous_allocation[node][sp])
                                    max_residual = max(max_residual, residual)
                        
                        # 如果残差小于阈值，提前终止
                        if max_residual < 1e-3:
                            neighborhood_search_converged = True
                            break
                    
                    # 提前终止：检查成本改进情况
                    current_total_cost = 0.0
                    for node in cache_nodes:
                        current_allocation = sca_neighborhood_allocation[node]
                        # 简化成本计算（仅用于提前终止判断）
                        node_cost = sum(current_allocation)  # 简化的成本估计
                        current_total_cost += node_cost
                    
                    if current_total_cost < best_cost_so_far - 1e-3:
                        best_cost_so_far = current_total_cost
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                        if no_improvement_count >= no_improvement_threshold:
                            neighborhood_search_converged = True
                            break
                    
                    # 保存当前分配用于下一次残差计算
                    previous_allocation = sca_neighborhood_allocation.copy()
                    
                    # 彻底优化：离散邻域搜索核心逻辑（SP级别版本）
                    # 完全避免视频级计算，采用SP级别成本估计
                    
                    for node in cache_nodes:
                        neighbors = cache_node_neighbors[node]
                        has_neighbors = len(neighbors) > 0

                        actual_capacity = single_cache_capacity
                        current_allocation = sca_neighborhood_allocation[node].copy()
                        
                        # 计算当前分配的成本（SP级别版本，完全避免视频级循环）
                        current_cost = 0.0
                        
                        for sp in range(nSP):
                            if cacheable_content and sp < len(cacheable_content):
                                # 使用节点本地估计的SP概率
                                node_sp_proba = estimate_sp_proba(node, sp)
                                
                                # SP级别成本计算（类似SCA_ADMM的线性化方法）
                                # 成本 = 分配量 × SP概率 × 成本系数
                                sp_alloc = current_allocation[sp]
                                
                                # 成本系数：基于历史数据的经验值
                                # 考虑邻居协同效应：有邻居时成本降低
                                neighbor_coefficient = 0.7 if has_neighbors else 0.9
                                
                                # 长期成本估计：基于历史请求频率
                                long_term_factor = 1.0
                                history = node_local_states[node]['request_history']
                                if 'sp_requests' in history and int(sp) in history['sp_requests']:
                                    sp_count = history['sp_requests'][int(sp)]
                                    total_count = sum(history['sp_requests'].values())
                                    if total_count > 0:
                                        long_term_factor = 1.0 - (sp_count / total_count) * 0.3
                                
                                # SP级别成本计算
                                sp_cost = sp_alloc * node_sp_proba * neighbor_coefficient * long_term_factor
                                current_cost += sp_cost

                        # 离散邻域搜索更新
                        # 生成邻域解
                        best_neighbor_allocation = current_allocation.copy()
                        best_neighbor_cost = current_cost
                        
                        # 探索邻域（基于梯度信息的智能生成）
                        # 计算当前分配的梯度信息
                        gradient = []
                        for sp in range(nSP):
                            if cacheable_content and sp < len(cacheable_content):
                                node_sp_proba = estimate_sp_proba(node, sp)
                                # 梯度方向：高概率SP应该增加分配
                                gradient.append(node_sp_proba)
                            else:
                                gradient.append(0.0)
                        
                        # 归一化梯度
                        gradient_sum = sum(gradient)
                        if gradient_sum > 0:
                            gradient = [g / gradient_sum for g in gradient]
                        
                        # 基于梯度信息生成邻域
                        for _ in range(neighborhood_size):
                            neighbor_allocation = current_allocation.copy()
                            
                            # 基于梯度选择SP（高梯度SP更可能被选择）
                            sp_to_adjust = rd.choices(range(nSP), weights=gradient, k=1)[0]
                            
                            # 基于梯度选择调整方向（高梯度SP倾向于增加分配）
                            if gradient[sp_to_adjust] > 0.5:
                                adjustment = 1  # 高概率SP增加分配
                            else:
                                adjustment = rd.choice([-1, 1])  # 低概率SP随机调整
                            
                            # 确保调整后分配仍然有效
                            if 0 <= neighbor_allocation[sp_to_adjust] + adjustment <= actual_capacity:
                                neighbor_allocation[sp_to_adjust] += adjustment
                                
                                # 确保总和不超过容量
                                total_alloc = sum(neighbor_allocation)
                                if total_alloc > actual_capacity:
                                    # 随机选择一个SP减少分配
                                    other_sp = rd.randint(0, nSP - 1)
                                    while other_sp == sp_to_adjust or neighbor_allocation[other_sp] <= 0:
                                        other_sp = rd.randint(0, nSP - 1)
                                    neighbor_allocation[other_sp] -= (total_alloc - actual_capacity)
                                elif total_alloc < actual_capacity:
                                    # 随机选择一个SP增加分配
                                    other_sp = rd.randint(0, nSP - 1)
                                    neighbor_allocation[other_sp] += (actual_capacity - total_alloc)
                                
                                # 彻底优化：使用SP级别成本函数，完全避免视频级计算
                                # 基于SCA_ADMM的线性化思想，使用SP级别的成本估计
                                
                                # 计算调整的SP的成本变化（SP级别近似）
                                sp_to_adjust = sp_to_adjust  # 从外部循环获取
                                sp_cost_change = 0.0  # 初始化成本变化变量
                                
                                if cacheable_content and sp_to_adjust < len(cacheable_content):
                                    node_sp_proba = estimate_sp_proba(node, sp_to_adjust)
                                    
                                    # 计算调整后的SP成本变化
                                    old_alloc = current_allocation[sp_to_adjust]
                                    new_alloc = neighbor_allocation[sp_to_adjust]
                                    
                                    # 使用SP级别的线性化成本模型（类似SCA_ADMM）
                                    # 成本变化 = 分配变化量 × SP概率 × 成本系数
                                    allocation_change = new_alloc - old_alloc
                                    
                                    # 成本系数：基于历史数据的经验值
                                    # 增加缓存：成本降低（负值），减少缓存：成本增加（正值）
                                    cost_coefficient = -0.8  # 经验值，类似SCA_ADMM的线性化系数
                                    
                                    sp_cost_change = allocation_change * node_sp_proba * cost_coefficient
                                    
                                # 基于当前成本加上调整的成本变化
                                neighbor_cost = current_cost + sp_cost_change
                                
                                # 接受准则：模拟退火
                                if neighbor_cost < best_neighbor_cost:
                                    best_neighbor_allocation = neighbor_allocation.copy()
                                    best_neighbor_cost = neighbor_cost
                                elif rd.random() < math.exp(-(neighbor_cost - best_neighbor_cost) / temperature):
                                    best_neighbor_allocation = neighbor_allocation.copy()
                                    best_neighbor_cost = neighbor_cost
                        
                        new_allocation[node] = best_neighbor_allocation
                        
                        # 更新本地状态
                        node_local_states[node]['neighborhood_search']['current_cost'] = best_neighbor_cost
                        node_local_states[node]['neighborhood_search']['best_allocation'] = best_neighbor_allocation.copy()

                    # 应用新分配
                    for node in cache_nodes:
                        sca_neighborhood_allocation[node] = new_allocation[node]
                    
                    # 温度衰减
                    temperature *= cooling_rate

                # 记录邻域搜索迭代次数
                #if neighborhood_search_converged:
                #    print(f"  SCA迭代 {sca_iter+1}: 邻域搜索提前终止于{search_iter}次迭代")
                
                # 应用时间平滑（与SCA_ADMM保持一致）
                for node in cache_nodes:
                    current_allocation = sca_neighborhood_allocation[node].copy()
                    if 'previous_allocation' in node_local_states[node]:
                        previous_allocation = node_local_states[node]['previous_allocation']
                        for sp in range(nSP):
                            sca_neighborhood_allocation[node][sp] = smoothing_alpha * current_allocation[sp] + (1 - smoothing_alpha) * previous_allocation[sp]
                    node_local_states[node]['previous_allocation'] = current_allocation.copy()

            # 记录当前interval的邻域搜索迭代次数
            L_gradient_tracking_iterations.append(total_neighborhood_search_iterations)
            
            # 更新历史请求信息（在线学习）
            interval_fixed_requests = fixed_requests[interval] if fixed_requests and len(fixed_requests) > interval else None
            (nominal_cost, _, _, avg_latency, cache_node_hit_rates) = evaluate_cost(
                sca_neighborhood_allocation, sca_neighborhood_allocation, best_allocation, request_nb, current_videos_proba, interval_fixed_requests
            )

            first_cost = fixed_first_cost
            best_cost = fixed_best_cost
            total_cost = nominal_cost
            
            L_total_cost.append(total_cost)
            L_nominal_cost.append(nominal_cost)
            L_first_cost.append(first_cost)
            L_best_cost.append(best_cost)
            L_avg_latency.append(avg_latency)

            # 对当前interval的分配进行取整后再保存
            integer_allocation_for_save = {}
            for node in sca_neighborhood_allocation:
                integer_allocation_for_save[node] = round_allocation(
                    sca_neighborhood_allocation[node], 
                    single_cache_capacity
                )
            
            # 保存分配（保存的是整数分配）
            if save_allocations and f:
                f.write(str(integer_allocation_for_save) + '\n')

        # 关闭文件
        if save_allocations and f:
            f.close()

        # 确保最终分配是整数且符合节点缓存容量限制
        for node in cache_nodes:
            if node in sca_neighborhood_allocation:
                sca_neighborhood_allocation[node] = round_allocation(
                    sca_neighborhood_allocation[node], 
                    single_cache_capacity
                )
        
        # 输出统计信息（与SCA_ADMM保持一致）
        print(f"\n最终SCA_neighborhood_search分配（整数化后）: {sca_neighborhood_allocation}")
        print(f"每周期SCA迭代次数: {L_sca_iterations}")
        if L_sca_iterations:
            avg_sca_iterations = sum(L_sca_iterations) / len(L_sca_iterations)
            print(f"平均每周期SCA迭代次数: {avg_sca_iterations:.2f}")
            
            print(f"每周期邻域搜索迭代次数: {L_gradient_tracking_iterations}")
            if L_gradient_tracking_iterations:
                avg_neighborhood_iterations = sum(L_gradient_tracking_iterations) / len(L_gradient_tracking_iterations)
                print(f"平均每周期邻域搜索迭代次数: {avg_neighborhood_iterations:.2f}")
                
                total_neighborhood_iterations = sum(L_gradient_tracking_iterations)
                total_sca_iterations = sum(L_sca_iterations)
                if total_sca_iterations > 0:
                    avg_neighborhood_per_sca = total_neighborhood_iterations / total_sca_iterations
                    print(f"总邻域搜索迭代次数: {total_neighborhood_iterations}")
                    print(f"平均每次SCA迭代的邻域搜索迭代次数: {avg_neighborhood_per_sca:.2f}")
        
        return sca_neighborhood_allocation, L_total_cost, L_nominal_cost, L_first_cost, L_best_cost, L_avg_latency

    # 如果方法是proportional_allocation，基于请求概率的分配方法
    elif method == 'proportional_allocation':
        print("使用基于请求概率的分配方法")
        print("  特点：根据每个interval的SP可缓存内容进行动态分配")
        print("  分配策略：基于SP请求概率的比例分配")
        print("  约束：满足节点空间容量限制")
        
        # 获取缓存节点列表
        cache_nodes = topology_manager.get_cache_nodes()
        num_cache_nodes = len(cache_nodes)
        
        # 为每个缓存节点计算缓存容量（每个节点使用固定的single_cache_capacity）
        node_cache_capacity = single_cache_capacity
        
        # 获取SP请求概率（从配置文件中读取）
        SP_proba = config['providers']['probabilities']
        print(f"SP请求概率: {SP_proba}")
        
        # 初始化分配
        proportional_allocation = {}
        for i, node in enumerate(cache_nodes):
            # 每个节点的实际缓存容量都是固定的single_cache_capacity
            actual_capacity = node_cache_capacity
            
            # 基于SP请求概率进行初始分配
            total_weight = sum(SP_proba)
            sp_ratios = [p / total_weight for p in SP_proba]
            
            # 根据比例分配缓存空间
            new_allocation = []
            for ratio in sp_ratios:
                allocated = int(actual_capacity * ratio)
                new_allocation.append(allocated)
            
            # 处理余数，确保总和等于实际容量
            total_allocated = sum(new_allocation)
            if total_allocated < actual_capacity:
                # 将余数分配给请求概率最高的SP
                max_sp_idx = SP_proba.index(max(SP_proba))
                new_allocation[max_sp_idx] += (actual_capacity - total_allocated)
            elif total_allocated > actual_capacity:
                # 从请求概率最低的SP中减去多余的部分
                min_sp_idx = SP_proba.index(min(SP_proba))
                new_allocation[min_sp_idx] -= (total_allocated - actual_capacity)
                if new_allocation[min_sp_idx] < 0:
                    new_allocation[min_sp_idx] = 0
            
            proportional_allocation[node] = new_allocation
        
        print(f"初始分配: {proportional_allocation}")
        
        # 准备输出文件
        allocations_file = os.path.join(results_dir, f"allocations_single_cache_capacity{single_cache_capacity}_request_rate{request_rate}_proportional_allocation.txt")
        f = open(allocations_file, "w") if save_allocations else None
        
        # 初始化成本列表
        L_total_cost = []
        L_nominal_cost = []
        L_first_cost = []
        L_best_cost = []
        L_avg_latency = []
        
        # 计算一次初始分配和最佳分配的成本，作为固定值
        request_nb = int(interval_size * request_rate)
        (_, fixed_first_cost, fixed_best_cost, _, _) = evaluate_cost(
            best_allocation, proportional_allocation, best_allocation, request_nb, initial_videos_proba
        )
        
        # 在nb_interval次迭代中不断优化
        for interval in range(nb_interval):
            #print(f"\n=== 第 {interval+1}/{nb_interval} 个时间间隔 ===")
            
            # 获取当前视频概率分布（包含可缓存内容）
            current_videos_proba = catalog()
            
            # 计算当前分配的成本
            (current_cost, _, _, current_latency, current_hit_rates) = evaluate_cost(
                proportional_allocation, proportional_allocation, best_allocation, request_nb, current_videos_proba
            )
            #print(f"当前成本: {current_cost:.4f}")
            #print(f"当前命中率: {current_hit_rates}")
            
            # 为每个缓存节点更新分配（基于SP可缓存内容）
            for node in cache_nodes:
                # 每个节点的实际缓存容量都是固定的single_cache_capacity
                actual_capacity = node_cache_capacity
                
                # 计算基于SP可缓存内容的分配比例
                # 使用每个SP的可缓存视频数量作为权重
                sp_weights = []
                for sp in range(nSP):
                    if cacheable_content and sp < len(cacheable_content):
                        sp_weight = len(cacheable_content[sp])  # 使用可缓存视频数量作为权重
                    else:
                        sp_weight = 1  # 默认权重
                    sp_weights.append(sp_weight)
                
                total_weight = sum(sp_weights)
                if total_weight > 0:
                    sp_ratios = [w / total_weight for w in sp_weights]
                else:
                    # 如果所有SP都没有可缓存内容，使用请求概率
                    sp_ratios = [p / sum(SP_proba) for p in SP_proba]
                
                # 根据比例分配缓存空间
                new_allocation = []
                for ratio in sp_ratios:
                    allocated = int(actual_capacity * ratio)
                    new_allocation.append(allocated)
                
                # 处理余数，确保总和等于实际容量
                total_allocated = sum(new_allocation)
                if total_allocated < actual_capacity:
                    # 将余数分配给权重最高的SP
                    max_sp_idx = sp_weights.index(max(sp_weights))
                    new_allocation[max_sp_idx] += (actual_capacity - total_allocated)
                elif total_allocated > actual_capacity:
                    # 从权重最低的SP中减去多余的部分
                    min_sp_idx = sp_weights.index(min(sp_weights))
                    new_allocation[min_sp_idx] -= (total_allocated - actual_capacity)
                    if new_allocation[min_sp_idx] < 0:
                        new_allocation[min_sp_idx] = 0
                
                # 确保所有分配都是非负的
                for i in range(len(new_allocation)):
                    if new_allocation[i] < 0:
                        new_allocation[i] = 0
                
                # 更新分配
                proportional_allocation[node] = new_allocation
            
            # 计算优化后的成本
            (nominal_cost, _, _, avg_latency, cache_node_hit_rates) = evaluate_cost(
                proportional_allocation, proportional_allocation, best_allocation, request_nb, current_videos_proba
            )
            
            # 打印优化结果
            #print(f"  调整后的分配: {proportional_allocation}")
            #print(f"  成本: {nominal_cost:.4f}")
            #print(f"  命中率: {cache_node_hit_rates}")
            
            # 保存成本
            L_total_cost.append(nominal_cost)
            L_nominal_cost.append(nominal_cost)
            L_first_cost.append(fixed_first_cost)
            L_best_cost.append(fixed_best_cost)
            L_avg_latency.append(avg_latency)
            
            # 保存分配到文件
            if save_allocations:
                f.write(f"Interval {interval+1}: {proportional_allocation}\n")
        
        # 关闭文件
        if save_allocations:
            f.close()
        
        # 打印最终结果
        print(f"\n最终基于请求概率的分配: {proportional_allocation}")
        
        return (proportional_allocation, L_total_cost, L_nominal_cost, L_first_cost, L_best_cost, L_avg_latency)
    
    # 如果方法是manual_allocation，使用手动输入的缓存分配
    elif method == 'manual_allocation':
        print("使用手动输入缓存分配方法")
        print("  特点：允许用户手动设置每个SP的缓存分配空间")
        print("  用途：用于对比性能上限，不考虑缓存节点空间限制")
        
        # 计算请求数量
        request_nb = int(interval_size * request_rate)
        
        # 从全局变量获取缓存节点列表
        cache_nodes = topology_manager.get_cache_nodes()
        
        # 从配置文件获取手动分配
        manual_allocation = get_manual_allocation(nSP, cache_nodes, config)
        
        # 为每个interval计算成本
        L_total_cost = []
        L_nominal_cost = []
        L_first_cost = []
        L_best_cost = []
        L_avg_latency = []
        
        for interval in range(nb_interval):
            # 使用当前interval的固定请求
            interval_fixed_requests = fixed_requests[interval] if fixed_requests and len(fixed_requests) > interval else None
            
            # 评估手动分配的性能
            # 检查是否需要输出调试信息
            debug_output = (debug_interval is not None and interval == debug_interval)
            (nominal_cost, first_cost, best_cost, avg_latency, cache_node_hit_rates) = evaluate_cost(
                manual_allocation, manual_allocation, best_allocation, request_nb, videos_proba, interval_fixed_requests, debug_output
            )
            
            L_total_cost.append(nominal_cost)
            L_nominal_cost.append(nominal_cost)
            L_first_cost.append(first_cost)
            L_best_cost.append(best_cost)
            L_avg_latency.append(avg_latency)
        
        print(f"手动分配成本: {sum(L_nominal_cost) / len(L_nominal_cost):.4f}")
        print(f"手动分配命中率: {cache_node_hit_rates}")
        
        # 返回手动分配和成本列表
        return manual_allocation, L_total_cost, L_nominal_cost, L_first_cost, L_best_cost, L_avg_latency
    
    # 如果方法是Q_learning，使用Q-learning算法（重构版本，与SCA方法保持一致）
    elif method == 'Q_learning':
        print("使用Q-learning算法（重构版本）")
        print("  特点：基于强化学习的多缓存节点协同优化")
        print("  结构：与SCA_ADMM和SCA_neighborhood_search保持一致")
        print("  奖励：每个缓存节点的命中率变化量")
        print("  动作：单位缓存移动（coeff * delta）")
        print("  策略：ε-greedy策略，合法动作检查")
        
        # 获取缓存节点列表
        cache_nodes = topology_manager.get_cache_nodes()
        num_cache_nodes = len(cache_nodes)
        
        # 初始化缓存分配（与SCA_ADMM相同的初始分配策略）
        q_learning_allocation = {}
        for i, node in enumerate(cache_nodes):
            actual_capacity = single_cache_capacity
            # 使用与SCA_ADMM相同的初始分配策略
            sp_video_proba_sum = []
            for sp in range(nSP):
                if cacheable_content and sp < len(cacheable_content):
                    sp_proba_sum = sum(initial_videos_proba[sp][v] for v in cacheable_content[sp])
                else:
                    sp_proba_sum = 0.0
                sp_video_proba_sum.append(sp_proba_sum)

            total_sp_proba = sum(sp_video_proba_sum) if sum(sp_video_proba_sum) > 0 else 1.0
            node_allocation = []
            remaining_cap = actual_capacity
            for sp in range(nSP):
                if sp == nSP - 1:
                    node_allocation.append(remaining_cap)
                else:
                    sp_alloc = int(actual_capacity * sp_video_proba_sum[sp] / total_sp_proba)
                    node_allocation.append(sp_alloc)
                    remaining_cap -= sp_alloc
            q_learning_allocation[node] = node_allocation

        print(f"初始分配: {q_learning_allocation}")

        # 计算一次初始分配和最佳分配的成本，作为固定值（与SCA_ADMM保持一致）
        request_nb = int(interval_size * request_rate)
        interval_fixed_requests = fixed_requests[0] if fixed_requests and len(fixed_requests) > 0 else None
        (_, fixed_first_cost, fixed_best_cost, _, _) = evaluate_cost(
            best_allocation, q_learning_allocation, best_allocation, request_nb, initial_videos_proba, interval_fixed_requests
        )

        # 为每个缓存节点创建状态空间和Q表
        states = {}
        state_to_index = {}
        Q = {}
        V = {}
        state_index = {}
        
        for node in cache_nodes:
            # 生成状态空间
            states[node] = states_nSP(single_cache_capacity, nSP, delta)
            # 创建状态到索引的映射
            state_to_index[node] = {}
            for idx, state in enumerate(states[node]):
                state_to_index[node][tuple(state)] = idx
            # 初始化Q表
            Q[node] = np.zeros((nSP**2, len(states[node])))
            # 初始化访问计数
            V[node] = np.zeros((nSP**2, len(states[node])))
            # 获取初始状态索引
            original_alloc = tuple(q_learning_allocation[node])
            state_index[node] = state_to_index[node].get(original_alloc, -1)
            #print(f"缓存节点 {node} 初始化完成，初始状态索引: {state_index[node]}")
        
        Memory = []  # 经验回放内存
        
        # 初始化成本记录列表（与SCA_ADMM保持一致）
        L_total_cost = []
        L_nominal_cost = []
        L_first_cost = []
        L_best_cost = []
        L_avg_latency = []
        
        # 使用对比模式中已生成的固定请求序列，确保公平比较
        # 如果外部没有传入fixed_requests，则使用默认的请求生成逻辑（与SCA方法保持一致）
        if fixed_requests is None or len(fixed_requests) == 0:
            fixed_requests = []
            
            # 获取可能的源节点（与对比模式保持一致）
            sp_nodes = topology_manager.get_sp_nodes()
            receiver_nodes = topology_manager.get_receiver_nodes()
            router_nodes = topology_manager.get_router_nodes()
            cache_nodes_list = topology_manager.get_cache_nodes()
            
            for interval in range(nb_interval):
                interval_requests = {
                    'requests': [],
                    'source_nodes': []
                }
                
                for _ in range(request_nb):
                    # 随机选择SP
                    sp_idx = rd.choices(range(nSP), weights=SP_proba)[0]
                    
                    # 随机选择视频
                    if cacheable_content and sp_idx < len(cacheable_content):
                        cacheable_videos = cacheable_content[sp_idx]
                        video_idx = rd.randint(0, len(cacheable_videos) - 1)
                        video_id = cacheable_videos[video_idx]
                    else:
                        video_id = rd.randint(0, nb_videos - 1)
                    
                    # 随机选择源节点（与对比模式保持一致）
                    possible_sources = []
                    if router_nodes:
                        possible_sources.extend(router_nodes)
                    if cache_nodes_list:
                        possible_sources.extend(cache_nodes_list)
                    
                    if possible_sources:
                        source_node = rd.choice(possible_sources)
                    else:
                        source_node = None
                    
                    interval_requests['requests'].append((sp_idx, video_id))
                    interval_requests['source_nodes'].append(source_node)
                
                fixed_requests.append(interval_requests)

        # 准备输出文件（与SCA方法保持一致）
        allocations_file = os.path.join(results_dir, f"allocations_single_cache_capacity{single_cache_capacity}_request_rate{request_rate}_Q_learning.txt")
        f = open(allocations_file, "w") if save_allocations else None
        
        # 初始化旧状态成本（用于计算奖励）
        old_nominal_cost = None
        
        # 主循环：处理每个时间间隔（与SCA方法保持一致的结构）
        for interval in range(nb_interval):
            current_videos_proba = catalog()
            interval_fixed_requests = fixed_requests[interval] if fixed_requests and len(fixed_requests) > interval else None
            
            # 检查是否需要输出调试信息
            debug_output = (debug_interval is not None and interval == debug_interval)
            
            # 评估当前分配的成本（与SCA方法保持一致）
            (nominal_cost, first_cost, best_cost, avg_latency, cache_node_hit_rates) = evaluate_cost(
                q_learning_allocation, q_learning_allocation, best_allocation, request_nb, current_videos_proba, interval_fixed_requests, debug_output
            )
            
            # 使用固定的初始成本和最佳成本
            first_cost = fixed_first_cost
            best_cost = fixed_best_cost
            
            # Q-learning参数调整（与SCA方法的时间间隔概念对应）
            if epsilon_decay:
                epsi = find_epsilon(interval)
            else:
                epsi = 0.2
            
            if alpha_scheduling:
                if interval == 0:
                    alfa = 0.5  # 提高初始学习率
                else:
                    alfa = 0.5 * (0.99 ** interval)  # 减缓衰减速度
                    alfa = max(alfa, 0.05)  # 提高最小学习率
            else:
                alfa = 0.3
            
            # 保存当前时间间隔的初始分配（与SCA方法保持一致）
            current_interval_allocation = {}
            for node in q_learning_allocation:
                current_interval_allocation[node] = q_learning_allocation[node].copy()
            
            old_allocation = deepcopy(q_learning_allocation)
            
            # 如果是第一次迭代，先计算旧状态的 nominal_cost 和每个缓存节点的旧命中率
            if interval == 0:
                (old_nominal_cost, _, _, _, old_cache_node_hit_rates) = evaluate_cost(
                    q_learning_allocation, q_learning_allocation, best_allocation, request_nb, videos_proba
                )
                # 初始化每个缓存节点的旧命中率字典
                old_cache_node_hit_rates = old_cache_node_hit_rates.copy()
            
            # 执行动作（多缓存节点版本）
            action, action_minus, action_plus, q_learning_allocation = take_action_multi_cache(
                q_learning_allocation, epsi, D, delta, Q, state_index, cache_nodes, nSP, states, state_to_index
            )
            
            # 保存分配
            if save_allocations and f:
                f.write(str(q_learning_allocation) + '\n')
            
            #print(f'时间间隔 {interval}: 分配: {q_learning_allocation}')
            
            # 计算新状态的 nominal_cost（未命中率）
            (nominal_cost, _, _, avg_latency, cache_node_hit_rates) = evaluate_cost(
                q_learning_allocation, q_learning_allocation, best_allocation, request_nb, current_videos_proba, interval_fixed_requests
            )
            
            # ========== 新奖励函数：每个缓存节点的命中率变化量 ==========
            # 为每个缓存节点计算独立的奖励
            new_gain = {}
            if interval == 0:
                # 第一次迭代无旧状态，奖励为0
                for node in cache_nodes:
                    new_gain[node] = 0.0
            else:
                # 为每个缓存节点计算基于其自身命中率变化的奖励
                for node in cache_nodes:
                    # 奖励 = 新命中率 - 旧命中率
                    old_hit_rate = old_cache_node_hit_rates.get(node, 0)
                    new_hit_rate = cache_node_hit_rates.get(node, 0)
                    if new_hit_rate > old_hit_rate:
                        node_reward = (new_hit_rate - old_hit_rate) * 100 + 0.1
                    else:
                        node_reward = (new_hit_rate - old_hit_rate) * 100
                    new_gain[node] = node_reward
            
            # 更新旧状态成本和旧命中率
            old_nominal_cost = nominal_cost
            old_cache_node_hit_rates = cache_node_hit_rates.copy()
            
            # ========== Q-learning 更新 ==========
            # 为每个缓存节点更新其Q表（参考library_rl.py的实现）
            for node in cache_nodes:
                # 获取新状态索引
                original_alloc = tuple(q_learning_allocation[node])
                state_index_prime = state_to_index[node].get(original_alloc, -1)
                # 确保状态索引有效
                if state_index_prime != -1:
                    # 计算最佳动作值（参考library_rl.py）
                    (best_score1, best_actions1) = af.find_max_list(Q[node][:, state_index_prime])
                    
                    # 更新Q表（参考library_rl.py的简单实现）
                    act = action[node] if action and node in action else 0
                    if act is not None and act < Q[node].shape[0] and state_index[node] < Q[node].shape[1]:
                        node_reward = new_gain[node] if node in new_gain else 0.0
                        old_q = Q[node][act, state_index[node]]
                        
                        # Q-learning更新公式：Q(s,a) = Q(s,a) + α * (r + γ * max_a' Q(s',a') - Q(s,a))
                        Q[node][act, state_index[node]] += alfa * (node_reward + gama * best_score1 - Q[node][act, state_index[node]])
                        
                        new_q = Q[node][act, state_index[node]]
                        #if interval % 10 == 0:
                            #print(f'  节点 {node} Q值更新: 动作{act}, {old_q:.4f} -> {new_q:.4f}, 奖励: {node_reward:.6f}')
                    
                    # 更新状态索引
                    state_index[node] = state_index_prime
            
            # 经验回放（多缓存节点版本，参考library_rl.py）
            if activate_memory:
                N = find_N(interval)
                Memory.append((q_learning_allocation, action, new_gain))
                
                for m in range(N):
                    if Memory:
                        [state_rd, action_rd, reward_rd] = rd.choice(Memory)
                        for node in cache_nodes:
                            # 获取状态索引
                            state_rd_index = get_state_index(state_rd[node], delta, states[node])
                            if state_rd_index != -1:
                                # 计算最佳动作值
                                (best_score1, best_actions1) = af.find_max_list(Q[node][:, state_rd_index])
                                # 更新Q表
                                act = action_rd[node] if action_rd and node in action_rd else 0
                                if act is not None and act < Q[node].shape[0] and state_rd_index < Q[node].shape[1]:
                                    reward_val = reward_rd[node] if reward_rd and node in reward_rd else 0.0
                                    Q[node][act, state_rd_index] += alfa * (reward_val + gama * best_score1 - Q[node][act, state_rd_index])
            
            # 记录成本（与SCA方法保持一致）
            total_cost = nominal_cost
            L_total_cost.append(total_cost)
            L_nominal_cost.append(nominal_cost)
            L_first_cost.append(first_cost)
            L_best_cost.append(best_cost)
            L_avg_latency.append(avg_latency)
        
        if save_allocations and f:
            f.close()
        
        # 打印最终结果（与SCA方法保持一致）
        print(f"\n最终Q-learning分配: {q_learning_allocation}")
        
        return (q_learning_allocation, L_total_cost, L_nominal_cost, L_first_cost, L_best_cost, L_avg_latency)
    
    # 如果方法是SARSA，使用SARSA算法（保持原有结构）
    elif method == 'SARSA':
        print("使用SARSA算法")
        print("  特点：基于强化学习的多缓存节点协同优化")
        print("  奖励：每个缓存节点的命中率变化量")
        print("  动作：单位缓存移动（coeff * delta）")
        print("  策略：ε-greedy策略，合法动作检查")
        
        # 获取缓存节点列表
        cache_nodes = topology_manager.get_cache_nodes()
        num_cache_nodes = len(cache_nodes)
        
        # 初始化缓存分配
        sarsa_allocation = {}
        for i, node in enumerate(cache_nodes):
            actual_capacity = single_cache_capacity
            # 使用均匀初始分配
            node_allocation = []
            remaining_cap = actual_capacity
            for sp in range(nSP):
                if sp == nSP - 1:
                    node_allocation.append(remaining_cap)
                else:
                    sp_alloc = int(actual_capacity / nSP)
                    node_allocation.append(sp_alloc)
                    remaining_cap -= sp_alloc
            sarsa_allocation[node] = node_allocation

        print(f"初始分配: {sarsa_allocation}")

        # 计算一次初始分配和最佳分配的成本，作为固定值（与SCA_ADMM保持一致）
        request_nb = int(interval_size * request_rate)
        interval_fixed_requests = fixed_requests[0] if fixed_requests and len(fixed_requests) > 0 else None
        (_, fixed_first_cost, fixed_best_cost, _, _) = evaluate_cost(
            best_allocation, sarsa_allocation, best_allocation, request_nb, initial_videos_proba, interval_fixed_requests
        )

        # 为每个缓存节点创建状态空间和Q表
        states = {}
        state_to_index = {}
        Q = {}
        V = {}
        state_index = {}
        
        for node in cache_nodes:
            # 生成状态空间
            states[node] = states_nSP(single_cache_capacity, nSP, delta)
            # 创建状态到索引的映射
            state_to_index[node] = {}
            for idx, state in enumerate(states[node]):
                state_to_index[node][tuple(state)] = idx
            # 初始化Q表
            Q[node] = np.zeros((nSP**2, len(states[node])))
            # 初始化访问计数
            V[node] = np.zeros((nSP**2, len(states[node])))
            # 获取初始状态索引
            original_alloc = tuple(sarsa_allocation[node])
            state_index[node] = state_to_index[node].get(original_alloc, -1)
            print(f"缓存节点 {node} 初始化完成，初始状态索引: {state_index[node]}")
        
        Memory = []  # 经验回放内存
        
        # 为每个缓存节点记录命中率变化量和奖励
        cache_node_hit_rate_changes = {node: [] for node in cache_nodes}
        cache_node_rewards = {node: [] for node in cache_nodes}
        
        # 准备输出文件
        print(f"保存分配历史: {save_allocations}")
        print(f"结果目录: {results_dir}")
        print(f"单个缓存节点容量: {single_cache_capacity}")
        print(f"请求率: {request_rate}")
        allocations_file = os.path.join(results_dir, f"allocations_single_cache_capacity{single_cache_capacity}_request_rate{request_rate}.txt")
        print(f"分配文件路径: {allocations_file}")
        f = open(allocations_file, "w") if save_allocations else None
        print(f"文件对象: {f}")
        
        # 初始化旧状态成本（用于计算奖励）
        old_nominal_cost = None
        
        for j in range(nb_interval):
            if epsilon_decay:
                epsi = find_epsilon(j)
            else:
                epsi = 0.2
            
            if alpha_scheduling:
                if j == 0:
                    alfa = 0.5  # 提高初始学习率
                else:
                    alfa = 0.5 * (0.99 ** j)  # 减缓衰减速度
                    alfa = max(alfa, 0.05)  # 提高最小学习率
            else:
                alfa = 0.3
            
            #print(f"迭代: {j} 学习率: {epsi:.4f} 学习率: {alfa:.4f}")
            old_allocation = deepcopy(sarsa_allocation)
            
            # 如果是第一次迭代，先计算旧状态的 nominal_cost 和每个缓存节点的旧命中率
            if j == 0:
                (old_nominal_cost, _, _, _, old_cache_node_hit_rates) = evaluate_cost(
                    sarsa_allocation, sarsa_allocation, best_allocation, request_nb, videos_proba
                )
                # 初始化每个缓存节点的旧命中率字典
                old_cache_node_hit_rates = old_cache_node_hit_rates.copy()
            
            # 执行动作（多缓存节点版本）
            action, action_minus, action_plus, sarsa_allocation = take_action_multi_cache(
                sarsa_allocation, epsi, D, delta, Q, state_index, cache_nodes, nSP, states, state_to_index
            )
            
            # 保存分配
            if save_allocations and f:
                f.write(str(sarsa_allocation) + '\n')
            
            print(f'迭代: {j} 分配: {sarsa_allocation}')
            
            # 计算新状态的 nominal_cost（未命中率）
            (nominal_cost, _, _, avg_latency, cache_node_hit_rates) = evaluate_cost(
                sarsa_allocation, sarsa_allocation, best_allocation, request_nb, videos_proba, fixed_requests
            )
            
            # 使用固定的初始成本和最佳成本
            first_cost = fixed_first_cost
            best_cost = fixed_best_cost
            
            # ========== 新奖励函数：每个缓存节点的命中率变化量 ==========
            # 为每个缓存节点计算独立的奖励
            new_gain = {}
            if j == 0:
                # 第一次迭代无旧状态，奖励为0
                for node in cache_nodes:
                    new_gain[node] = 0.0
            else:
                # 为每个缓存节点计算基于其自身命中率变化的奖励
                for node in cache_nodes:
                    # 奖励 = 新命中率 - 旧命中率
                    old_hit_rate = old_cache_node_hit_rates.get(node, 0)
                    new_hit_rate = cache_node_hit_rates.get(node, 0)
                    if new_hit_rate > old_hit_rate:
                        node_reward = (new_hit_rate - old_hit_rate) * 100 + 0.1
                    else:
                        node_reward = (new_hit_rate - old_hit_rate) * 100
                    new_gain[node] = node_reward
            
            # 记录每个缓存节点的命中率变化量和奖励
            for node in cache_nodes:
                if j > 0:
                    old_hit_rate = old_cache_node_hit_rates.get(node, 0)
                    new_hit_rate = cache_node_hit_rates.get(node, 0)
                    hit_rate_change = new_hit_rate - old_hit_rate
                    cache_node_hit_rate_changes[node].append(hit_rate_change)
                    cache_node_rewards[node].append(new_gain[node])
                else:
                    # 第一次迭代，没有变化量
                    cache_node_hit_rate_changes[node].append(0.0)
                    cache_node_rewards[node].append(0.0)
            
            # 更新旧状态成本和旧命中率
            old_nominal_cost = nominal_cost
            old_cache_node_hit_rates = cache_node_hit_rates.copy()
            

            
            # ========== SARSA 更新 ==========
            if method == 'SARSA':
                # 为每个缓存节点更新其Q表
                for node in cache_nodes:
                    # 获取新状态索引
                    original_alloc = tuple(sarsa_allocation[node])
                    state_index_prime = state_to_index[node].get(original_alloc, -1)
                    # 确保状态索引有效
                    if state_index_prime != -1:
                        # 计算下一个动作
                        next_action, next_action_minus, next_action_plus, _ = take_action_multi_cache(
                            sarsa_allocation, epsi, D, delta, Q, state_index, cache_nodes, nSP, states, state_to_index
                        )
                        # 更新Q表
                        act = action[node]
                        next_act = next_action[node]
                        if (act is not None and next_act is not None and 
                            act < Q[node].shape[0] and state_index[node] < Q[node].shape[1] and 
                            next_act < Q[node].shape[0] and state_index_prime < Q[node].shape[1]):
                            node_reward = new_gain[node]
                            old_q = Q[node][act, state_index[node]]
                            Q[node][act, state_index[node]] += alfa * (node_reward + gama * Q[node][next_act, state_index_prime] - Q[node][act, state_index[node]])
                            new_q = Q[node][act, state_index[node]]
                            if j % 10 == 0:
                                print(f'  节点 {node} SARSA更新: 动作{act}, {old_q:.4f} -> {new_q:.4f}, 奖励: {node_reward:.6f}')
                        # 更新状态索引
                        state_index[node] = state_index_prime
            
            # 经验回放（多缓存节点版本）
            if activate_memory:
                N = find_N(j)
                Memory.append((sarsa_allocation, action, new_gain))
                
                for m in range(N):
                    if Memory:
                        [state_rd, action_rd, reward_rd] = rd.choice(Memory)
                        if method == 'SARSA':
                            for node in cache_nodes:
                                state_rd_index = get_state_index(state_rd[node], delta, states[node])
                                if state_rd_index != -1:
                                    # 计算下一个动作
                                    next_action, _, _, _ = take_action_multi_cache(
                                        state_rd, epsi, D, delta, Q, state_index, cache_nodes, nSP, states, state_to_index
                                    )
                                    act = action_rd[node]
                                    next_act = next_action[node]
                                    if (act is not None and next_act is not None and 
                                        act < Q[node].shape[0] and state_rd_index < Q[node].shape[1] and 
                                        next_act < Q[node].shape[0]):
                                        Q[node][act, state_rd_index] += alfa * (reward_rd[node] + gama * Q[node][next_act, state_rd_index] - Q[node][act, state_rd_index])
            
            # 记录成本
            total_cost = nominal_cost
            L_total_cost.append(total_cost)
            L_nominal_cost.append(nominal_cost)
            L_first_cost.append(first_cost)
            L_best_cost.append(best_cost)
            L_avg_latency.append(avg_latency)
        
        if save_allocations and f:
            f.close()
        
        # 打印最终结果
        print(f"\n最终SARSA分配: {sarsa_allocation}")
        
        return (sarsa_allocation, L_total_cost, L_nominal_cost, L_first_cost, L_best_cost, L_avg_latency)
    
    # 默认情况：单缓存节点或未识别的多缓存节点方法
    else:
        L_total_cost = []
        L_nominal_cost = []
        L_first_cost = []
        L_best_cost = []
        L_avg_latency = []
        
        request_nb = int(interval_size * request_rate)
        
        # 计算一次初始分配和最佳分配的成本，作为固定值
        (_, fixed_first_cost, fixed_best_cost, _, _) = evaluate_cost(
            best_allocation, allocation, best_allocation, request_nb, videos_proba
        )
    
    # 检查是否是多缓存节点的情况
    if isinstance(allocation, dict):
        # 多缓存节点的情况
        cache_nodes = topology_manager.get_cache_nodes()
        num_cache_nodes = len(cache_nodes)
        
        first_allocation = deepcopy(allocation)
        
        # 为每个缓存节点创建单独的Q表
        Q = {}
        V = {}
        states = {}
        state_index = {}
        state_to_index = {}  # 状态到索引的映射字典，用于快速查找
        
        # 为每个缓存节点计算状态空间和初始化Q表
        for node in cache_nodes:
            # 计算每个缓存节点的缓存容量（每个节点使用固定的single_cache_capacity）
            node_cache_capacity = single_cache_capacity
            print(f"为缓存节点 {node} 生成状态空间...")
            print(f"节点缓存容量: {node_cache_capacity}, nSP: {nSP}, delta: {delta}")
            # 生成状态空间
            start_time = time.time()
            states[node] = states_nSP(node_cache_capacity, nSP, delta)
            end_time = time.time()
            print(f"状态空间生成完成，耗时: {end_time - start_time:.2f}秒")
            print(f"状态数量: {len(states[node])}")
            # 创建状态到索引的映射字典
            state_to_index[node] = {}
            for idx, state in enumerate(states[node]):
                # 使用原始顺序的元组作为键，保留SP的顺序信息
                state_to_index[node][tuple(state)] = idx
            # 初始化Q表
            Q[node] = np.zeros((nSP**2, len(states[node])))
            # 初始化访问计数
            V[node] = np.zeros((nSP**2, len(states[node])))
            # 获取初始状态索引
            original_alloc = tuple(allocation[node])
            state_index[node] = state_to_index[node].get(original_alloc, -1)
            print(f"缓存节点 {node} 初始化完成，初始状态索引: {state_index[node]}")
        
        Memory = []  # 经验回放内存
        
        # 为每个缓存节点记录命中率变化量和奖励
        cache_node_hit_rate_changes = {node: [] for node in cache_nodes}
        cache_node_rewards = {node: [] for node in cache_nodes}
        
        # 准备输出文件
        print(f"保存分配历史: {save_allocations}")
        print(f"结果目录: {results_dir}")
        print(f"单个缓存节点容量: {single_cache_capacity}")
        print(f"请求率: {request_rate}")
        allocations_file = os.path.join(results_dir, f"allocations_single_cache_capacity{single_cache_capacity}_request_rate{request_rate}.txt")
        print(f"分配文件路径: {allocations_file}")
        f = open(allocations_file, "w") if save_allocations else None
        print(f"文件对象: {f}")
        
        # 初始化旧状态成本（用于计算奖励）
        old_nominal_cost = None
        
        for j in range(nb_interval):
            if epsilon_decay:
                epsi = find_epsilon(j)
            else:
                epsi = 0.2
            
            if alpha_scheduling:
                if j == 0:
                    alfa = 0.5  # 提高初始学习率
                else:
                    alfa = 0.5 * (0.99 ** j)  # 减缓衰减速度
                    alfa = max(alfa, 0.05)  # 提高最小学习率
            else:
                alfa = 0.3
            
            #print(f"迭代: {j} 学习率: {epsi:.4f} 学习率: {alfa:.4f}")
            old_allocation = deepcopy(allocation)
            
            # 如果是第一次迭代，先计算旧状态的 nominal_cost 和每个缓存节点的旧命中率
            if j == 0:
                (old_nominal_cost, _, _, _, old_cache_node_hit_rates) = evaluate_cost(
                    allocation, first_allocation, best_allocation, request_nb, videos_proba
                )
                # 初始化每个缓存节点的旧命中率字典
                old_cache_node_hit_rates = old_cache_node_hit_rates.copy()
            
            # 执行动作（多缓存节点版本）
            action, action_minus, action_plus, allocation = take_action_multi_cache(
                allocation, epsi, D, delta, Q, state_index, cache_nodes, nSP, states, state_to_index
            )
            
            # 保存分配
            if save_allocations and f:
                f.write(str(allocation) + '\n')
            
            print(f'迭代: {j} 分配: {allocation}')
            
            # 计算新状态的 nominal_cost（未命中率）
            (nominal_cost, _, _, avg_latency, cache_node_hit_rates) = evaluate_cost(
                allocation, first_allocation, best_allocation, request_nb, videos_proba, fixed_requests
            )
            
            # 使用固定的初始成本和最佳成本
            first_cost = fixed_first_cost
            best_cost = fixed_best_cost
            
            # ========== 新奖励函数：每个缓存节点的命中率变化量 ==========
            # 为每个缓存节点计算独立的奖励
            new_gain = {}
            if j == 0:
                # 第一次迭代无旧状态，奖励为0
                for node in cache_nodes:
                    new_gain[node] = 0.0
            else:
                # 为每个缓存节点计算基于其自身命中率变化的奖励
                for node in cache_nodes:
                    # 奖励 = 新命中率 - 旧命中率
                    old_hit_rate = old_cache_node_hit_rates.get(node, 0)
                    new_hit_rate = cache_node_hit_rates.get(node, 0)
                    if new_hit_rate > old_hit_rate:
                        node_reward = (new_hit_rate - old_hit_rate) * 100 + 0.1
                    else:
                        node_reward = (new_hit_rate - old_hit_rate) * 100
                    new_gain[node] = node_reward
            
            # 记录每个缓存节点的命中率变化量和奖励
            for node in cache_nodes:
                if j > 0:
                    old_hit_rate = old_cache_node_hit_rates.get(node, 0)
                    new_hit_rate = cache_node_hit_rates.get(node, 0)
                    hit_rate_change = new_hit_rate - old_hit_rate
                    cache_node_hit_rate_changes[node].append(hit_rate_change)
                    cache_node_rewards[node].append(new_gain[node])
                else:
                    # 第一次迭代，没有变化量
                    cache_node_hit_rate_changes[node].append(0.0)
                    cache_node_rewards[node].append(0.0)
            
            # 更新旧状态成本和旧命中率
            old_nominal_cost = nominal_cost
            old_cache_node_hit_rates = cache_node_hit_rates.copy()
            

            
            # ========== Q-learning 更新 ==========
            if method == 'Q_learning':
                # 为每个缓存节点更新其Q表
                for node in cache_nodes:
                    # 获取新状态索引
                    original_alloc = tuple(allocation[node])
                    state_index_prime = state_to_index[node].get(original_alloc, -1)
                    # 确保状态索引有效
                    if state_index_prime != -1:
                        # 计算最佳动作值
                        (best_score1, best_actions1) = af.find_max_list(Q[node][:, state_index_prime])
                        # 更新Q表
                        act = best_actions1 #action[node]
                        if act is not None and act < Q[node].shape[0] and state_index[node] < Q[node].shape[1]:
                            node_reward = new_gain[node]
                            old_q = Q[node][act, state_index[node]]
                            Q[node][act, state_index[node]] += alfa * (node_reward + gama * best_score1 - Q[node][act, state_index[node]])
                            new_q = Q[node][act, state_index[node]]
                            if j % 10 == 0:
                                print(f'  节点 {node} Q值更新: 动作{act}, {old_q:.4f} -> {new_q:.4f}, 奖励: {node_reward:.6f}')
                        # 更新状态索引
                        state_index[node] = state_index_prime
            elif method == 'SARSA':
                # 为每个缓存节点更新其Q表
                for node in cache_nodes:
                    # 获取新状态索引
                    original_alloc = tuple(allocation[node])
                    state_index_prime = state_to_index[node].get(original_alloc, -1)
                    # 确保状态索引有效
                    if state_index_prime != -1:
                        # 计算下一个动作
                        next_action, next_action_minus, next_action_plus, _ = take_action_multi_cache(
                            allocation, epsi, D, delta, Q, state_index, cache_nodes, nSP, states, state_to_index
                        )
                        # 更新Q表
                        act = action[node]
                        next_act = next_action[node]
                        if (act is not None and next_act is not None and 
                            act < Q[node].shape[0] and state_index[node] < Q[node].shape[1] and 
                            next_act < Q[node].shape[0] and state_index_prime < Q[node].shape[1]):
                            node_reward = new_gain[node]
                            Q[node][act, state_index[node]] += alfa * (node_reward + gama * Q[node][next_act, state_index_prime] - Q[node][act, state_index[node]])
                        # 更新状态索引
                        state_index[node] = state_index_prime
            elif method == 'COLLABORATIVE':
                # 协同优化方法的预留接口
                pass
            elif method == 'DQN':
                if not PYTORCH_AVAILABLE:
                    raise RuntimeError("PyTorch未安装，无法使用DQN方法。请先安装PyTorch: pip install torch")
                
                # 为每个缓存节点记录命中率变化量和奖励
                cache_node_hit_rate_changes = {node: [] for node in cache_nodes}
                cache_node_rewards = {node: [] for node in cache_nodes}
                
                Memory = []  # 经验回放内存
                
                # 准备输出文件
                print(f"保存分配历史: {save_allocations}")
                print(f"结果目录: {results_dir}")
                print(f"单个缓存节点容量: {single_cache_capacity}")
                print(f"请求率: {request_rate}")
                allocations_file = os.path.join(results_dir, f"allocations_single_cache_capacity{single_cache_capacity}_request_rate{request_rate}.txt")
                print(f"分配文件路径: {allocations_file}")
                f = open(allocations_file, "w") if save_allocations else None
                print(f"文件对象: {f}")
                
                first_allocation = deepcopy(allocation)
                
                # 为每个缓存节点初始化DQN和目标网络
                dqn_models = {}
                target_dqn_models = {}
                replay_buffers = {}
                state_dims = {}
                
                for node in cache_nodes:
                    node_cache_capacity = single_cache_capacity
                    state_dim = len(allocation[node])
                    action_dim = nSP**2
                    
                    dqn_models[node] = DQNNetwork(state_dim, action_dim)
                    target_dqn_models[node] = DQNNetwork(state_dim, action_dim)
                    target_dqn_models[node].load_state_dict(dqn_models[node].state_dict())
                    replay_buffers[node] = ReplayBuffer(capacity=10000)
                    state_dims[node] = state_dim
                
                optimizer = optim.Adam(dqn_models[cache_nodes[0]].parameters(), lr=0.001)
                
                # DQN训练循环
                target_update_freq = 10
                batch_size = 32
                
                old_nominal_cost = None
                
                for j in range(nb_interval):
                    if epsilon_decay:
                        epsi = find_epsilon(j)
                    else:
                        epsi = 0.2
                    
                    old_allocation = deepcopy(allocation)
                    
                    # 如果是第一次迭代，先计算旧状态的 nominal_cost 和每个缓存节点的旧命中率
                    if j == 0:
                        (old_nominal_cost, _, _, _, old_cache_node_hit_rates) = evaluate_cost(
                            allocation, first_allocation, best_allocation, request_nb, videos_proba
                        )
                        old_cache_node_hit_rates = old_cache_node_hit_rates.copy()
                    
                    # 动态调整delta
                    current_delta = delta * (delta_decay ** j)
                    current_delta = max(current_delta, min_delta)
                    
                    # DQN动作选择和执行
                    actions_selected = {}
                    for node in cache_nodes:
                        current_state = allocation[node]
                        
                        # epsilon-greedy动作选择
                        if rd.random() < epsi:
                            # 随机动作
                            act = rd.randint(0, nSP**2 - 1)
                        else:
                            # 使用DQN选择最佳动作
                            with torch.no_grad():
                                state_tensor = torch.FloatTensor(current_state)
                                q_values = dqn_models[node](state_tensor)
                                act = q_values.argmax().item()
                        
                        actions_selected[node] = act
                        
                        # 执行动作：解码动作并更新分配
                        action_sp_from = act // nSP
                        action_sp_to = act % nSP
                        
                        # 确保动作合法
                        if current_state[action_sp_from] >= current_delta:
                            allocation[node][action_sp_from] = max(0, current_state[action_sp_from] - current_delta)
                            allocation[node][action_sp_to] = min(single_cache_capacity, current_state[action_sp_to] + current_delta)
                    
                    # 保存分配
                    if save_allocations and f:
                        f.write(str(allocation) + '\n')
                    
                    print(f'迭代: {j} 分配: {allocation}')
                    
                    # 计算新状态的 nominal_cost（未命中率）
                    (nominal_cost, _, _, avg_latency, cache_node_hit_rates) = evaluate_cost(
                        allocation, first_allocation, best_allocation, request_nb, videos_proba
                    )
                    
                    # 使用固定的初始成本和最佳成本
                    first_cost = fixed_first_cost
                    best_cost = fixed_best_cost
                    
                    # ========== 奖励函数：每个缓存节点的命中率变化量 ==========
                    new_gain = {}
                    if j == 0:
                        for node in cache_nodes:
                            new_gain[node] = 0.0
                    else:
                        for node in cache_nodes:
                            old_hit_rate = old_cache_node_hit_rates.get(node, 0)
                            new_hit_rate = cache_node_hit_rates.get(node, 0)
                            if new_hit_rate > old_hit_rate:
                                node_reward = (new_hit_rate - old_hit_rate) * 100 + 0.1
                            else:
                                node_reward = (new_hit_rate - old_hit_rate) * 100
                            new_gain[node] = node_reward
                    
                    # 记录每个缓存节点的命中率变化量和奖励
                    for node in cache_nodes:
                        if j > 0:
                            old_hit_rate = old_cache_node_hit_rates.get(node, 0)
                            new_hit_rate = cache_node_hit_rates.get(node, 0)
                            hit_rate_change = new_hit_rate - old_hit_rate
                            cache_node_hit_rate_changes[node].append(hit_rate_change)
                            cache_node_rewards[node].append(new_gain[node])
                        else:
                            cache_node_hit_rate_changes[node].append(0.0)
                            cache_node_rewards[node].append(0.0)
                    
                    # 更新旧状态成本和旧命中率
                    old_nominal_cost = nominal_cost
                    old_cache_node_hit_rates = cache_node_hit_rates.copy()
                    
                    # 可选：打印调试信息
                    if j % 10 == 0:
                        print(f'\n=== 迭代 {j} 详细调试信息 ===')
                        print(f'  当前delta: {current_delta:.2f}')
                        print(f'  旧未命中率: {old_nominal_cost:.4f}')
                        print(f'  新未命中率: {nominal_cost:.4f}')
                        for node in cache_nodes:
                            # 计算分配变化
                            if j > 0:
                                allocation_change = [new - old for new, old in zip(allocation[node], old_allocation[node])]
                                print(f'  节点 {node} 分配变化: {allocation_change}')
                            else:
                                print(f'  节点 {node} 初始分配: {allocation[node]}')
                            print(f'  节点 {node} 命中率: {cache_node_hit_rates[node]:.4f}')
                            if j > 0:
                                print(f'  节点 {node} 奖励: {new_gain[node]:.6f}')
                    
                    # 将经验添加到回放缓冲区
                    for node in cache_nodes:
                        next_state = allocation[node]
                        reward = new_gain[node]
                        done = (j == nb_interval - 1)
                        replay_buffers[node].push(
                            old_allocation[node], 
                            actions_selected[node],  # 使用当前节点执行的动作
                            reward, 
                            next_state, 
                            done
                        )
                    
                    # DQN训练：从经验回放中采样并更新
                    for node in cache_nodes:
                        if len(replay_buffers[node]) >= batch_size:
                            batch = replay_buffers[node].sample(batch_size)
                            
                            states_batch = torch.FloatTensor(np.array([exp[0] for exp in batch]))
                            actions_batch = torch.LongTensor([exp[1] for exp in batch])
                            rewards_batch = torch.FloatTensor([exp[2] for exp in batch])
                            next_states_batch = torch.FloatTensor(np.array([exp[3] for exp in batch]))
                            dones_batch = torch.FloatTensor([exp[4] for exp in batch])
                            
                            # 计算当前Q值
                            current_q_values = dqn_models[node](states_batch).gather(1, actions_batch.unsqueeze(1))
                            
                            # 计算目标Q值
                            with torch.no_grad():
                                next_q_values = target_dqn_models[node](next_states_batch).max(1)[0]
                                target_q_values = rewards_batch + gama * next_q_values * (1 - dones_batch)
                            
                            # 计算损失并更新
                            loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
                            optimizer.zero_grad()
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(dqn_models[node].parameters(), 1.0)
                            optimizer.step()
                    
                    # 定期更新目标网络
                    if j % target_update_freq == 0:
                        for node in cache_nodes:
                            target_dqn_models[node].load_state_dict(dqn_models[node].state_dict())
                    
                    # 打印调试信息
                    if j % 10 == 0:
                        print(f'\n=== 迭代 {j} DQN训练信息 ===')
                        print(f'  旧未命中率: {old_nominal_cost:.4f}')
                        print(f'  新未命中率: {nominal_cost:.4f}')
                        for node in cache_nodes:
                            if j > 0:
                                allocation_change = [new - old for new, old in zip(allocation[node], old_allocation[node])]
                                print(f'  节点 {node} 分配变化: {allocation_change}')
                            else:
                                print(f'  节点 {node} 初始分配: {allocation[node]}')
                            print(f'  节点 {node} 命中率: {cache_node_hit_rates[node]:.4f}')
                            if j > 0:
                                print(f'  节点 {node} 奖励: {new_gain[node]:.6f}')
                            with torch.no_grad():
                                state_tensor = torch.FloatTensor(allocation[node])
                                q_vals = dqn_models[node](state_tensor)
                                print(f'  节点 {node} DQN Q值: max={q_vals.max().item():.4f}, min={q_vals.min().item():.4f}, mean={q_vals.mean().item():.4f}')
                    
                    # 记录成本
                    total_cost = nominal_cost
                    L_total_cost.append(total_cost)
                    L_nominal_cost.append(nominal_cost)
                    L_first_cost.append(first_cost)
                    L_best_cost.append(best_cost)
                    L_avg_latency.append(avg_latency)
                
                if f:
                    f.close()
                
                # 绘制每个缓存节点的命中率变化量和奖励图表
                time_points = [i * interval_size for i in range(nb_interval)]
                for node in cache_nodes:
                    plt.figure(figsize=(12, 6))
                    plt.plot(time_points, cache_node_hit_rate_changes[node], label='Hit Rate Change')
                    plt.plot(time_points, cache_node_rewards[node], label='Reward')
                    plt.xlabel('Time (seconds)')
                    plt.ylabel('Value')
                    plt.title(f'Cache Node {node}: Hit Rate Change and Reward (Request Rate: {request_rate})')
                    plt.legend()
                    plt.grid(True)
                    node_plot_path = os.path.join(figures_dir, f'cache_node_{node}_hit_rate_reward_{request_rate}.png')
                    plt.savefig(node_plot_path)
                    plt.close()
                    print(f"已保存缓存节点 {node} 的命中率变化和奖励图表: {node_plot_path}")
                
                return [L_total_cost, L_nominal_cost, L_first_cost, L_best_cost, L_avg_latency]
            
            # 记录成本（用于绘图，不影响学习）
            # 这里 total_cost 仅为记录，不再用于奖励
            total_cost = nominal_cost  # 直接使用未命中率作为总成本
            L_total_cost.append(total_cost)
            L_nominal_cost.append(nominal_cost)
            L_first_cost.append(first_cost)
            L_best_cost.append(best_cost)
            L_avg_latency.append(avg_latency)
            
            # 经验回放
            if activate_memory:
                N = find_N(j)
                # 存储经验：(状态, 动作, 奖励)
                # 状态是每个节点的分配字典，动作是每个节点的动作字典，奖励是每个节点的独立奖励
                Memory.append((allocation, action, new_gain))
                
                if len(Memory) > 1000:
                    Memory.pop(0)
                
                for m in range(N):
                    if Memory:
                        [state_rd, action_rd, reward_rd] = rd.choice(Memory)
                        if method == 'Q_learning':
                            for node in cache_nodes:
                                # 获取状态索引
                                state_rd_index = get_state_index(state_rd[node], delta, states[node])
                                if state_rd_index != -1:
                                    # 计算最佳动作值
                                    (best_score1, best_actions1) = af.find_max_list(Q[node][:, state_rd_index])
                                    # 更新Q表
                                    act = action_rd[node]
                                    if act is not None and act < Q[node].shape[0] and state_rd_index < Q[node].shape[1]:
                                        Q[node][act, state_rd_index] += alfa * (reward_rd[node] + gama * best_score1 - Q[node][act, state_rd_index])
                        elif method == 'SARSA':
                            for node in cache_nodes:
                                state_rd_index = get_state_index(state_rd[node], delta, states[node])
                                if state_rd_index != -1:
                                    # 计算下一个动作
                                    next_action, _, _, _ = take_action_multi_cache(
                                        state_rd, epsi, D, delta, Q, state_index, cache_nodes, nSP, states, state_to_index
                                    )
                                    act = action_rd[node]
                                    next_act = next_action[node]
                                    if (act is not None and next_act is not None and 
                                        act < Q[node].shape[0] and state_rd_index < Q[node].shape[1] and 
                                        next_act < Q[node].shape[0]):
                                        next_state_index = get_state_index(state_rd[node], delta, states[node])
                                        if next_state_index != -1:
                                            Q[node][act, state_rd_index] += alfa * (reward_rd[node] + gama * Q[node][next_act, next_state_index] - Q[node][act, state_rd_index])
        
        if f:
            f.close()
        
        # 绘制每个缓存节点的命中率变化量和奖励图表
        time_points = [i * interval_size for i in range(nb_interval)]
        for node in cache_nodes:
            plt.figure(figsize=(12, 6))
            plt.plot(time_points, cache_node_hit_rate_changes[node], label='Hit Rate Change')
            plt.plot(time_points, cache_node_rewards[node], label='Reward')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Value')
            plt.title(f'Cache Node {node}: Hit Rate Change and Reward (Request Rate: {request_rate})')
            plt.legend()
            plt.grid(True)
            node_plot_path = os.path.join(figures_dir, f'cache_node_{node}_hit_rate_reward_{request_rate}.png')
            plt.savefig(node_plot_path)
            plt.close()
            print(f"已保存缓存节点 {node} 的命中率变化和奖励图表: {node_plot_path}")
    else:
        # 单缓存节点的情况（兼容旧格式，同样修改奖励函数）
        states = states_nSP(single_cache_capacity, nSP, delta)
        first_allocation = list(allocation)
        state_index = get_state_index(allocation, delta, states)
        
        Q = np.zeros((nSP**2, len(states)))
        V = np.zeros((nSP**2, len(states)))
        
        M = 3600  # alfa调度参数
        ep = 0.01  # alfa调度参数
        alfa = 0.9  # 初始学习率
        
        # 计算一次初始分配和最佳分配的成本，作为固定值
        (_, fixed_first_cost, fixed_best_cost, _, _) = evaluate_cost(
            best_allocation, allocation, best_allocation, request_nb, videos_proba
        )
        
        Memory = []  # 经验回放内存
        
        # 准备输出文件
        print(f"保存分配历史: {save_allocations}")
        print(f"结果目录: {results_dir}")
        print(f"单个缓存节点容量: {single_cache_capacity}")
        print(f"请求率: {request_rate}")
        allocations_file = os.path.join(results_dir, f"allocations_single_cache_capacity{single_cache_capacity}_request_rate{request_rate}.txt")
        print(f"分配文件路径: {allocations_file}")
        f = open(allocations_file, "w") if save_allocations else None
        print(f"文件对象: {f}")
        
        old_nominal_cost = None
        
        for j in range(nb_interval):
            if epsilon_decay:
                epsi = find_epsilon(j)
            else:
                epsi = 0.2
            
            if alpha_scheduling:
                if j == 0:
                    alfa = alfa
                else:
                    alfa = alfa * ((1 - 1 / (1 + M + j)) ** (0.5 + ep))
            else:
                alfa = 0.9
            
            old_allocation = deepcopy(allocation)
            
            if j == 0:
                (old_nominal_cost, _, _, _, _) = evaluate_cost(
                    allocation, first_allocation, best_allocation, request_nb, videos_proba
                )
            
            # 执行动作
            action, action_minus, action_plus, allocation = take_action(allocation, epsi, D, delta, Q, state_index)
            
            # 保存分配
            if save_allocations and f:
                f.write(str(allocation) + '\n')
            
            print(f'迭代: {j} 分配: {allocation}')
            
            # 计算新状态的 nominal_cost
            (nominal_cost, _, _, avg_latency, _) = evaluate_cost(
                allocation, first_allocation, best_allocation, request_nb, videos_proba
            )
            
            # 使用固定的初始成本和最佳成本
            first_cost = fixed_first_cost
            best_cost = fixed_best_cost
            
            # 奖励 = 旧未命中率 - 新未命中率
            if j == 0:
                reward = 0.0
            else:
                reward = old_nominal_cost - nominal_cost
            old_nominal_cost = nominal_cost
            
            # 更新Q表
            state_index_prime = get_state_index(allocation, delta, states)
            (best_score1, best_actions1) = af.find_max_list(Q[:, state_index_prime])
            
            if action_plus == action_minus:
                for act in range(nSP):
                    if method == 'Q_learning':
                        Q[act * nSP][state_index] += alfa * (reward + gama * best_score1 - Q[act][state_index])
                    if method == 'SARSA':
                        Q[act * nSP][state_index] += alfa * (reward + gama * Q[act][state_index_prime] - Q[act][state_index])
                V[action][state_index] += 1
            else:
                if method == 'Q_learning':
                    Q[action][state_index] += alfa * (reward + gama * best_score1 - Q[action][state_index])
                if method == 'SARSA':
                    Q[action][state_index] += alfa * (reward + gama * Q[action][state_index_prime] - Q[action][state_index])
            
            state_index = state_index_prime
            total_cost = nominal_cost
            L_total_cost.append(total_cost)
            L_nominal_cost.append(nominal_cost)
            L_first_cost.append(first_cost)
            L_best_cost.append(best_cost)
            L_avg_latency.append(avg_latency)
            
            # 经验回放
            if activate_memory:
                N = find_N(j)
                Memory.append((allocation, action, reward))
                
                for m in range(N):
                    if Memory:
                        [state_rd, action_rd, reward_rd] = rd.choice(Memory)
                        state_rd_index = get_state_index(state_rd, delta, states)
                        if action_plus == action_minus:
                            for act in range(nSP):
                                if method == 'Q_learning':
                                    Q[act * nSP][state_rd_index] += alfa * (reward_rd + gama * best_score1 - Q[act][state_rd_index])
                                if method == 'SARSA':
                                    Q[act * nSP][state_rd_index] += alfa * (reward_rd + gama * Q[act][state_index_prime] - Q[act][state_rd_index])
                            V[action][state_index] += 1
                        else:
                            if method == 'Q_learning':
                                Q[action_rd][state_rd_index] += alfa * (reward_rd + gama * best_score1 - Q[action][state_rd_index])
                            if method == 'SARSA':
                                Q[action_rd][state_rd_index] += alfa * (reward_rd + gama * Q[action][state_index_prime] - Q[action][state_rd_index])
        
        if f:
            f.close()
    
    return [L_total_cost, L_nominal_cost, L_first_cost, L_best_cost, L_avg_latency]

def compute_estimated_cost(SP_proba, allocation, videos):
    """
    计算估计成本
    """
    if not allocation:
        return 1.0
    result = sum(SP_proba[k] * cacheability[k] * videos[k][i] for k in range(len(SP_proba)) for i in range(allocation[k]))
    return (1 - result)

def plot_results(total_cost, nominal_cost, cost_first, cost_best, avg_latency, interval_size, request_rate):
    """
    绘制结果图表
    支持多缓存节点的情况
    """
    import matplotlib.pyplot as plt
    time = [i * interval_size for i in range(len(total_cost))]
    
    # 成本图表
    plt.figure(figsize=(12, 6))
    plt.plot(time, total_cost, label='Total Cost')
    plt.plot(time, nominal_cost, label='Nominal Cost')
    plt.plot(time, cost_first, label='First Allocation Cost')
    plt.plot(time, cost_best, label='Best Allocation Cost')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Cost')
    plt.title(f'Cost Evolution (Request Rate: {request_rate})')
    plt.legend()
    plt.grid(True)
    cost_plot_path = os.path.join(figures_dir, f'cost_evolution_{request_rate}.png')
    plt.savefig(cost_plot_path)
    plt.close()
    
    # 时延图表（如果启用了网络模拟）
    if network_enabled and avg_latency:
        plt.figure(figsize=(12, 6))
        plt.plot(time, avg_latency, label='Average Latency')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Latency (ms)')
        plt.title(f'Latency Evolution (Request Rate: {request_rate})')
        plt.legend()
        plt.grid(True)
        latency_plot_path = os.path.join(figures_dir, f'latency_evolution_{request_rate}.png')
        plt.savefig(latency_plot_path)
        plt.close()
    
    # 拓扑可视化（如果启用了拓扑）
    if topology_manager:
        # 禁用中文字符警告
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        
        plt.figure(figsize=(12, 8))
        topology_manager.visualize(plt.gcf(), show=False)
        topology_plot_path = os.path.join(figures_dir, f'topology_{topology_type}.png')
        plt.savefig(topology_plot_path)
        plt.close()

def main(method=None, debug_interval=None):
    """
    主函数
    
    参数:
    method: 要运行的优化方法
    debug_interval: 要输出调试信息的interval编号，如果为None则禁用调试输出
    """
    init()
    
    # 如果debug_interval为None，尝试从配置文件中读取
    if debug_interval is None and 'debug_interval' in config['simulation']:
        debug_interval = config['simulation']['debug_interval']
        print(f"从配置文件中读取debug_interval: {debug_interval}")
    
    videos_proba = catalog()
    best_allocation =  decide_opt_alloc(videos_proba)
    
    # 初始化多缓存节点的分配
    cache_nodes = topology_manager.get_cache_nodes()
    if cache_nodes:
        # 为每个缓存节点分配初始缓存空间
        initial_allocation = {}
        # 计算每个缓存节点的缓存容量（每个节点使用固定的single_cache_capacity）
        node_cache_capacity = single_cache_capacity
        
        for i, node in enumerate(cache_nodes):
            # 为每个缓存节点创建一个列表，存储每个SP的缓存分配
            # 初始平均分配给每个SP
            node_allocation = [0] * nSP
            # 每个节点的实际缓存容量都是固定的single_cache_capacity
            actual_capacity = node_cache_capacity
            
            # 平均分配给每个SP
            avg_per_sp = actual_capacity // nSP
            rem_per_sp = actual_capacity % nSP
            
            for sp in range(nSP):
                node_allocation[sp] = avg_per_sp + (1 if sp < rem_per_sp else 0)
            
            initial_allocation[node] = node_allocation

        if debug_mode:
            print(f"最佳分配: {best_allocation}")
            print(f"初始分配: {initial_allocation}")

        if should_i_simulate:
            # 从配置中获取参数
            request_rate = config['simulation']['request_rate']
            interval_size = config['simulation']['interval_size']
            delta = config['simulation']['delta']
            if method is None:
                method = config['simulation']['method']
            D = config['simulation']['D']
            
            nb_interval = int(simulation_time / interval_size)
            
            # 检查method是否为列表（对比模式）
            if isinstance(method, list):
                print("使用对比模式，同时运行多种分配策略")
                print(f"要对比的方法: {method}")
                
                # 生成固定的请求序列，确保所有方法使用相同的请求分布
                print("\n生成固定的请求序列...")
                request_nb = int(interval_size * request_rate)
                fixed_requests = []
                
                # 获取SP节点和接收器节点
                sp_nodes = topology_manager.get_sp_nodes()
                receiver_nodes = topology_manager.get_receiver_nodes()
                router_nodes = topology_manager.get_router_nodes()
                cache_nodes = topology_manager.get_cache_nodes()
                
                # 为每个interval生成固定的请求序列
                for interval in range(nb_interval):
                    interval_requests = {
                        'requests': [],
                        'source_nodes': []
                    }
                    
                    # 生成固定的请求序列
                    for r in range(request_nb):
                        request = request_creation(videos_proba)
                        interval_requests['requests'].append(request)
                        
                        # 随机选择一个路由器节点或缓存节点作为请求源
                        possible_sources = []
                        if router_nodes:
                            possible_sources.extend(router_nodes)
                        if cache_nodes:
                            possible_sources.extend(cache_nodes)

                        if possible_sources:
                            source_node = rd.choice(possible_sources)
                        else:
                            source_node = None
                        interval_requests['source_nodes'].append(source_node)
                    
                    fixed_requests.append(interval_requests)
                
                print(f"生成了 {nb_interval} 个interval的固定请求，每个interval包含 {request_nb} 个请求")
                
                # 为静态分配方法计算所有interval的成本（包含运行时间统计）
                static_methods = ['best_allocation', 'cooperative_best_allocation', 'global_opt_allocation', 'equal_allocation', 'manual_allocation']
                static_method_results = {}
                
                # 添加静态方法的运行时间统计
                import time
                
                for compare_method in static_methods:
                    if compare_method in method:
                        print(f"\n{'='*50}")
                        print(f"运行方法: {compare_method}")
                        print(f"{'='*50}")
                        
                        # 记录开始时间
                        start_time = time.time()
                        
                        try:
                            # 检查是否需要输出调试信息
                            debug_output = (debug_interval is not None and debug_interval >= 0)
                            result = optimize_nSP(
                                initial_allocation, videos_proba, best_allocation, 
                                request_rate, nb_interval, interval_size, 0.9, delta, D, compare_method, 
                                fixed_requests=fixed_requests, debug_interval=debug_interval
                            )
                            
                            # 记录结束时间并计算运行时间
                            end_time = time.time()
                            runtime = end_time - start_time
                            
                            # 将运行时间添加到结果中
                            result_with_runtime = result + (runtime,)
                            static_method_results[compare_method] = result_with_runtime
                            
                            print(f"方法 {compare_method} 运行时间: {runtime:.2f} 秒")
                            
                        except Exception as e:
                            import traceback
                            error_traceback = traceback.format_exc()
                            print(f"方法 {compare_method} 执行失败: {e}")
                            print(f"错误详情:")
                            print(error_traceback)
                            continue
                
                # 存储各种方法的结果
                compare_results = static_method_results.copy()
                
                # 处理动态分配方法（如SCA_ADMM, SCA_neighborhood_search, proportional_allocation, Q_learning）
                dynamic_methods = ['SCA_ADMM', 'SCA_neighborhood_search', 'proportional_allocation', 'Q_learning']
                
                # 添加运行时间统计
                import time
                method_runtimes = {}
                
                for compare_method in dynamic_methods:
                    if compare_method in method:
                        print(f"\n{'='*50}")
                        print(f"运行方法: {compare_method}")
                        print(f"{'='*50}")
                        
                        # 记录开始时间
                        start_time = time.time()
                        
                        try:
                            # 检查是否需要输出调试信息
                            debug_output = (debug_interval is not None and debug_interval >= 0)
                            result = optimize_nSP(
                                initial_allocation, videos_proba, best_allocation, 
                                request_rate, nb_interval, interval_size, 0.9, delta, D, compare_method, 
                                fixed_requests=fixed_requests, debug_interval=debug_interval
                            )
                            
                            # 记录结束时间并计算运行时间
                            end_time = time.time()
                            runtime = end_time - start_time
                            method_runtimes[compare_method] = runtime
                            
                            # 将运行时间添加到结果中（作为第7个元素）
                            result_with_runtime = result + (runtime,)
                            compare_results[compare_method] = result_with_runtime
                            
                            print(f"方法 {compare_method} 运行时间: {runtime:.2f} 秒")
                            
                        except Exception as e:
                            import traceback
                            error_traceback = traceback.format_exc()
                            print(f"方法 {compare_method} 执行失败: {e}")
                            print(f"错误详情:")
                            print(error_traceback)
                            continue
                
                # 绘制对比图
                print(f"\n{'='*50}")
                print("生成对比图表")
                print(f"{'='*50}")
                
                figures_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')
                os.makedirs(figures_dir, exist_ok=True)
                
                # 创建对比图
                import matplotlib.pyplot as plt
                plt.figure(figsize=(16, 12))
                
                # 命中率对比图
                plt.subplot(2, 2, 1)
                for method_name, result in compare_results.items():
                    if len(result) >= 3:
                        nominal_costs = result[2]  # L_nominal_cost
                        hit_rates = [1 - cost for cost in nominal_costs]
                        plt.plot(hit_rates, label=method_name)
                plt.xlabel('Time (intervals)')
                plt.ylabel('Cache Hit Rate')
                plt.title('Cache Hit Rate Comparison')
                plt.legend()
                plt.grid(True)
                
                # 时延对比图
                plt.subplot(2, 2, 2)
                for method_name, result in compare_results.items():
                    if len(result) >= 6:
                        latencies = result[5]  # L_avg_latency
                        plt.plot(latencies, label=method_name)
                plt.xlabel('Time (intervals)')
                plt.ylabel('Average Latency (ms)')
                plt.title('Average Latency Comparison')
                plt.legend()
                plt.grid(True)
                
                # 成本对比图
                plt.subplot(2, 2, 3)
                for method_name, result in compare_results.items():
                    if len(result) >= 2:
                        total_costs = result[1]  # L_total_cost
                        plt.plot(total_costs, label=method_name)
                plt.xlabel('Time (intervals)')
                plt.ylabel('Total Cost')
                plt.title('Total Cost Comparison')
                plt.legend()
                plt.grid(True)
                
                # 最终结果对比条形图
                plt.subplot(2, 2, 4)
                final_results = []
                method_names = []
                for method_name, result in compare_results.items():
                    if len(result) >= 3:
                        final_hit_rate = 1 - sum(result[1]) / len(result[1])  # 整个实验期间的平均命中率
                        final_results.append(final_hit_rate)
                        method_names.append(method_name)
                
                x_pos = range(len(method_names))
                plt.bar(x_pos, final_results)
                plt.xticks(x_pos, method_names, rotation=45)
                plt.ylabel('Final Hit Rate')
                plt.title('Final Hit Rate Comparison')
                
                plt.tight_layout()
                compare_fig_path = os.path.join(figures_dir, f'compare_methods_{request_rate}.png')
                plt.savefig(compare_fig_path)
                plt.close()
                print(f"对比图表已保存: {compare_fig_path}")
                
                # 保存对比结果到CSV
                results_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', f'compare_results_{request_rate}.csv')
                os.makedirs(os.path.dirname(results_file), exist_ok=True)
                
                with open(results_file, 'w') as f:
                    f.write('Method,Interval,Nominal_Cost,Hit_Rate,Avg_Latency\n')
                    for method_name, result in compare_results.items():
                        if len(result) >= 5:
                            for i in range(len(result[1])):
                                hit_rate = 1 - result[1][i]
                                # 检查是否为返回6个值的方法
                                if method_name in ['best_allocation', 'equal_allocation', 'cooperative_best_allocation', 
                          'proportional_allocation', 'SCA_ADMM', 'global_opt_allocation', 
                          'SCA_gradient_tracking', 'manual_allocation', 
                          'SCA_neighborhood_search', 'Q_learning']:
                                    # 对于这两种方法，返回值格式是：(allocation, L_total_cost, L_nominal_cost, L_first_cost, L_best_cost, L_avg_latency)
                                    avg_lat = result[5][i]  # L_avg_latency
                                else:
                                    # 对于其他方法，返回值格式是：(total_cost, nominal_cost, cost_first, cost_best, avg_latency)
                                    avg_lat = result[4][i]  # avg_latency
                                f.write(f'{method_name},{i},{result[1][i]},{hit_rate},{avg_lat}\n')
                
                print(f"对比结果已保存: {results_file}")
                
                # 打印对比摘要（包含运行时间）
                print(f"\n{'='*70}")
                print("对比结果摘要（包含运行时间）")
                print(f"{'='*70}")
                print(f"{'方法':<20} {'最终命中率':<12} {'平均时延':<12} {'最终成本':<12} {'运行时间(秒)':<15}")
                print("-" * 75)
                for method_name, result in compare_results.items():
                    if len(result) >= 5:
                        # 检查是否为返回6个值的方法
                        if method_name in ['best_allocation', 'equal_allocation', 'cooperative_best_allocation', 
                                          'proportional_allocation', 'SCA_ADMM', 'global_opt_allocation', 
                                          'SCA_neighborhood_search', 'manual_allocation', 'Q_learning']:
                            # 对于这些方法，返回值格式是：(allocation, L_total_cost, L_nominal_cost, L_first_cost, L_best_cost, L_avg_latency, runtime)
                            if len(result) >= 6 and len(result[5]) > 0:
                                final_hit_rate = 1 - sum(result[1]) / len(result[1])  # 整个实验期间的平均命中率
                                avg_latency = sum(result[5]) / len(result[5])  # L_avg_latency
                                final_cost = sum(result[1]) / len(result[1])  # 整个实验期间的平均成本
                                runtime = result[6] if len(result) >= 7 else 0.0  # 运行时间
                            else:
                                # 如果时延数据不可用，使用默认值
                                final_hit_rate = 1 - sum(result[1]) / len(result[1])
                                avg_latency = 0.0
                                final_cost = sum(result[1]) / len(result[1])
                                runtime = result[6] if len(result) >= 7 else 0.0  # 运行时间
                        else:
                            # 对于其他方法，返回值格式是：(total_cost, nominal_cost, cost_first, cost_best, avg_latency, runtime)
                            if len(result) >= 5 and len(result[4]) > 0:
                                final_hit_rate = 1 - sum(result[1]) / len(result[1])  # 整个实验期间的平均命中率
                                avg_latency = sum(result[4]) / len(result[4])  # avg_latency
                                final_cost = sum(result[1]) / len(result[1])  # 整个实验期间的平均成本
                                runtime = result[5] if len(result) >= 6 else 0.0  # 运行时间
                            else:
                                # 如果时延数据不可用，使用默认值
                                final_hit_rate = 1 - sum(result[1]) / len(result[1])
                                avg_latency = 0.0
                                final_cost = sum(result[1]) / len(result[1])
                                runtime = result[5] if len(result) >= 6 else 0.0  # 运行时间
                        print(f"{method_name:<20} {final_hit_rate:<12.4f} {avg_latency:<12.4f} {final_cost:<12.4f} {runtime:<15.2f}")
                
                # 为每个方法生成拓扑图
                print(f"\n{'='*50}")
                print("Generating Topology")
                print(f"{'='*50}")
                
                for method_name, result in compare_results.items():
                    print(f"Generating topology for {method_name}...")
                    # 检查是否为返回6个值的方法
                    if method_name in ['best_allocation', 'equal_allocation', 'cooperative_best_allocation', 
                                      'proportional_allocation', 'SCA_ADMM', 'global_opt_allocation', 
                                      'SCA_gradient_tracking', 'manual_allocation']:
                        # 对于这些方法，返回值格式包含分配信息
                        if len(result) > 0:
                            # 生成拓扑图
                            if topology_manager:
                                # 禁用中文字符警告
                                import warnings
                                warnings.filterwarnings("ignore", category=UserWarning)
                                
                                plt.figure(figsize=(10, 8))
                                topology_manager.visualize(plt.gcf(), show=False)
                                topology_plot_path = os.path.join(figures_dir, f'topology_{topology_type}_{method_name}.png')
                                plt.savefig(topology_plot_path)
                                plt.close()
                                print(f"Topology saved: {topology_plot_path}")
                
                # 选择第一个方法的结果作为默认返回
                if compare_results:
                    first_method = list(compare_results.keys())[0]
                    first_result = compare_results[first_method]
                    if first_method == 'best_allocation' or first_method == 'equal_allocation' or first_method == 'cooperative_best_allocation' or first_method == 'proportional_allocation' or first_method == 'SCA_ADMM' or first_method == 'global_opt_allocation' or first_method == 'SCA_gradient_tracking':
                        total_cost = first_result[1]
                        nominal_cost = first_result[2]
                        cost_first = first_result[3]
                        cost_best = first_result[4]
                        avg_latency = first_result[5]
                    else:
                        # 对于其他方法，尝试解包5个值，如果失败则使用默认值
                        try:
                            total_cost, nominal_cost, cost_first, cost_best, avg_latency = first_result
                        except ValueError:
                            # 如果解包失败，使用默认值
                            total_cost = [1.0] * nb_interval
                            nominal_cost = [1.0] * nb_interval
                            cost_first = [1.0] * nb_interval
                            cost_best = [1.0] * nb_interval
                            avg_latency = [0.0] * nb_interval
                else:
                    # 如果所有方法都失败，使用默认值
                    total_cost = [1.0] * nb_interval
                    nominal_cost = [1.0] * nb_interval
                    cost_first = [1.0] * nb_interval
                    cost_best = [1.0] * nb_interval
                    avg_latency = [0.0] * nb_interval
            else:
                # 单方法模式
                # 运行优化
                result = optimize_nSP(
                    initial_allocation, videos_proba, best_allocation, 
                    request_rate, nb_interval, interval_size, gamma, delta, D, method
                )
                
                if method == 'best_allocation' or method == 'equal_allocation' or method == 'cooperative_best_allocation' or method == 'proportional_allocation' or method == 'SCA_ADMM' or method == 'global_opt_allocation':
                    # 这些方法返回的第一个值是分配，后面是成本列表
                    best_alloc_result = result[0]
                    total_cost = result[1]
                    nominal_cost = result[2]
                    cost_first = result[3]
                    cost_best = result[4]
                    avg_latency = result[5]
                else:
                    # 其他方法返回的是成本列表
                    total_cost, nominal_cost, cost_first, cost_best, avg_latency = result
            
            # 保存结果
            if save_results:
                filename = os.path.join(results_dir, f"results_single_cache_capacity{single_cache_capacity}_request_rate{request_rate}.csv")
                with open(filename, "w") as f:
                    headers = "Time_seconds,Time_hours,Total_Cost,Nominal_Cost,Cost_First,Best_Cost"
                    if network_enabled:
                        headers += ",Avg_Latency"
                    f.write(headers + "\n")
                    
                    for i in range(len(total_cost)):
                        row = f"{i*interval_size},{i*interval_size/3600},{total_cost[i]},{nominal_cost[i]},{cost_first[i]},{cost_best[i]}"
                        if network_enabled:
                            row += f",{avg_latency[i]}"
                        f.write(row + "\n")

                if debug_mode:
                    print(f"结果已保存到: {filename}")

            # 绘制结果
            plot_results(total_cost, nominal_cost, cost_first, cost_best, avg_latency, interval_size, request_rate)
            if debug_mode:
                print("图表已生成")
    else:
        # 兼容旧格式
        prop_allocation = [int(single_cache_capacity * SP_proba[i]) for i in range(nSP)]
        
        print(f"最佳分配: {best_allocation}")
        print(f"最佳成本: {compute_estimated_cost(SP_proba, best_allocation, videos_proba)}")
        print(f"比例分配: {prop_allocation}")
        print(f"比例分配成本: {compute_estimated_cost(SP_proba, prop_allocation, videos_proba)}")
    
    # 如果是对比模式，返回所有方法的结果
    if isinstance(method, list):
        return compare_results
    else:
        return total_cost, nominal_cost, cost_first, cost_best, avg_latency

def get_manual_allocation(nSP, cache_nodes, config):
    """
    从配置文件获取手动输入的缓存分配
    """
    print(f"\n=== 手动分配配置 ===")
    print(f"SP数量: {nSP}")
    print(f"缓存节点: {cache_nodes}")
    
    # 从配置文件读取分配设置
    if 'manual_allocation' in config and config['manual_allocation'].get('enabled', False):
        allocation_list = config['manual_allocation'].get('allocation', [10, 10, 10])
        print(f"从配置文件读取分配: {allocation_list}")
    else:
        # 如果配置中未启用手动分配，使用默认值
        allocation_list = [10, 10, 10]
        print(f"手动分配未启用，使用默认分配: {allocation_list}")
    
    # 验证分配列表长度
    if len(allocation_list) != nSP:
        print(f"警告: 配置中的分配列表长度应为{nSP}，但得到{len(allocation_list)}")
        print(f"将使用默认分配: {[10] * nSP}")
        allocation_list = [10] * nSP
    
    # 验证非负值
    if any(x < 0 for x in allocation_list):
        print(f"警告: 配置中的分配值不能为负数，将使用默认分配: {[10] * nSP}")
        allocation_list = [10] * nSP
    
    # 创建分配字典
    manual_allocation = {}
    for node in cache_nodes:
        manual_allocation[node] = allocation_list.copy()
    
    print(f"手动分配设置成功: {allocation_list}")
    print(f"每个节点的分配: {manual_allocation}")
    return manual_allocation

if __name__ == "__main__":
    main()