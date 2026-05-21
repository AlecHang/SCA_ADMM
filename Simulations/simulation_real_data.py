#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实数据缓存分配模拟代码
使用真实请求数据进行缓存分配实验
与simulation_code.py保持一致的接口
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
import json
from collections import Counter

from topology_manager import TopologyManager

custom_config_path = None

try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("PyTorch未安装，DQN方法将不可用。")


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


class DQNNetwork(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DQNNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc4 = torch.nn.Linear(hidden_dim // 2, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


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
cacheable_content = None
fixed_seed = None

node_request_history = {}
cache_node_neighbors = {}

network_enabled = None
nodes = 0
bandwidth = 0

topology_type = None
topology_params = None
cache_nodes_count = None
sp_nodes_count = None
router_nodes_count = None
topology_manager = None

save_allocations = True
save_results = True
results_dir = "results"
figures_dir = "figures"
cooperative_caching = False

min_delta = 10
delta_decay = 0.99

user_to_cache_latency = 5
cache_to_cache_latency = 10
cache_to_sp_latency = 30

real_data_config = None
real_data_rankings = None
real_data_stats = None
real_data_requests = None
sp_names = ['youtube', 'netflix', 'douyin']





def load_config():
    """从配置文件加载参数"""
    global config, list_alpha, SP_proba, video_nb_list, conss_zipf, nSP
    global single_cache_capacity, nb_videos, gamma, epsilon, alpha_de_sarsa
    global simulation_time, cacheability, real_data_config
    global topology_type, topology_params, cache_nodes_count, sp_nodes_count, router_nodes_count
    global save_allocations, save_results, results_dir, figures_dir, fixed_seed
    global min_delta, delta_decay
    global cooperative_caching, user_to_cache_latency, cache_to_cache_latency, cache_to_sp_latency
    global sp_names

    global custom_config_path
    if custom_config_path is not None and os.path.exists(custom_config_path):
        config_path = custom_config_path
    else:
        config_path = os.path.join(os.path.dirname(__file__), 'config_real_data.yml')

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    sim_config = config['simulation']
    simulation_time = sim_config['time']
    single_cache_capacity = sim_config['single_cache_capacity']
    nb_videos = sim_config['nb_videos']
    min_delta = sim_config.get('min_delta', 10)
    delta_decay = sim_config.get('delta_decay', 0.99)

    prov_config = config['providers']
    nSP = prov_config['count']
    SP_proba = prov_config['probabilities']
    cacheability = prov_config['cacheability']
    list_alpha = prov_config['zipf_alphas']

    rl_config = config['rl']
    gamma = rl_config['gamma']
    epsilon = rl_config['epsilon']
    fixed_seed = rl_config.get('fixed_seed', True)

    topo_config = config['topology']
    topology_type = topo_config['type']
    topology_params = topo_config['parameters']
    cache_nodes_count = topo_config.get('cache_nodes_ratio', topo_config.get('cache_nodes', 10))
    sp_nodes_count = nSP
    router_nodes_count = topo_config.get('router_nodes', None)

    out_config = config['output']
    save_allocations = out_config['save_allocations']
    save_results = out_config['save_results']
    results_dir = out_config['results_dir']
    figures_dir = out_config['figures_dir']
    global network_enabled
    network_enabled = out_config.get('network_enabled', False)

    global cooperative_caching, user_to_cache_latency, cache_to_cache_latency, cache_to_sp_latency
    coop_config = config.get('cooperative_caching', {})
    cooperative_caching = coop_config.get('enabled', False)
    user_to_cache_latency = coop_config.get('user_to_cache_latency', 5)
    cache_to_cache_latency = coop_config.get('cache_to_cache_latency', 10)
    cache_to_sp_latency = coop_config.get('cache_to_sp_latency', 30)

    real_data_config = config.get('real_data', {})
    sp_names = real_data_config.get('sp_names', ['youtube', 'netflix', 'douyin'])

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    video_nb_list = [nb_videos for i in range(nSP)]
    conss_zipf = [af.zipf_norm(list_alpha[i], video_nb_list[i]) for i in range(nSP)]

    print("配置加载完成:")
    print(f"  模拟时间: {simulation_time}秒")
    print(f"  单个缓存节点容量: {single_cache_capacity}")
    print(f"  服务提供商数量: {nSP}")
    print(f"  拓扑类型: {topology_type}")
    print(f"  真实数据模式: {real_data_config.get('enabled', False)}")


def init(seed=None):
    """初始化全局变量
    
    参数:
        seed: 可选的随机种子。如果提供，将使用该种子初始化随机数生成器，
              确保拓扑创建的可重复性。如果为None，使用配置文件中的设置。
    """
    load_config()

    if not load_real_data():
        print("警告: 真实数据加载失败，使用模拟数据")

    # 设置随机种子
    if seed is not None:
        # 使用外部提供的种子（用于重复实验）
        rd.seed(seed)
        print(f"随机种子设置: 外部指定 ({seed})")
    elif fixed_seed:
        rd.seed(3231)
        print(f"随机种子设置: 固定 (3231)")
    else:
        rd.seed()
        print(f"随机种子设置: 不固定 (系统时间)")

    global topology_manager
    topology_manager = TopologyManager(
        topology_type,
        topology_params,
        cache_nodes_count,
        sp_nodes_count,
        router_nodes_count
    )
    topology_manager.create_topology()
    print(f"拓扑创建完成，节点总数: {len(topology_manager.graph.nodes())}")
    print(f"缓存节点: {topology_manager.get_cache_nodes()}")
    print(f"SP节点: {topology_manager.get_sp_nodes()}")
    print(f"路由器节点: {topology_manager.get_router_nodes()}")
    print(f"接收器节点: {topology_manager.get_receiver_nodes()}")

    print("\n=== 缓存节点邻居列表 ===")
    cache_nodes = topology_manager.get_cache_nodes()
    for node in cache_nodes:
        all_neighbors = topology_manager.get_neighbors(node)
        neighbor_cache_nodes = [n for n in all_neighbors if n in cache_nodes]
        print(f"节点 {node} 的邻居缓存节点: {neighbor_cache_nodes}")

    print("\n=== 初始化视频目录 ===")
    initial_videos_proba = catalog()
    print(f"视频目录初始化完成，可缓存内容长度: {len(cacheable_content) if cacheable_content else 0}")
    for SP in range(nSP):
        print(f"SP {SP} 可缓存内容数量: {len(cacheable_content[SP]) if cacheable_content and SP < len(cacheable_content) else 0}")


def zipf_distribution(alpha, nb_videos, norm):
    """创建遵循Zipf定律的视频请求概率分布"""
    probabilites_pi = np.zeros(nb_videos)
    for i in range(1, nb_videos+1):
        pi = (1.0/i**alpha) * (1.0/norm)
        probabilites_pi[i-1] = pi
    return probabilites_pi


def estimate_videos_proba_from_history():
    """从节点请求历史估计视频流行度分布（在线学习）"""
    global node_request_history, cacheable_content, nSP, nb_videos

    videos = np.zeros((nSP, nb_videos))

    for sp in range(nSP):
        sp_video_counts = {}

        for node, node_history in node_request_history.items():
            if sp in node_history:
                for video, count in node_history[sp].items():
                    if video not in sp_video_counts:
                        sp_video_counts[video] = 0
                    sp_video_counts[video] += count

        total_count = sum(sp_video_counts.values())
        if total_count > 0:
            for video, count in sp_video_counts.items():
                try:
                    video_idx = int(video) if isinstance(video, str) else video
                    if 0 <= video_idx < nb_videos:
                        videos[sp][video_idx] = count / total_count
                except (ValueError, TypeError):
                    pass
        else:
            for video in range(nb_videos):
                videos[sp][video] = 1.0 / nb_videos

    return videos


def catalog(use_online_estimation=False):
    """创建视频目录 - 支持真实流行度数据或在线估计"""
    global cacheable_content, videos_proba, real_data_rankings, real_data_stats

    if use_online_estimation:
        return estimate_videos_proba_from_history()

    if cacheable_content is not None:
        return videos_proba

    videos = np.zeros((nSP, nb_videos))
    cacheable_content = []

    for SP in range(nSP):
        sp_name = sp_names[SP]

        cacheable = []
        if real_data_rankings and sp_name in real_data_rankings:
            rankings = real_data_rankings[sp_name]
            
            # 从真实排名数据构建概率分布（基于请求计数）
            total_requests = real_data_stats.get(sp_name, {}).get('total_requests', 1)
            for rank in range(min(nb_videos, len(rankings))):
                content_id, req_count = rankings[rank]
                cacheable.append(content_id)
                # 基于真实请求计数计算概率
                videos[SP][rank] = req_count / total_requests if total_requests > 0 else 1.0 / (rank + 1)
        else:
            # 回退到 zipf 分布
            videos[SP] = zipf_distribution(list_alpha[SP], video_nb_list[SP], conss_zipf[SP])
            for video_id in range(nb_videos):
                if rd.random() <= cacheability[SP]:
                    cacheable.append(video_id)
        cacheable_content.append(cacheable)

    videos_proba = videos
    return videos


def record_request(node, sp, video):
    """记录节点的请求历史"""
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
    与Zipf版本保持一致
    """
    global node_request_history

    if node not in node_request_history or sp not in node_request_history[node]:
        return 1.0 / nb_videos

    history = node_request_history[node][sp]
    total_requests = sum(history.values())
    if total_requests == 0:
        return 1.0 / nb_videos

    short_term_prob = history.get(video, 0) / min(total_requests, window_size)

    long_term_prob = short_term_prob

    node_local_states = None
    try:
        node_local_states = globals().get('node_local_states')
        if node_local_states and node in node_local_states:
            node_history = node_local_states[node]['request_history']
            if 'video_requests' in node_history and sp in node_history['video_requests'] and video in node_history['video_requests'][sp]:
                historical_counts = node_history['video_requests'][sp].get(video, 0)
                historical_total = sum(node_history['video_requests'][sp].values())
                if historical_total > 0:
                    long_term_prob = historical_counts / historical_total
    except (NameError, KeyError):
        pass

    alpha = 0.7
    combined_prob = alpha * short_term_prob + (1 - alpha) * long_term_prob

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

    if node in node_request_history:
        sp_request_counts = {}
        total_requests = 0
        for sp_idx in range(nSP):
            if sp_idx in node_request_history[node]:
                count = sum(node_request_history[node][sp_idx].values())
                sp_request_counts[sp_idx] = count
                total_requests += count

        if total_requests > 0:
            current_proba = sp_request_counts.get(sp, 0) / min(total_requests, window_size)

            long_term_proba = current_proba

            node_local_states = None
            try:
                node_local_states = globals().get('node_local_states')
                if node_local_states and node in node_local_states:
                    history = node_local_states[node]['request_history']
                    if 'sp_requests' in history and sp in history['sp_requests']:
                        historical_counts = history['sp_requests'].get(sp, 0)
                        historical_total = sum(history['sp_requests'].values())
                        if historical_total > 0:
                            long_term_proba = historical_counts / historical_total
            except (NameError, KeyError):
                pass

            alpha = 0.7
            combined_proba = alpha * current_proba + (1 - alpha) * long_term_proba

            if 'previous_sp_proba' in globals() and node in previous_sp_proba and sp in previous_sp_proba[node]:
                combined_proba = smoothing * combined_proba + (1 - smoothing) * previous_sp_proba[node][sp]

            if 'previous_sp_proba' in globals():
                if node not in previous_sp_proba:
                    previous_sp_proba[node] = {}
                previous_sp_proba[node][sp] = combined_proba

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

    sp_importance = 1.0 / (sp + 1)
    total_importance = sum(1.0 / (s + 1) for s in range(nSP))
    return sp_importance / total_importance


def load_real_data():
    """加载真实请求数据 - 使用分析脚本处理过的数据"""
    global real_data_config, real_data_rankings, real_data_stats, real_data_requests

    real_data_dir = os.path.join(os.path.dirname(__file__), 'real_data')

    # 加载流行度排名、统计数据和实际请求数据
    rankings_file = os.path.join(real_data_dir, 'sp_popularity_rankings.json')
    stats_file = os.path.join(real_data_dir, 'sp_statistics.json')
    requests_file = os.path.join(real_data_dir, 'sp_requests.json')

    if os.path.exists(rankings_file) and os.path.exists(stats_file) and os.path.exists(requests_file):
        with open(rankings_file, 'r', encoding='utf-8') as f:
            real_data_rankings = json.load(f)
        with open(stats_file, 'r', encoding='utf-8') as f:
            real_data_stats = json.load(f)
        with open(requests_file, 'r', encoding='utf-8') as f:
            real_data_requests = json.load(f)
    else:
        print("真实数据文件不存在，请先运行 analyze_real_data.py")
        return False

    # 从统计文件读取实际使用的筛选比例
    first_sp = next(iter(real_data_stats.values()), None)
    if first_sp and 'ratio_used' in first_sp:
        data_ratio = first_sp['ratio_used']
    else:
        data_ratio = real_data_config.get('data_ratio', 0.01)
    print(f"\n数据筛选比例: {data_ratio*100:.2f}%")

    # 显示加载的数据统计
    total_filtered = sum(stats['total_requests'] for stats in real_data_stats.values())
    print(f"\n已加载筛选后总请求数: {total_filtered}")
    for sp_name, stats in real_data_stats.items():
        print(f"  {sp_name}: {stats['total_requests']} 请求, {len(real_data_requests[sp_name])} 条记录")

    return True


def calculate_real_sp_proba(sp_names_list):
    """根据真实数据计算实际的SP分布"""
    global real_data_requests

    sp_counts = {sp_name: 0 for sp_name in sp_names_list}

    for sp_name, requests in real_data_requests.items():
        if sp_name in sp_counts:
            sp_counts[sp_name] = len(requests)

    total_requests = sum(sp_counts.values())

    if total_requests == 0:
        return [1.0 / len(sp_names_list)] * len(sp_names_list)

    real_sp_proba = [sp_counts[sp_name] / total_requests for sp_name in sp_names_list]

    return real_sp_proba


def create_real_data_requests(sp_names_list, requests_per_interval, nb_intervals):
    """根据真实数据创建请求序列 - 与Zipf版本保持一致使用SP_proba"""
    fixed_requests = []

    for interval in range(nb_intervals):
        interval_requests = []
        interval_source_nodes = []

        cache_nodes_list = topology_manager.get_cache_nodes()
        router_nodes_list = topology_manager.get_router_nodes()

        for r in range(requests_per_interval):
            sp_idx = rd.choices(range(nSP), weights=SP_proba, k=1)[0]
            sp_name = sp_names_list[sp_idx]

            if sp_name in real_data_requests and real_data_requests[sp_name]:
                content_id = rd.choice(real_data_requests[sp_name])
                interval_requests.append((sp_idx, content_id))
            else:
                video_choice = rd.random()
                S2 = 0
                selected_video = 0
                max_video = len(videos_proba[sp_idx]) - 1
                while selected_video < max_video and video_choice > S2:
                    S2 += videos_proba[sp_idx][selected_video]
                    selected_video += 1
                # 确保不越界
                selected_video = min(selected_video, max_video)
                interval_requests.append((sp_idx, selected_video))

            possible_sources = []
            if router_nodes_list:
                possible_sources.extend(router_nodes_list)
            if cache_nodes_list:
                possible_sources.extend(cache_nodes_list)

            if possible_sources:
                source_node = rd.choice(possible_sources)
            else:
                source_node = None
            interval_source_nodes.append(source_node)

        fixed_requests.append({
            'requests': interval_requests,
            'source_nodes': interval_source_nodes
        })

    return fixed_requests


def request_creation(video_probability):
    """创建模拟请求"""
    S = SP_proba[0]
    SP_choice = rd.random()
    selected_SP = 0
    max_SP = len(SP_proba) - 1

    while selected_SP < max_SP and SP_choice > S:
        selected_SP += 1
        S += SP_proba[selected_SP]
    # 确保不越界
    selected_SP = min(selected_SP, max_SP)

    video_choice = rd.random()
    S2 = 0
    selected_video = 0
    max_video = len(video_probability[selected_SP]) - 1
    while selected_video < max_video and video_choice > S2:
        S2 += video_probability[selected_SP][selected_video]
        selected_video += 1
    # 确保不越界
    selected_video = min(selected_video, max_video)

    return [selected_SP, selected_video]


def calculate_latency(source_node, dest_node):
    """计算节点间的时延"""
    if not network_enabled:
        return 0

    if topology_manager:
        path_length = topology_manager.get_path_length(source_node, dest_node)
        if path_length != float('inf'):
            base_latency = 1
            per_hop_latency = 5
            return base_latency + path_length * per_hop_latency

    return 0


def decide_opt_alloc(distrib):
    """计算最优缓存分配 - 支持多个缓存节点"""
    cache_nodes = topology_manager.get_cache_nodes()
    num_cache_nodes = len(cache_nodes)

    if num_cache_nodes == 0:
        return {}

    allocation = {}
    node_cache_capacity = single_cache_capacity

    for i, node in enumerate(cache_nodes):
        node_allocation = [0] * nSP
        actual_capacity = node_cache_capacity

        pointer_vec = [0] * nSP

        for slot in range(actual_capacity):
            bestSP = 0
            best_score = -1

            for currentSP in range(nSP):
                if cacheable_content and currentSP < len(cacheable_content):
                    cacheable_videos = cacheable_content[currentSP]
                    if pointer_vec[currentSP] < len(cacheable_videos):
                        video_idx = pointer_vec[currentSP]
                        score = distrib[currentSP][video_idx] * SP_proba[currentSP]
                        if score > best_score:
                            best_score = score
                            bestSP = currentSP

            node_allocation[bestSP] += 1
            pointer_vec[bestSP] += 1

        allocation[node] = node_allocation

    return allocation


def decide_cooperative_opt_alloc(distrib):
    """计算考虑邻居协作的最优缓存分配"""
    cache_nodes = topology_manager.get_cache_nodes()
    num_cache_nodes = len(cache_nodes)

    if num_cache_nodes == 0:
        return {}

    cache_node_neighbors = {}
    for node in cache_nodes:
        all_neighbors = topology_manager.get_neighbors(node)
        neighbors = [n for n in all_neighbors if n in cache_nodes]
        cache_node_neighbors[node] = neighbors

    allocation = {}
    node_cache_capacity = single_cache_capacity

    for i, node in enumerate(cache_nodes):
        allocation[node] = [0] * nSP
        actual_capacity = node_cache_capacity
        pointer_vec = [0] * nSP

        for slot in range(actual_capacity):
            bestSP = 0
            best_score = -1
            best_video_idx = -1

            for currentSP in range(nSP):
                if cacheable_content and currentSP < len(cacheable_content):
                    cacheable_videos = cacheable_content[currentSP]
                    if pointer_vec[currentSP] < len(cacheable_videos):
                        video_rank = pointer_vec[currentSP]
                        video_id = cacheable_videos[video_rank]

                        neighbor_has_video = False
                        for neighbor in cache_node_neighbors[node]:
                            if neighbor in allocation:
                                if allocation[neighbor][currentSP] > video_rank:
                                    neighbor_has_video = True
                                    break

                        base_score = distrib[currentSP][video_rank] * SP_proba[currentSP]

                        if neighbor_has_video:
                            score = base_score * 0.5
                        else:
                            score = base_score

                        if score > best_score:
                            best_score = score
                            bestSP = currentSP
                            best_video_idx = video_rank

            if bestSP is not None and best_video_idx != -1:
                allocation[node][bestSP] += 1
                pointer_vec[bestSP] += 1

    return allocation


def decide_global_opt_alloc(distrib, debug=False):
    """计算全局最优缓存分配"""
    import time

    start_time = time.time()

    if debug:
        print(f"[DEBUG] decide_global_opt_alloc 开始...")

    cache_nodes = topology_manager.get_cache_nodes()
    num_cache_nodes = len(cache_nodes)

    if num_cache_nodes == 0:
        return {}

    if debug:
        print(f"[DEBUG] 缓存节点数: {num_cache_nodes}")

    cache_node_neighbors = {}
    for node in cache_nodes:
        all_neighbors = topology_manager.get_neighbors(node)
        neighbors = [n for n in all_neighbors if n in cache_nodes]
        cache_node_neighbors[node] = neighbors

    node_cache_capacity = single_cache_capacity

    allocation = {}
    node_capacities = {}
    for i, node in enumerate(cache_nodes):
        actual_capacity = node_cache_capacity
        node_capacities[node] = actual_capacity
        allocation[node] = [0] * nSP

    node_importance = {}
    node_centrality = {}
    for node in cache_nodes:
        degree = len(cache_node_neighbors[node])
        centrality = 0
        for neighbor in cache_node_neighbors[node]:
            centrality += len(cache_node_neighbors.get(neighbor, []))
        node_centrality[node] = centrality
        node_importance[node] = degree + 0.5 * centrality

    max_importance = max(node_importance.values()) if node_importance else 1
    if max_importance == 0:
        max_importance = 1
    for node in node_importance:
        node_importance[node] /= max_importance

    network_size_factor = min(1.0, max(0.7, num_cache_nodes / 15.0))

    if debug:
        print(f"[DEBUG] 网络规模因子: {network_size_factor:.3f}")
        step1_time = time.time() - start_time
        print(f"[DEBUG] 步骤1（初始化）完成: {step1_time:.2f}秒")

    video_to_index = {}
    for sp in range(nSP):
        if cacheable_content and sp < len(cacheable_content):
            cacheable_videos = cacheable_content[sp]
            video_to_index[sp] = {video: idx for idx, video in enumerate(cacheable_videos)}

    video_global_importance = {}
    if debug:
        print(f"[DEBUG] 开始计算视频全局重要性...")
    for sp in range(nSP):
        if cacheable_content and sp < len(cacheable_content):
            cacheable_videos = cacheable_content[sp]
            max_videos = min(1000, len(cacheable_videos))
            for video_rank, video_id in enumerate(cacheable_videos[:max_videos]):
                global_importance = 0
                for other_node in cache_nodes:
                    other_prob = distrib[sp][video_rank] if sp < len(distrib) and video_rank < len(distrib[sp]) else 0
                    importance_weight = node_importance.get(other_node, 1)
                    max_centrality = max(node_centrality.values()) if node_centrality else 1
                    if max_centrality == 0:
                        max_centrality = 1
                    centrality_weight = node_centrality.get(other_node, 0) / max_centrality
                    global_importance += other_prob * SP_proba[sp] * (0.6 * importance_weight + 0.4 * centrality_weight) * (1 + 0.3 * (1 - network_size_factor))
                video_global_importance[(sp, video_id)] = global_importance
    if debug:
        step2_time = time.time() - start_time
        print(f"[DEBUG] 步骤2（视频全局重要性）完成: {step2_time:.2f}秒, 视频数: {len(video_global_importance)}")

    def calculate_global_cache_distribution(allocation):
        video_cache_frequency = {}
        video_cache_coverage = {}
        for node in cache_nodes:
            for sp in range(nSP):
                if cacheable_content and sp < len(cacheable_content):
                    cacheable_videos = cacheable_content[sp]
                    for video_rank in range(min(allocation[node][sp], 500)):
                        video = cacheable_videos[video_rank]
                        key = (sp, video)
                        video_cache_frequency[key] = video_cache_frequency.get(key, 0) + 1

        for sp in range(nSP):
            if cacheable_content and sp < len(cacheable_content):
                cacheable_videos = cacheable_content[sp]
                max_videos = min(1000, len(cacheable_videos))
                for video_rank in range(min(max_videos, len(cacheable_videos))):
                    video = cacheable_videos[video_rank]
                    coverage = 0
                    for node in cache_nodes:
                        if cacheable_content and sp < len(cacheable_content):
                            if video_rank < allocation[node][sp]:
                                node_prob = distrib[sp][video_rank] if sp < len(distrib) and video_rank < len(distrib[sp]) else 0
                                importance_weight = node_importance.get(node, 1)
                                max_centrality = max(node_centrality.values()) if node_centrality else 1
                                if max_centrality == 0:
                                    max_centrality = 1
                                centrality_weight = node_centrality.get(node, 0) / max_centrality
                                coverage += node_prob * (0.6 * importance_weight + 0.4 * centrality_weight) * (1 + 0.3 * (1 - network_size_factor))
                    video_cache_coverage[(sp, video)] = coverage

        return video_cache_frequency, video_cache_coverage

    sp_global_importance = {}
    for sp in range(nSP):
        sp_importance = 0
        if cacheable_content and sp < len(cacheable_content):
            cacheable_videos = cacheable_content[sp]
            max_videos = min(200, len(cacheable_videos))
            for video_idx in cacheable_videos[:max_videos]:
                sp_importance += video_global_importance.get((sp, video_idx), 0)
        sp_global_importance[sp] = sp_importance

    total_sp_importance = sum(sp_global_importance.values())
    sp_importance_ratio = {}
    for sp in range(nSP):
        sp_importance_ratio[sp] = sp_global_importance[sp] / total_sp_importance if total_sp_importance > 0 else 1.0 / nSP

    for i, node in enumerate(cache_nodes):
        actual_capacity = node_capacities[node]
        pointer_vec = [0] * nSP

        for slot in range(actual_capacity):
            bestSP = 0
            best_score = -1
            best_video_offset = -1

            for currentSP in range(nSP):
                if cacheable_content and currentSP < len(cacheable_content):
                    cacheable_videos = cacheable_content[currentSP]
                    max_video_offset = min(pointer_vec[currentSP] + 30, len(cacheable_videos))
                    for video_offset in range(pointer_vec[currentSP], max_video_offset):
                        video_id = cacheable_videos[video_offset]
                        video_rank = video_offset

                        neighbor_has_video = False
                        for neighbor in cache_node_neighbors[node]:
                            if neighbor in allocation:
                                if allocation[neighbor][currentSP] > video_offset:
                                    neighbor_has_video = True
                                    break

                        local_score = distrib[currentSP][video_rank] * SP_proba[currentSP]

                        global_score = video_global_importance.get((currentSP, video_id), 0)

                        node_imp = node_importance.get(node, 1)
                        max_centrality = max(node_centrality.values()) if node_centrality else 1
                        if max_centrality == 0:
                            max_centrality = 1
                        node_cen = node_centrality.get(node, 0) / max_centrality
                        local_weight = (0.3 + 0.15 * (1 - node_imp)) * (1 - network_size_factor * 0.15)
                        global_weight = (0.5 + 0.1 * node_cen) * (1 + network_size_factor * 0.15)
                        total_weight = local_weight + global_weight
                        local_weight /= total_weight
                        global_weight /= total_weight

                        total_score = local_weight * local_score + global_weight * global_score

                        if neighbor_has_video:
                            score = total_score * 0.05
                        else:
                            score = total_score

                        if score > best_score:
                            best_score = score
                            bestSP = currentSP
                            best_video_offset = video_offset

            if bestSP is not None and best_video_offset != -1:
                pointer_vec[bestSP] = best_video_offset + 1
                pointer_vec[bestSP] = min(pointer_vec[bestSP], len(cacheable_content[bestSP]) if cacheable_content and bestSP < len(cacheable_content) else pointer_vec[bestSP])
                allocation[node][bestSP] += 1

    max_iterations = 15
    if debug:
        print(f"[DEBUG] 开始迭代优化（最多 {max_iterations} 轮）...")
    for iteration in range(max_iterations):
        if debug:
            iter_start = time.time()
            print(f"[DEBUG] 迭代 {iteration + 1}/{max_iterations}...")
        improved = False

        video_cache_frequency, video_cache_coverage = calculate_global_cache_distribution(allocation)

        global_cache_balance = {}
        for sp in range(nSP):
            sp_allocation = sum(allocation[node][sp] for node in cache_nodes)
            global_cache_balance[sp] = sp_allocation / sum(sp_global_importance.values()) if sum(sp_global_importance.values()) > 0 else 0

        for node in cache_nodes:
            actual_capacity = node_capacities[node]
            current_allocation = sum(allocation[node])

            if current_allocation == actual_capacity:
                video_scores = {}
                for sp in range(nSP):
                    if cacheable_content and sp < len(cacheable_content):
                        cacheable_videos = cacheable_content[sp]
                        for video_rank in range(min(allocation[node][sp], 300)):
                            video = cacheable_videos[video_rank]
                            video_key = (sp, video)
                            local_score = distrib[sp][video_rank] * SP_proba[sp]
                            global_score = video_global_importance.get(video_key, 0)

                            cache_frequency = video_cache_frequency.get(video_key, 0)
                            frequency_score = 1.0 / (1 + cache_frequency)

                            coverage = video_cache_coverage.get(video_key, 0)
                            coverage_score = 1.0 / (1 + coverage)

                            balance_score = 1.0 / (1 + abs(global_cache_balance.get(sp, 0) - sp_importance_ratio.get(sp, 0)))

                            node_imp = node_importance.get(node, 1)
                            max_centrality = max(node_centrality.values()) if node_centrality else 1
                            if max_centrality == 0:
                                max_centrality = 1
                            node_cen = node_centrality.get(node, 0) / max_centrality
                            local_weight = (0.2 + 0.15 * (1 - node_imp)) * (1 - network_size_factor * 0.15)
                            global_weight = (0.4 + 0.1 * node_cen) * (1 + network_size_factor * 0.15)
                            frequency_weight = 0.1
                            coverage_weight = 0.1
                            balance_weight = 0.1

                            total_weight = local_weight + global_weight + frequency_weight + coverage_weight + balance_weight
                            local_weight /= total_weight
                            global_weight /= total_weight
                            frequency_weight /= total_weight
                            coverage_weight /= total_weight
                            balance_weight /= total_weight

                            total_score = local_weight * local_score + global_weight * global_score + frequency_weight * frequency_score + coverage_weight * coverage_score + balance_weight * balance_score
                            video_scores[video_key] = total_score

                sorted_videos = sorted(video_scores.items(), key=lambda x: x[1], reverse=True)

                low_score_videos = sorted(video_scores.items(), key=lambda x: x[1])[:30]
                for (sp1, video1), score1 in low_score_videos:
                    if allocation[node][sp1] <= 0:
                        continue

                    for (sp2, video2), score2 in sorted_videos:
                        if sp1 == sp2 and video1 == video2:
                            continue

                        neighbor_sp2_allocation = 0
                        for neighbor in cache_node_neighbors[node]:
                            neighbor_sp2_allocation += allocation[neighbor][sp2]

                        video2_key = (sp2, video2)
                        video2_frequency = video_cache_frequency.get(video2_key, 0)
                        video2_coverage = video_cache_coverage.get(video2_key, 0)

                        if neighbor_sp2_allocation < allocation[node][sp2] + 1 and score2 > score1 * 1.01 and video2_frequency < 3 and video2_coverage < 0.5:
                            if sp2 in video_to_index and video2 in video_to_index[sp2]:
                                video2_position = video_to_index[sp2][video2]
                                if video2_position < allocation[node][sp2] + 1:
                                    allocation[node][sp1] -= 1
                                    allocation[node][sp2] += 1
                                    improved = True
                                    break
                    if improved:
                        break

        if iteration % 2 == 1:
            video_cache_frequency, video_cache_coverage = calculate_global_cache_distribution(allocation)
            over_cached = [k for k, v in video_cache_frequency.items() if v > 3]
            under_cached = sorted(video_global_importance.items(), key=lambda x: x[1], reverse=True)[:150]
            under_cached = [k for k, v in under_cached if video_cache_frequency.get(k, 0) < 2 and video_cache_coverage.get(k, 0) < 0.3]

            for over_key in over_cached:
                sp1, video1 = over_key
                for node in cache_nodes:
                    if allocation[node][sp1] > 0:
                        if sp1 in video_to_index and video1 in video_to_index[sp1]:
                            video_position = video_to_index[sp1][video1]
                            if video_position < allocation[node][sp1]:
                                for under_key in under_cached:
                                    sp2, video2 = under_key
                                    if sp2 in video_to_index and video2 in video_to_index[sp2]:
                                        neighbor_has_video = False
                                        for neighbor in cache_node_neighbors[node]:
                                            if neighbor in allocation:
                                                if allocation[neighbor][sp2] > video_to_index[sp2][video2]:
                                                    neighbor_has_video = True
                                                    break
                                        if not neighbor_has_video:
                                            allocation[node][sp1] -= 1
                                            allocation[node][sp2] += 1
                                            improved = True
                                            break
                    if improved:
                        break

        if not improved:
            if debug:
                print(f"[DEBUG] 迭代 {iteration + 1} 后无改进，提前结束")
            break
        
        if debug:
            iter_time = time.time() - iter_start
            print(f"[DEBUG] 迭代 {iteration + 1} 完成，用时 {iter_time:.2f}秒，改进: {improved}")

    end_time = time.time()
    print(f"全局最优缓存分配方法执行时间: {end_time - start_time:.2f}秒")
    
    if debug:
        print(f"[DEBUG] decide_global_opt_alloc 完成，总用时: {end_time - start_time:.2f}秒")

    return allocation


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

    cache_node_hits = {}
    cache_nodes = topology_manager.get_cache_nodes()

    if len(cache_nodes) == 0:
        avg_latency = user_to_cache_latency + cache_to_sp_latency
        cache_node_hit_rates = {}
        return (1.0, 1.0, 1.0, avg_latency, cache_node_hit_rates)

    for node in cache_nodes:
        cache_node_hits[node] = 0
    
    cache_node_neighbors = {}
    for node in cache_nodes:
        all_neighbors = topology_manager.get_neighbors(node)
        neighbors = [n for n in all_neighbors if n in cache_nodes]
        cache_node_neighbors[node] = neighbors
    
    sp_nodes = topology_manager.get_sp_nodes()
    receiver_nodes = topology_manager.get_receiver_nodes()
    router_nodes = topology_manager.get_router_nodes()

    requests = []
    source_nodes = []

    if fixed_requests:
        if isinstance(fixed_requests, dict) and 'requests' in fixed_requests:
            requests = fixed_requests['requests']
            source_nodes = fixed_requests.get('source_nodes', [])
        elif isinstance(fixed_requests, list):
            requests = fixed_requests
            for request in requests:
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
        for r in range(requests_nb):
            request = request_creation(video_probabi)
            requests.append(request)

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
    
    processed_requests = 0
    
    for r in range(min(requests_nb, len(requests))):
        request = requests[r]
        source_node = source_nodes[r]
        
        if source_node is None:
            continue
            
        SP_of_the_video_requested = request[0]
        video_id = request[1]

        processed_requests += 1

        closest_cache = None
        min_latency = float('inf')

        for cache_node in cache_nodes:
            latency = topology_manager.get_latency(source_node, cache_node)
            if latency < min_latency:
                min_latency = latency
                closest_cache = cache_node

        cache_hit = False
        hit_node = None
        latency = user_to_cache_latency + min_latency
        
        if closest_cache in allocation:
            if cacheable_content and SP_of_the_video_requested < len(cacheable_content):
                cacheable_videos = cacheable_content[SP_of_the_video_requested]
                if video_id in cacheable_videos:
                    video_idx = cacheable_videos.index(video_id)
                    if allocation[closest_cache][SP_of_the_video_requested] > video_idx:
                        cache_hit = True
                        hit_node = closest_cache
        
        total_latency += latency

        if not cache_hit and cooperative_caching:
            for neighbor_node in cache_node_neighbors.get(closest_cache, []):
                if neighbor_node in allocation:
                    if cacheable_content and SP_of_the_video_requested < len(cacheable_content):
                        cacheable_videos = cacheable_content[SP_of_the_video_requested]
                        if video_id in cacheable_videos:
                            video_idx = cacheable_videos.index(video_id)
                            if allocation[neighbor_node][SP_of_the_video_requested] > video_idx:
                                cache_hit = True
                                hit_node = neighbor_node
                                latency += cache_to_cache_latency
                                break

        if not cache_hit:
            cost += 1
            sp_node = sp_nodes[SP_of_the_video_requested] if sp_nodes and SP_of_the_video_requested < len(sp_nodes) else None
            if sp_node and closest_cache:
                sp_latency = topology_manager.get_latency(closest_cache, sp_node)
                latency += sp_latency
            latency += cache_to_sp_latency
        else:
            if hit_node:
                cache_node_hits[hit_node] += 1
                record_request(hit_node, SP_of_the_video_requested, video_id)

        total_latency += latency
        
        if isinstance(best_alloc, dict):
            best_cache_hit = False
            for cache_node in cache_nodes:
                if cache_node in best_alloc:
                    if cacheable_content and SP_of_the_video_requested < len(cacheable_content):
                        cacheable_videos = cacheable_content[SP_of_the_video_requested]
                        if video_id in cacheable_videos:
                            video_idx = cacheable_videos.index(video_id)
                            if best_alloc[cache_node][SP_of_the_video_requested] > video_idx:
                                best_cache_hit = True
                                break
            if not best_cache_hit:
                b_cost += 1
        else:
            if cacheable_content and SP_of_the_video_requested < len(cacheable_content):
                cacheable_videos = cacheable_content[SP_of_the_video_requested]
                if video_id in cacheable_videos:
                    video_idx = cacheable_videos.index(video_id)
                    b_allocated_cache_space = best_alloc[SP_of_the_video_requested]
                    if b_allocated_cache_space <= video_idx:
                        b_cost += 1
                else:
                    b_cost += 1
            else:
                b_cost += 1
        
        if isinstance(first_alloc, dict):
            first_cache_hit = False
            for cache_node in cache_nodes:
                if cache_node in first_alloc:
                    if cacheable_content and SP_of_the_video_requested < len(cacheable_content):
                        cacheable_videos = cacheable_content[SP_of_the_video_requested]
                        if video_id in cacheable_videos:
                            video_idx = cacheable_videos.index(video_id)
                            if first_alloc[cache_node][SP_of_the_video_requested] > video_idx:
                                first_cache_hit = True
                                break
            if not first_cache_hit:
                f_cost += 1
        else:
            if cacheable_content and SP_of_the_video_requested < len(cacheable_content):
                cacheable_videos = cacheable_content[SP_of_the_video_requested]
                if video_id in cacheable_videos:
                    video_idx = cacheable_videos.index(video_id)
                    f_allocated_cache_space = first_alloc[SP_of_the_video_requested]
                    if f_allocated_cache_space <= video_idx:
                        f_cost += 1
                else:
                    f_cost += 1
            else:
                f_cost += 1
    
    actual_requests = processed_requests if processed_requests > 0 else 1
    cost = cost / actual_requests
    b_cost = b_cost / actual_requests
    f_cost = f_cost / actual_requests
    avg_latency = total_latency / actual_requests
    
    cache_node_hit_rates = {}
    for node in cache_nodes:
        cache_node_hit_rates[node] = cache_node_hits[node] / actual_requests
    
    return (cost, f_cost, b_cost, avg_latency, cache_node_hit_rates)


def states_nSP(capacity, numberSP, delta2):
    """
    生成所有可能的状态（缓存分配）
    """
    MAX_STATES = 1000000
    state_count = 0

    def generate_states(cap, sp, delta):
        nonlocal state_count
        states = []

        if sp == 1:
            if state_count < MAX_STATES:
                states.append([cap])
                state_count += 1
            return states
        elif sp == 2:
            for j in range(cap + 1):
                if state_count >= MAX_STATES:
                    break
                states.append([j, cap - j])
                state_count += 1
            return states
        else:
            for i in range(cap + 1):
                if state_count >= MAX_STATES:
                    break
                sub_states = generate_states(cap - i, sp - 1, delta)
                for sub_state in sub_states:
                    states.append([i] + sub_state)
                    state_count += 1
                    if state_count >= MAX_STATES:
                        break
            return states

    return generate_states(capacity, numberSP, delta2)


def get_state_index(allocation, delta, states):
    total = sum(allocation) if sum(allocation) > 0 else 1
    normalized = [x / total for x in allocation]
    state_tuple = tuple([int(x * 10) for x in normalized])

    for idx, s in enumerate(states):
        if tuple(s) == state_tuple:
            return idx
    return -1


def find_epsilon(time):
    A = 0.5
    B = 0.15
    C = 0.003

    standardized_time = (time - B * simulation_time) / (A * simulation_time)
    cosh = np.cosh(math.exp(-standardized_time))
    epsilon = 0.9 - (0.8 / cosh + (time * C / simulation_time))
    return max(epsilon, 0.15)


def find_N(time):
    D = 0.15
    E = 0.3
    F = 0.5

    standardized_time = (time - D * simulation_time) / (E * simulation_time)
    cosh = np.cosh(math.exp(-standardized_time))
    N = round((100 / cosh + (time * F / simulation_time)))
    return N


def take_action_multi_cache(allocation, epsilon, D, delta, Q, state_index, cache_nodes, nSP, states, state_to_index):
    """
    执行多缓存节点的动作
    """
    old_allocation = deepcopy(allocation)
    D_size = len(D)

    actions = {}
    action_plus = {}
    action_minus = {}

    adjusted_delta = min(delta, 2)

    for node in cache_nodes:
        node_state_index = state_index[node]

        alea = rd.random()
        coeff_ind = rd.randint(0, D_size-1) if D_size > 0 else 0
        coeff = D[coeff_ind] if D_size > 0 else 1

        action_step = min(coeff * adjusted_delta, 1)

        if alea <= epsilon:
            action = rd.randint(0, nSP**2 - 1)
            action_plus_val = action // nSP
            action_minus_val = action % nSP

            if action_plus_val == action_minus_val:
                action_plus[node] = None
                action_minus[node] = None
                actions[node] = None
                continue

            temp_allocation = allocation[node].copy()
            temp_allocation[action_plus_val] += action_step
            temp_allocation[action_minus_val] -= action_step

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
                action_plus[node] = None
                action_minus[node] = None
                actions[node] = None

        else:
            q_values = Q[node][:, node_state_index].flatten()
            (best_score, best_actions) = af.find_max_list(q_values)

            if best_actions:
                action = rd.choice(best_actions)
                action_plus_val = action // nSP
                action_minus_val = action % nSP

                if action_plus_val == action_minus_val:
                    action_plus[node] = None
                    action_minus[node] = None
                    actions[node] = None
                    continue

                temp_allocation = allocation[node].copy()
                temp_allocation[action_plus_val] += action_step
                temp_allocation[action_minus_val] -= action_step

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
                    action_plus[node] = None
                    action_minus[node] = None
                    actions[node] = None
            else:
                action = rd.randint(0, nSP**2 - 1)
                action_plus_val = action // nSP
                action_minus_val = action % nSP

                if action_plus_val == action_minus_val:
                    action_plus[node] = None
                    action_minus[node] = None
                    actions[node] = None
                    continue

                temp_allocation = allocation[node].copy()
                temp_allocation[action_plus_val] += action_step
                temp_allocation[action_minus_val] -= action_step

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
                    action_plus[node] = None
                    action_minus[node] = None
                    actions[node] = None

    for node in cache_nodes:
        original_alloc = tuple(allocation[node])
        new_state_index = state_to_index[node].get(original_alloc, -1)
        if new_state_index != -1:
            state_index[node] = new_state_index

    return (actions, action_minus, action_plus, allocation)


def optimize_nSP(allocation, initial_videos_proba, best_allocation, request_rate, nb_interval, interval_size, gama, delta, D, method, fixed_requests=None, debug_interval=None, debug=False):
    """优化缓存分配 - 完整版本，支持所有8种方法"""
    global save_allocations, results_dir, epsilon_decay, alpha_scheduling, activate_memory
    global single_cache_capacity, nSP, topology_manager
    global cacheable_content, videos_proba, SP_proba

    if debug:
        print(f"[DEBUG] 开始执行方法: {method}")
        print(f"[DEBUG] 参数: nb_interval={nb_interval}, interval_size={interval_size}, request_rate={request_rate}")

    if method == 'best_allocation':
        print("使用最佳分配方法")
        request_nb = int(interval_size * request_rate)

        L_total_cost = []
        L_nominal_cost = []
        L_first_cost = []
        L_best_cost = []
        L_avg_latency = []

        for interval in range(nb_interval):
            interval_fixed_requests = fixed_requests[interval] if fixed_requests and len(fixed_requests) > interval else None
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

        allocations_file = os.path.join(results_dir, f"allocations_real_data_single_cache_capacity{single_cache_capacity}_request_rate{request_rate}.txt")
        f = open(allocations_file, "w") if save_allocations else None
        if save_allocations and f:
            f.write(str(best_allocation) + '\n')
        if f:
            f.close()

        return best_allocation, L_total_cost, L_nominal_cost, L_first_cost, L_best_cost, L_avg_latency

    elif method == 'cooperative_best_allocation':
        print("使用考虑邻居协作的最佳分配方法")
        cooperative_allocation = decide_cooperative_opt_alloc(initial_videos_proba)
        print(f"考虑邻居协作的最佳分配: {cooperative_allocation}")

        request_nb = int(interval_size * request_rate)
        L_total_cost = []
        L_nominal_cost = []
        L_first_cost = []
        L_best_cost = []
        L_avg_latency = []

        for interval in range(nb_interval):
            interval_fixed_requests = fixed_requests[interval] if fixed_requests and len(fixed_requests) > interval else None
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

        allocations_file = os.path.join(results_dir, f"allocations_real_data_single_cache_capacity{single_cache_capacity}_request_rate{request_rate}_cooperative.txt")
        f = open(allocations_file, "w") if save_allocations else None
        if save_allocations and f:
            f.write(str(cooperative_allocation) + '\n')
        if f:
            f.close()

        return cooperative_allocation, L_total_cost, L_nominal_cost, L_first_cost, L_best_cost, L_avg_latency

    elif method == 'global_opt_allocation':
        print("使用全局最优缓存分配方法")
        global_allocation = decide_global_opt_alloc(initial_videos_proba, debug=debug)
        print(f"全局最优缓存分配: {global_allocation}")

        request_nb = int(interval_size * request_rate)
        L_total_cost = []
        L_nominal_cost = []
        L_first_cost = []
        L_best_cost = []
        L_avg_latency = []

        for interval in range(nb_interval):
            interval_fixed_requests = fixed_requests[interval] if fixed_requests and len(fixed_requests) > interval else None
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

        allocations_file = os.path.join(results_dir, f"allocations_real_data_single_cache_capacity{single_cache_capacity}_request_rate{request_rate}_global_opt.txt")
        f = open(allocations_file, "w") if save_allocations else None
        if save_allocations and f:
            f.write(str(global_allocation) + '\n')
        if f:
            f.close()

        return global_allocation, L_total_cost, L_nominal_cost, L_first_cost, L_best_cost, L_avg_latency

    elif method == 'equal_allocation':
        print("使用平均分配方法")

        cache_nodes = topology_manager.get_cache_nodes()
        if len(cache_nodes) == 0:
            avg_latency = user_to_cache_latency + cache_to_sp_latency
            L_total_cost = [1.0] * nb_interval
            L_nominal_cost = [1.0] * nb_interval
            L_first_cost = [1.0] * nb_interval
            L_best_cost = [1.0] * nb_interval
            L_avg_latency = [avg_latency] * nb_interval
            return {}, L_total_cost, L_nominal_cost, L_first_cost, L_best_cost, L_avg_latency

        equal_allocation = {}
        node_cache_capacity = single_cache_capacity

        for i, node in enumerate(cache_nodes):
            actual_capacity = node_cache_capacity
            per_sp_capacity = actual_capacity // nSP
            extra = actual_capacity % nSP

            node_allocation = []
            for sp_idx in range(nSP):
                node_allocation.append(per_sp_capacity)
                if sp_idx == 0:
                    node_allocation[0] += extra

            equal_allocation[node] = node_allocation

        print(f"平均分配: {equal_allocation}")

        request_nb = int(interval_size * request_rate)
        L_total_cost = []
        L_nominal_cost = []
        L_first_cost = []
        L_best_cost = []
        L_avg_latency = []

        for interval in range(nb_interval):
            interval_fixed_requests = fixed_requests[interval] if fixed_requests and len(fixed_requests) > interval else None
            debug_output = (debug_interval is not None and interval == debug_interval)
            (nominal_cost, first_cost, best_cost, avg_latency, cache_node_hit_rates) = evaluate_cost(
                equal_allocation, allocation, best_allocation, request_nb, initial_videos_proba, interval_fixed_requests, debug_output
            )
            L_total_cost.append(nominal_cost)
            L_nominal_cost.append(nominal_cost)
            L_first_cost.append(first_cost)
            L_best_cost.append(best_cost)
            L_avg_latency.append(avg_latency)

        print(f"平均分配成本: {sum(L_nominal_cost) / len(L_nominal_cost):.4f}")

        return equal_allocation, L_total_cost, L_nominal_cost, L_first_cost, L_best_cost, L_avg_latency

    elif method == 'SCA_ADMM':
        print("使用SCA-ADMM方法（真实数据版）")

        cache_nodes = topology_manager.get_cache_nodes()
        node_cache_capacity = single_cache_capacity

        sca_admm_allocation = {}
        for i, node in enumerate(cache_nodes):
            actual_capacity = node_cache_capacity
            base_allocation = actual_capacity // nSP
            remainder = actual_capacity % nSP

            node_allocation = []
            for sp in range(nSP):
                if sp < remainder:
                    node_allocation.append(base_allocation + 1)
                else:
                    node_allocation.append(base_allocation)

            sca_admm_allocation[node] = node_allocation

        max_sca_iterations_per_interval = 20
        max_admm_iterations = 100
        tau = 0.05
        rho = 0.2
        lambda_lasso = 0

        smoothing_alpha = 0.9
        max_change = 0.25
        convergence_threshold = 0.05

        cache_node_neighbors = {}
        for node in cache_nodes:
            all_neighbors = topology_manager.get_neighbors(node)
            neighbors = [n for n in all_neighbors if n in cache_nodes]
            cache_node_neighbors[node] = neighbors

        def adaptive_parameter_adjustment(primal_residual, dual_residual, iteration, current_rho, current_tau):
            """
            基于残差自适应调整ADMM参数（真实数据版本）
            """
            new_rho = current_rho
            new_tau = current_tau
            
            if primal_residual > 0 and dual_residual > 0:
                residual_ratio = primal_residual / dual_residual
                
                if residual_ratio > 10:
                    new_rho = min(current_rho * 2.0, 1.0)
                elif residual_ratio < 0.1:
                    new_rho = max(current_rho / 2.0, 0.05)
            
            new_tau = current_tau * (0.95 ** min(iteration, 10))
            
            return new_rho, new_tau

        node_local_states = {}
        previous_sp_video_scores = {}
        previous_sp_proba = {}
        neighbor_history = {}

        for node in cache_nodes:
            neighbors = cache_node_neighbors[node]
            node_local_states[node] = {
                'z': {},
                'v': {},
                'request_history': {
                    'sp_requests': {},
                    'video_requests': {},
                    'time_series': []
                },
                'performance_history': []
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

        allocations_file = os.path.join(results_dir, f"allocations_real_data_single_cache_capacity{single_cache_capacity}_request_rate{request_rate}_SCA_ADMM.txt")
        f = open(allocations_file, "w") if save_allocations else None

        # 保存初始分配作为第一行
        if save_allocations and f:
            initial_allocation_for_save = {}
            for node in sca_admm_allocation:
                initial_allocation_for_save[node] = [int(x) for x in sca_admm_allocation[node]]
            f.write(str(initial_allocation_for_save) + '\n')

        L_total_cost = []
        L_nominal_cost = []
        L_first_cost = []
        L_best_cost = []
        L_avg_latency = []

        request_nb = int(interval_size * request_rate)

        for interval in range(nb_interval):
            current_videos_proba = catalog()

            interval_fixed_requests = fixed_requests[interval] if fixed_requests and len(fixed_requests) > interval else None
            debug_output = (debug_interval is not None and interval == debug_interval)

            online_best_allocation = decide_opt_alloc(current_videos_proba)

            (current_cost, _, _, current_latency, current_hit_rates) = evaluate_cost(
                sca_admm_allocation, sca_admm_allocation, online_best_allocation, request_nb, current_videos_proba, interval_fixed_requests, debug_output
            )

            current_interval_allocation = {}
            for node in sca_admm_allocation:
                current_interval_allocation[node] = sca_admm_allocation[node].copy()

            for sca_iter in range(max_sca_iterations_per_interval):
                linear_coeffs = {}
                for node in cache_nodes:
                    coeffs = {}
                    neighbors = cache_node_neighbors[node]
                    x_current = current_interval_allocation[node]

                    for sp in range(nSP):
                        node_sp_proba = estimate_sp_proba(node, sp)
                        h_i = 1.0 - math.exp(-x_current[sp] / 50.0)
                        dh_i = math.exp(-x_current[sp] / 50.0) / 50.0

                        product_term = 1.0
                        for neighbor in neighbors:
                            if neighbor in current_interval_allocation:
                                neighbor_alloc = current_interval_allocation[neighbor][sp]
                                h_j = 1.0 - math.exp(-neighbor_alloc / 50.0)
                                product_term *= (1 - h_j)

                        signal_boost = 100.0
                        coeffs[sp] = node_sp_proba * product_term * dh_i * signal_boost
                    linear_coeffs[node] = coeffs

                z = {}
                v = {}
                for node in cache_nodes:
                    z[node] = node_local_states[node]['z']
                    v[node] = node_local_states[node]['v']

                current_rho = rho
                current_tau = tau

                for admm_iter in range(max_admm_iterations):
                    new_allocation = {}

                    for node in cache_nodes:
                        neighbors = cache_node_neighbors[node]
                        has_neighbors = len(neighbors) > 0

                        neighbor_video_cache = {}
                        for neighbor in neighbors:
                            neighbor_video_cache[neighbor] = {}
                            if neighbor in sca_admm_allocation:
                                for sp in range(nSP):
                                    neighbor_sp_alloc = sca_admm_allocation[neighbor][sp]
                                    neighbor_video_cache[neighbor][sp] = int(neighbor_sp_alloc)

                        actual_capacity = node_cache_capacity

                        sp_video_scores = []
                        for sp in range(nSP):
                            if cacheable_content and sp < len(cacheable_content):
                                cacheable_videos = cacheable_content[sp]
                                sp_score = 0.0
                                node_sp_proba = estimate_sp_proba(node, sp)
                                
                                for video_idx, video in enumerate(cacheable_videos):
                                    if video_idx >= actual_capacity:
                                        break
                                    video_proba = estimate_video_proba(node, sp, video)
                                    
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
                                    
                                    cache_benefit = 1.0 if not neighbor_has_video else 0.5
                                    cache_benefit *= neighbor_benefit
                                    
                                    long_term_benefit = 1.0
                                    if node in node_local_states:
                                        history = node_local_states[node]['request_history']
                                        if 'video_requests' in history and sp in history['video_requests']:
                                            video_history = history['video_requests'][sp]
                                            if video in video_history:
                                                video_count = video_history[video]
                                                total_count = sum(video_history.values())
                                                if total_count > 0:
                                                    long_term_benefit = 1.0 + (video_count / total_count) * 0.5
                                    
                                    sp_score += video_proba * node_sp_proba * cache_benefit * long_term_benefit
                                sp_video_scores.append(sp_score)
                            else:
                                sp_video_scores.append(0.0)

                        if has_neighbors:
                            sum_z_minus_v = [0.0] * nSP
                            for neighbor in neighbors:
                                for sp in range(nSP):
                                    sum_z_minus_v[sp] += z[node][neighbor][sp] - v[node][neighbor][sp] / rho
                            avg_z_minus_v = [s / len(neighbors) for s in sum_z_minus_v]
                        else:
                            avg_z_minus_v = [0.0] * nSP
                        
                        num_neighbors = len(neighbors) if neighbors else 1
                        x_i = sca_admm_allocation[node].copy()
                        
                        # 增强信号强度以驱动优化
                        signal_boost = 100.0
                        boosted_scores = [s * signal_boost for s in sp_video_scores]
                        
                        if has_neighbors:
                            for sp in range(nSP):
                                numerator = boosted_scores[sp] + current_rho * num_neighbors * avg_z_minus_v[sp] + current_tau * x_i[sp]
                                denominator = current_rho * num_neighbors + current_tau
                                new_value = numerator / denominator
                                
                                old_value = x_i[sp]
                                if new_value > old_value:
                                    new_value = min(old_value * (1 + max_change), new_value)
                                else:
                                    new_value = max(old_value * (1 - max_change), new_value)
                                
                                x_i[sp] = new_value
                        else:
                            gradient = boosted_scores.copy()
                            gradient_step = 10.0
                            
                            for sp in range(nSP):
                                x_i[sp] += gradient_step * gradient[sp]
                            
                            for sp in range(nSP):
                                x_i[sp] = max(0, x_i[sp])
                            
                            total = sum(x_i)
                            if total > 0:
                                scale = actual_capacity / total
                                for sp in range(nSP):
                                    x_i[sp] *= scale
                            
                            old_x_i = sca_admm_allocation[node].copy()
                            for sp in range(nSP):
                                if x_i[sp] > old_x_i[sp]:
                                    x_i[sp] = min(old_x_i[sp] * (1 + max_change), x_i[sp])
                                else:
                                    x_i[sp] = max(old_x_i[sp] * (1 - max_change), x_i[sp])
                            
                            total = sum(x_i)
                            if total > 0:
                                scale = actual_capacity / total
                                for sp in range(nSP):
                                    x_i[sp] *= scale

                        for sp in range(nSP):
                            x_i[sp] = max(0, x_i[sp])

                        total = sum(x_i)
                        if total > 0:
                            scale = actual_capacity / total
                            for sp in range(nSP):
                                x_i[sp] *= scale

                        new_allocation[node] = x_i
                    
                    for node in cache_nodes:
                        neighbors = cache_node_neighbors[node]
                        for neighbor in neighbors:
                            a = new_allocation[node].copy()
                            for sp in range(nSP):
                                a[sp] += v[node][neighbor][sp] / rho
                            b = new_allocation[neighbor].copy()
                            for sp in range(nSP):
                                b[sp] += v[neighbor][node][sp] / rho
                            
                            d = [a[sp] - b[sp] for sp in range(nSP)]
                            norm_d = np.linalg.norm(d)
                            
                            if norm_d > 0:
                                threshold = (2 * lambda_lasso) / rho
                                if norm_d > threshold:
                                    scale = 1 - threshold / norm_d
                                    d = [scale * x for x in d]
                                else:
                                    d = [0.0] * nSP
                            
                            m = [(a[sp] + b[sp]) / 2 for sp in range(nSP)]
                            z[node][neighbor] = [m[sp] + d[sp] / 2 for sp in range(nSP)]
                            z[neighbor][node] = [m[sp] - d[sp] / 2 for sp in range(nSP)]
                    
                    for node in cache_nodes:
                        neighbors = cache_node_neighbors[node]
                        for neighbor in neighbors:
                            for sp in range(nSP):
                                v[node][neighbor][sp] += current_rho * (new_allocation[node][sp] - z[node][neighbor][sp])
                                v[neighbor][node][sp] += current_rho * (new_allocation[neighbor][sp] - z[neighbor][node][sp])
                    
                    primal_residual = 0.0
                    dual_residual = 0.0
                    
                    for node in cache_nodes:
                        neighbors = cache_node_neighbors[node]
                        for neighbor in neighbors:
                            for sp in range(nSP):
                                primal_residual += (new_allocation[node][sp] - z[node][neighbor][sp])**2
                    primal_residual = np.sqrt(primal_residual)
                    
                    for node in cache_nodes:
                        neighbors = cache_node_neighbors[node]
                        for neighbor in neighbors:
                            for sp in range(nSP):
                                dual_residual += (v[node][neighbor][sp] - node_local_states[node]['v'].get(neighbor, [0]*nSP)[sp])**2
                    dual_residual = np.sqrt(dual_residual) * current_rho
                    
                    if admm_iter > 0 and admm_iter % 5 == 0:
                        current_rho, current_tau = adaptive_parameter_adjustment(
                            primal_residual, dual_residual, admm_iter, current_rho, current_tau
                        )
                    
                    allocation_diff = 0.0
                    for node in cache_nodes:
                        for sp in range(nSP):
                            allocation_diff += abs(new_allocation[node][sp] - sca_admm_allocation[node][sp])
                    
                    if primal_residual < 1e-3 and dual_residual < 1e-3:
                        break
                
                sca_admm_allocation = new_allocation
                
                # 更新节点本地状态
                for node in cache_nodes:
                    node_local_states[node]['z'] = z[node]
                    node_local_states[node]['v'] = v[node]
            
            # 四舍五入到整数
            for node in cache_nodes:
                actual_capacity = node_cache_capacity
                node_allocation = sca_admm_allocation[node]
                rounded_allocation = [round(x) for x in node_allocation]
                total = sum(rounded_allocation)
                if total != actual_capacity:
                    diff = actual_capacity - total
                    max_idx = rounded_allocation.index(max(rounded_allocation))
                    rounded_allocation[max_idx] += diff
                rounded_allocation = [max(0, x) for x in rounded_allocation]
                sca_admm_allocation[node] = rounded_allocation

            (nominal_cost, first_cost, best_cost, avg_latency, cache_node_hit_rates) = evaluate_cost(
                sca_admm_allocation, sca_admm_allocation, online_best_allocation, request_nb, current_videos_proba, interval_fixed_requests
            )
            L_total_cost.append(nominal_cost)
            L_nominal_cost.append(nominal_cost)
            L_first_cost.append(first_cost)
            L_best_cost.append(best_cost)
            L_avg_latency.append(avg_latency)

            integer_allocation_for_save = {}
            for node in sca_admm_allocation:
                integer_allocation_for_save[node] = [max(0, int(round(x))) for x in sca_admm_allocation[node]]

            if save_allocations and f:
                f.write(str(integer_allocation_for_save) + '\n')

            # 更新邻居历史信息
            for node in cache_nodes:
                neighbors = cache_node_neighbors[node]
                current_allocation = sca_admm_allocation[node].copy()
                recent_hit_rate = cache_node_hit_rates.get(node, 0.0)
                top_videos = {sp: [] for sp in range(nSP)}
                
                if node in node_local_states:
                    history = node_local_states[node]['request_history']
                    if 'video_requests' in history:
                        for sp in range(nSP):
                            if sp in history['video_requests']:
                                video_counts = history['video_requests'][sp]
                                sorted_videos = sorted(video_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                                top_videos[sp] = [v[0] for v in sorted_videos]
                
                for neighbor in neighbors:
                    if node in neighbor_history and neighbor in neighbor_history[node]:
                        neighbor_history[node][neighbor]['allocation_history'].append(current_allocation)
                        neighbor_history[node][neighbor]['hit_rate_history'].append(recent_hit_rate)
                        neighbor_history[node][neighbor]['request_pattern'] = top_videos

        if f:
            f.close()

        print(f"SCA-ADMM成本: {sum(L_nominal_cost) / len(L_nominal_cost):.4f}")

        print("\n最终缓存分配结果:")
        for node in sorted(sca_admm_allocation.keys()):
            alloc = sca_admm_allocation[node]
            print(f"  节点 {node}: {alloc} (总计: {sum(alloc)})")

        return sca_admm_allocation, L_total_cost, L_nominal_cost, L_first_cost, L_best_cost, L_avg_latency

    elif method == 'SCA_neighborhood_search':
        print("使用SCA_neighborhood_search方法（离散邻域搜索优化）")
        print("  特点：SCA外循环 + 离散邻域搜索替代ADMM")
        print("  在线学习：只依赖自身和邻居历史信息")
        print("  视频级别信息交互：模仿SCA_ADMM的视频级别协同")
        print("  优化方法：离散邻域搜索，适合整数规划问题")

        start_time = time.time()
        cache_nodes = topology_manager.get_cache_nodes()
        num_cache_nodes = len(cache_nodes)

        sca_neighborhood_allocation = {}
        for i, node in enumerate(cache_nodes):
            actual_capacity = single_cache_capacity
            base_allocation = actual_capacity // nSP
            remainder = actual_capacity % nSP
            node_allocation = []
            for sp in range(nSP):
                if sp < remainder:
                    node_allocation.append(base_allocation + 1)
                else:
                    node_allocation.append(base_allocation)
            sca_neighborhood_allocation[node] = node_allocation

        print(f"初始分配: {sca_neighborhood_allocation}")

        allocations_file = os.path.join(results_dir, f"allocations_real_data_single_cache_capacity{single_cache_capacity}_request_rate{request_rate}_SCA_neighborhood_search.txt")
        f = open(allocations_file, "w") if save_allocations else None

        # 保存初始分配作为第一行
        if save_allocations and f:
            initial_allocation_for_save = {}
            for node in sca_neighborhood_allocation:
                initial_allocation_for_save[node] = [int(x) for x in sca_neighborhood_allocation[node]]
            f.write(str(initial_allocation_for_save) + '\n')

        request_nb = int(interval_size * request_rate)
        interval_fixed_requests = fixed_requests[0] if fixed_requests and len(fixed_requests) > 0 else None
        (_, fixed_first_cost, fixed_best_cost, _, _) = evaluate_cost(
            best_allocation, sca_neighborhood_allocation, best_allocation, request_nb, initial_videos_proba, interval_fixed_requests
        )

        max_sca_iterations_per_interval = 20
        max_neighborhood_search_iterations = 50
        neighborhood_size = 3
        temperature = 1.0 #0.3 #1.0
        cooling_rate = 0.95
        no_improvement_threshold = 5
        smoothing_alpha = 0.9

        cache_node_neighbors = {}
        for node in cache_nodes:
            all_neighbors = topology_manager.get_neighbors(node)
            neighbors = [n for n in all_neighbors if n in cache_nodes]
            cache_node_neighbors[node] = neighbors

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

        L_total_cost = []
        L_nominal_cost = []
        L_first_cost = []
        L_best_cost = []
        L_avg_latency = []
        L_sca_iterations = []
        L_gradient_tracking_iterations = []

        for interval in range(nb_interval):
            current_videos_proba = catalog()
            interval_fixed_requests = fixed_requests[interval] if fixed_requests and len(fixed_requests) > interval else None
            debug_output = (debug_interval is not None and interval == debug_interval)

            online_best_allocation = decide_opt_alloc(current_videos_proba)

            (current_cost, _, _, current_latency, current_hit_rates) = evaluate_cost(
                sca_neighborhood_allocation, sca_neighborhood_allocation, online_best_allocation, request_nb, current_videos_proba, interval_fixed_requests, debug_output
            )

            current_interval_allocation = {}
            for node in sca_neighborhood_allocation:
                current_interval_allocation[node] = sca_neighborhood_allocation[node].copy()

            L_sca_iterations.append(max_sca_iterations_per_interval)
            total_neighborhood_search_iterations = 0

            for sca_iter in range(max_sca_iterations_per_interval):
                neighborhood_search_converged = False
                previous_allocation = sca_neighborhood_allocation.copy()

                no_improvement_count = 0
                best_cost_so_far = float('inf')

                for search_iter in range(max_neighborhood_search_iterations):
                    total_neighborhood_search_iterations += 1

                    if search_iter > 0:
                        max_residual = 0.0
                        for node in cache_nodes:
                            if node in previous_allocation and node in sca_neighborhood_allocation:
                                for sp in range(nSP):
                                    residual = abs(sca_neighborhood_allocation[node][sp] - previous_allocation[node][sp])
                                    max_residual = max(max_residual, residual)

                        if max_residual < 1e-3:
                            neighborhood_search_converged = True
                            break

                    current_total_cost = 0.0
                    for node in cache_nodes:
                        current_allocation = sca_neighborhood_allocation[node]
                        node_cost = sum(current_allocation)
                        current_total_cost += node_cost

                    if current_total_cost < best_cost_so_far - 1e-3:
                        best_cost_so_far = current_total_cost
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                        if no_improvement_count >= no_improvement_threshold:
                            neighborhood_search_converged = True
                            break

                    previous_allocation = sca_neighborhood_allocation.copy()

                    new_allocation = {}

                    for node in cache_nodes:
                        neighbors = cache_node_neighbors[node]
                        has_neighbors = len(neighbors) > 0

                        actual_capacity = single_cache_capacity
                        current_allocation = sca_neighborhood_allocation[node].copy()

                        current_cost = 0.0

                        for sp in range(nSP):
                            if cacheable_content and sp < len(cacheable_content):
                                node_sp_proba = estimate_sp_proba(node, sp)
                                sp_alloc = current_allocation[sp]
                                neighbor_coefficient = 0.7 if has_neighbors else 0.9
                                long_term_factor = 1.0
                                history = node_local_states[node]['request_history']
                                if 'sp_requests' in history and int(sp) in history['sp_requests']:
                                    sp_count = history['sp_requests'][int(sp)]
                                    total_count = sum(history['sp_requests'].values())
                                    if total_count > 0:
                                        long_term_factor = 1.0 - (sp_count / total_count) * 0.3

                                sp_cost = sp_alloc * node_sp_proba * neighbor_coefficient * long_term_factor
                                current_cost += sp_cost

                        best_neighbor_allocation = current_allocation.copy()
                        best_neighbor_cost = current_cost

                        gradient = []
                        for sp in range(nSP):
                            if cacheable_content and sp < len(cacheable_content):
                                node_sp_proba = estimate_sp_proba(node, sp)
                                gradient.append(node_sp_proba)
                            else:
                                gradient.append(0.0)

                        gradient_sum = sum(gradient)
                        if gradient_sum > 0:
                            gradient = [g / gradient_sum for g in gradient]

                        for _ in range(neighborhood_size):
                            neighbor_allocation = current_allocation.copy()

                            sp_to_adjust = rd.choices(range(nSP), weights=gradient, k=1)[0]

                            if gradient[sp_to_adjust] > 0.5:
                                adjustment = 1
                            else:
                                adjustment = rd.choice([-1, 1])

                            if 0 <= neighbor_allocation[sp_to_adjust] + adjustment <= actual_capacity:
                                neighbor_allocation[sp_to_adjust] += adjustment

                                total_alloc = sum(neighbor_allocation)
                                if total_alloc > actual_capacity:
                                    other_sp = rd.randint(0, nSP - 1)
                                    while other_sp == sp_to_adjust or neighbor_allocation[other_sp] <= 0:
                                        other_sp = rd.randint(0, nSP - 1)
                                    neighbor_allocation[other_sp] -= (total_alloc - actual_capacity)
                                elif total_alloc < actual_capacity:
                                    other_sp = rd.randint(0, nSP - 1)
                                    neighbor_allocation[other_sp] += (actual_capacity - total_alloc)

                                sp_cost_change = 0.0

                                if cacheable_content and sp_to_adjust < len(cacheable_content):
                                    node_sp_proba = estimate_sp_proba(node, sp_to_adjust)
                                    old_alloc = current_allocation[sp_to_adjust]
                                    new_alloc = neighbor_allocation[sp_to_adjust]
                                    allocation_change = new_alloc - old_alloc
                                    cost_coefficient = -0.8
                                    sp_cost_change = allocation_change * node_sp_proba * cost_coefficient

                                neighbor_cost = current_cost + sp_cost_change

                                if neighbor_cost < best_neighbor_cost:
                                    best_neighbor_allocation = neighbor_allocation.copy()
                                    best_neighbor_cost = neighbor_cost
                                elif rd.random() < math.exp(-(neighbor_cost - best_neighbor_cost) / temperature):
                                    best_neighbor_allocation = neighbor_allocation.copy()
                                    best_neighbor_cost = neighbor_cost

                        new_allocation[node] = best_neighbor_allocation

                        node_local_states[node]['neighborhood_search']['current_cost'] = best_neighbor_cost
                        node_local_states[node]['neighborhood_search']['best_allocation'] = best_neighbor_allocation.copy()

                    for node in cache_nodes:
                        sca_neighborhood_allocation[node] = new_allocation[node]

                    temperature *= cooling_rate

                L_gradient_tracking_iterations.append(total_neighborhood_search_iterations)

                for node in cache_nodes:
                    current_allocation = sca_neighborhood_allocation[node].copy()
                    if 'previous_allocation' in node_local_states[node]:
                        previous_allocation = node_local_states[node]['previous_allocation']
                        for sp in range(nSP):
                            sca_neighborhood_allocation[node][sp] = smoothing_alpha * current_allocation[sp] + (1 - smoothing_alpha) * previous_allocation[sp]
                    node_local_states[node]['previous_allocation'] = current_allocation.copy()

            interval_fixed_requests = fixed_requests[interval] if fixed_requests and len(fixed_requests) > interval else None
            (nominal_cost, first_cost, best_cost, avg_latency, cache_node_hit_rates) = evaluate_cost(
                sca_neighborhood_allocation, sca_neighborhood_allocation, online_best_allocation, request_nb, current_videos_proba, interval_fixed_requests
            )
            total_cost = nominal_cost

            L_total_cost.append(total_cost)
            L_nominal_cost.append(nominal_cost)
            L_first_cost.append(first_cost)
            L_best_cost.append(best_cost)
            L_avg_latency.append(avg_latency)

            integer_allocation_for_save = {}
            for node in sca_neighborhood_allocation:
                integer_allocation_for_save[node] = [max(0, min(single_cache_capacity, int(round(x)))) for x in sca_neighborhood_allocation[node]]

            if save_allocations and f:
                f.write(str(integer_allocation_for_save) + '\n')

        if save_allocations and f:
            f.close()

        for node in cache_nodes:
            if node in sca_neighborhood_allocation:
                # 四舍五入并限制范围
                rounded = [max(0, min(single_cache_capacity, int(round(x)))) for x in sca_neighborhood_allocation[node]]
                total = sum(rounded)
                
                # 调整总和使其精确等于 single_cache_capacity
                if total != single_cache_capacity:
                    diff = single_cache_capacity - total
                    if diff > 0:
                        # 总和不足，将差值加到最大的元素上
                        max_idx = rounded.index(max(rounded))
                        rounded[max_idx] += diff
                    else:
                        # 总和超出，从最大的元素减去多余部分
                        while diff < 0:
                            max_idx = rounded.index(max(rounded))
                            if rounded[max_idx] > 0:
                                rounded[max_idx] -= 1
                                diff += 1
                            else:
                                break
                
                sca_neighborhood_allocation[node] = rounded

        print(f"\n最终SCA_neighborhood_search分配（整数化后）: {sca_neighborhood_allocation}")
        #print(f"每周期SCA迭代次数: {L_sca_iterations}")
        if L_sca_iterations:
            avg_sca_iter = sum(L_sca_iterations) / len(L_sca_iterations)
            #print(f"平均SCA迭代次数: {avg_sca_iter:.1f}")
        #print(f"每周期邻域搜索迭代次数: {L_gradient_tracking_iterations}")
        if L_gradient_tracking_iterations:
            avg_search_iter = sum(L_gradient_tracking_iterations) / len(L_gradient_tracking_iterations)
            #print(f"平均邻域搜索迭代次数: {avg_search_iter:.1f}")

        #print(f"SCA_neighborhood_search成本: {sum(L_nominal_cost) / len(L_nominal_cost):.4f}")

        if debug:
            total_time = time.time() - start_time
            avg_interval_time = total_time / nb_interval if nb_interval > 0 else 0
            print(f"[DEBUG] SCA_neighborhood_search 完成: 总用时={total_time:.2f}秒, 平均间隔用时={avg_interval_time:.2f}秒")
            if L_sca_iterations:
                avg_sca_iter = sum(L_sca_iterations) / len(L_sca_iterations)
                print(f"[DEBUG] 平均SCA迭代次数: {avg_sca_iter:.1f}")

        return sca_neighborhood_allocation, L_total_cost, L_nominal_cost, L_first_cost, L_best_cost, L_avg_latency

        if f:
            f.close()

        print(f"SCA_neighborhood_search成本: {sum(L_nominal_cost) / len(L_nominal_cost):.4f}")
        
        if debug:
            total_time = time.time() - start_time
            avg_interval_time = total_time / nb_interval if nb_interval > 0 else 0
            print(f"[DEBUG] SCA_neighborhood_search 完成: 总用时={total_time:.2f}秒, 平均间隔用时={avg_interval_time:.2f}秒")
            if L_sca_iterations:
                avg_sca_iter = sum(L_sca_iterations) / len(L_sca_iterations)
                print(f"[DEBUG] 平均SCA迭代次数: {avg_sca_iter:.1f}")

        return sca_neighborhood_allocation, L_total_cost, L_nominal_cost, L_first_cost, L_best_cost, L_avg_latency

    elif method == 'proportional_allocation':
        print("使用基于请求概率的分配方法")

        cache_nodes = topology_manager.get_cache_nodes()
        if len(cache_nodes) == 0:
            avg_latency = user_to_cache_latency + cache_to_sp_latency
            L_total_cost = [1.0] * nb_interval
            L_nominal_cost = [1.0] * nb_interval
            L_first_cost = [1.0] * nb_interval
            L_best_cost = [1.0] * nb_interval
            L_avg_latency = [avg_latency] * nb_interval
            return {}, L_total_cost, L_nominal_cost, L_first_cost, L_best_cost, L_avg_latency

        node_cache_capacity = single_cache_capacity
        proportional_allocation = {}

        for i, node in enumerate(cache_nodes):
            actual_capacity = node_cache_capacity
            new_allocation = []

            for sp in range(nSP):
                if sp < len(SP_proba):
                    allocation_ratio = SP_proba[sp]
                else:
                    allocation_ratio = 1.0 / nSP
                new_allocation.append(int(actual_capacity * allocation_ratio))

            total_allocation = sum(new_allocation)
            if total_allocation < actual_capacity:
                new_allocation[0] += actual_capacity - total_allocation

            proportional_allocation[node] = new_allocation

        print(f"初始分配: {proportional_allocation}")

        allocations_file = os.path.join(results_dir, f"allocations_real_data_single_cache_capacity{single_cache_capacity}_request_rate{request_rate}_proportional_allocation.txt")
        f = open(allocations_file, "w") if save_allocations else None

        L_total_cost = []
        L_nominal_cost = []
        L_first_cost = []
        L_best_cost = []
        L_avg_latency = []

        request_nb = int(interval_size * request_rate)

        for interval in range(nb_interval):
            current_videos_proba = catalog()

            interval_fixed_requests = fixed_requests[interval] if fixed_requests and len(fixed_requests) > interval else None

            (current_cost, _, _, current_latency, current_hit_rates) = evaluate_cost(
                proportional_allocation, proportional_allocation, best_allocation, request_nb, current_videos_proba, interval_fixed_requests
            )

            node_sp_probas = {}
            for node in cache_nodes:
                node_sp_probas[node] = []
                for sp in range(nSP):
                    prob = estimate_sp_proba(node, sp)
                    node_sp_probas[node].append(prob)

            for node in cache_nodes:
                actual_capacity = node_cache_capacity
                sp_probas = node_sp_probas[node]

                total_proba = sum(sp_probas)
                if total_proba == 0:
                    total_proba = 1.0

                new_allocation = []
                for sp in range(nSP):
                    allocation_ratio = sp_probas[sp] / total_proba
                    new_allocation.append(int(actual_capacity * allocation_ratio))

                total_allocation = sum(new_allocation)
                if total_allocation < actual_capacity:
                    new_allocation[0] += actual_capacity - total_allocation

                proportional_allocation[node] = new_allocation

            (nominal_cost, first_cost, best_cost, avg_latency, cache_node_hit_rates) = evaluate_cost(
                proportional_allocation, proportional_allocation, best_allocation, request_nb, current_videos_proba, interval_fixed_requests
            )

            L_total_cost.append(nominal_cost)
            L_nominal_cost.append(nominal_cost)
            L_first_cost.append(first_cost)
            L_best_cost.append(best_cost)
            L_avg_latency.append(avg_latency)

            if save_allocations and f:
                f.write(f"Interval {interval+1}: {proportional_allocation}\n")

        if f:
            f.close()

        print(f"基于请求概率的分配成本: {sum(L_nominal_cost) / len(L_nominal_cost):.4f}")

        return proportional_allocation, L_total_cost, L_nominal_cost, L_first_cost, L_best_cost, L_avg_latency

    elif method == 'Q_learning':
        print("使用Q-learning算法（重构版本）")
        print("  特点：基于强化学习的多缓存节点协同优化")
        print("  结构：与SCA_ADMM和SCA_neighborhood_search保持一致")
        print("  奖励：每个缓存节点的命中率变化量")
        print("  动作：单位缓存移动（coeff * delta）")
        print("  策略：ε-greedy策略，合法动作检查")

        start_time = time.time()
        cache_nodes = topology_manager.get_cache_nodes()
        num_cache_nodes = len(cache_nodes)

        q_learning_allocation = {}
        for i, node in enumerate(cache_nodes):
            actual_capacity = single_cache_capacity
            base_allocation = actual_capacity // nSP
            remainder = actual_capacity % nSP
            node_allocation = []
            for sp in range(nSP):
                if sp < remainder:
                    node_allocation.append(base_allocation + 1)
                else:
                    node_allocation.append(base_allocation)
            q_learning_allocation[node] = node_allocation

        print(f"初始分配: {q_learning_allocation}")

        allocations_file = os.path.join(results_dir, f"allocations_real_data_single_cache_capacity{single_cache_capacity}_request_rate{request_rate}_Q_learning.txt")
        f = open(allocations_file, "w") if save_allocations else None

        # 保存初始分配作为第一行
        if save_allocations and f:
            initial_allocation_for_save = {}
            for node in q_learning_allocation:
                initial_allocation_for_save[node] = [int(x) for x in q_learning_allocation[node]]
            f.write(str(initial_allocation_for_save) + '\n')

        request_nb = int(interval_size * request_rate)
        interval_fixed_requests = fixed_requests[0] if fixed_requests and len(fixed_requests) > 0 else None
        (_, fixed_first_cost, fixed_best_cost, _, _) = evaluate_cost(
            best_allocation, q_learning_allocation, best_allocation, request_nb, initial_videos_proba, interval_fixed_requests
        )

        states = {}
        state_to_index = {}
        Q = {}
        V = {}
        state_index = {}

        for node in cache_nodes:
            states[node] = states_nSP(single_cache_capacity, nSP, delta)
            state_to_index[node] = {}
            for idx, state in enumerate(states[node]):
                state_to_index[node][tuple(state)] = idx
            Q[node] = np.zeros((nSP**2, len(states[node])))
            V[node] = np.zeros((nSP**2, len(states[node])))
            original_alloc = tuple(q_learning_allocation[node])
            state_index[node] = state_to_index[node].get(original_alloc, -1)

        Memory = []

        L_total_cost = []
        L_nominal_cost = []
        L_first_cost = []
        L_best_cost = []
        L_avg_latency = []

        old_nominal_cost = None

        for interval in range(nb_interval):
            current_videos_proba = catalog()
            interval_fixed_requests = fixed_requests[interval] if fixed_requests and len(fixed_requests) > interval else None
            debug_output = (debug_interval is not None and interval == debug_interval)

            (nominal_cost, first_cost, best_cost, avg_latency, cache_node_hit_rates) = evaluate_cost(
                q_learning_allocation, q_learning_allocation, best_allocation, request_nb, current_videos_proba, interval_fixed_requests, debug_output
            )

            first_cost = fixed_first_cost
            best_cost = fixed_best_cost

            if epsilon_decay:
                epsi = find_epsilon(interval)
            else:
                epsi = 0.2

            if alpha_scheduling:
                if interval == 0:
                    alfa = 0.5
                else:
                    alfa = 0.5 * (0.99 ** interval)
                    alfa = max(alfa, 0.05)
            else:
                alfa = 0.3

            current_interval_allocation = {}
            for node in q_learning_allocation:
                current_interval_allocation[node] = q_learning_allocation[node].copy()

            old_allocation = deepcopy(q_learning_allocation)

            if interval == 0:
                (old_nominal_cost, _, _, _, old_cache_node_hit_rates) = evaluate_cost(
                    q_learning_allocation, q_learning_allocation, best_allocation, request_nb, initial_videos_proba
                )
                old_cache_node_hit_rates = old_cache_node_hit_rates.copy()

            action, action_minus, action_plus, q_learning_allocation = take_action_multi_cache(
                q_learning_allocation, epsi, D, delta, Q, state_index, cache_nodes, nSP, states, state_to_index
            )

            if save_allocations and f:
                f.write(str(q_learning_allocation) + '\n')

            (nominal_cost, _, _, avg_latency, cache_node_hit_rates) = evaluate_cost(
                q_learning_allocation, q_learning_allocation, best_allocation, request_nb, current_videos_proba, interval_fixed_requests
            )

            new_gain = {}
            if interval == 0:
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

            old_nominal_cost = nominal_cost
            old_cache_node_hit_rates = cache_node_hit_rates.copy()

            for node in cache_nodes:
                original_alloc = tuple(q_learning_allocation[node])
                state_index_prime = state_to_index[node].get(original_alloc, -1)
                if state_index_prime != -1:
                    (best_score1, best_actions1) = af.find_max_list(Q[node][:, state_index_prime])

                    act = action[node] if action and node in action else 0
                    if act is not None and act < Q[node].shape[0] and state_index[node] < Q[node].shape[1]:
                        node_reward = new_gain[node] if node in new_gain else 0.0
                        old_q = Q[node][act, state_index[node]]

                        Q[node][act, state_index[node]] += alfa * (node_reward + gama * best_score1 - Q[node][act, state_index[node]])

                        new_q = Q[node][act, state_index[node]]

                    state_index[node] = state_index_prime

            if activate_memory:
                N = find_N(interval)
                Memory.append((q_learning_allocation, action, new_gain))

                for m in range(N):
                    if Memory:
                        [state_rd, action_rd, reward_rd] = rd.choice(Memory)
                        for node in cache_nodes:
                            state_rd_index = get_state_index(state_rd[node], delta, states[node])
                            if state_rd_index != -1:
                                (best_score1, best_actions1) = af.find_max_list(Q[node][:, state_rd_index])
                                act = action_rd[node] if action_rd and node in action_rd else 0
                                if act is not None and act < Q[node].shape[0] and state_rd_index < Q[node].shape[1]:
                                    reward_val = reward_rd[node] if reward_rd and node in reward_rd else 0.0
                                    Q[node][act, state_rd_index] += alfa * (reward_val + gama * best_score1 - Q[node][act, state_rd_index])

            total_cost = nominal_cost

            L_total_cost.append(total_cost)
            L_nominal_cost.append(nominal_cost)
            L_first_cost.append(first_cost)
            L_best_cost.append(best_cost)
            L_avg_latency.append(avg_latency)

        print(f"Q-learning成本: {sum(L_nominal_cost) / len(L_nominal_cost):.4f}")

        if debug:
            total_time = time.time() - start_time
            avg_interval_time = total_time / nb_interval if nb_interval > 0 else 0
            print(f"[DEBUG] Q_learning 完成: 总用时={total_time:.2f}秒, 平均间隔用时={avg_interval_time:.2f}秒")

        if f:
            f.close()

        return q_learning_allocation, L_total_cost, L_nominal_cost, L_first_cost, L_best_cost, L_avg_latency

    else:
        print(f"未知方法: {method}")
        return allocation, [0.0] * nb_interval, [0.0] * nb_interval, [0.0] * nb_interval, [0.0] * nb_interval, [0.0] * nb_interval


def main(method=None, debug_interval=None):
    """主函数"""
    init()

    if debug_interval is None and 'debug_interval' in config['simulation']:
        debug_interval = config['simulation']['debug_interval']
        print(f"从配置文件中读取debug_interval: {debug_interval}")

    videos_proba = catalog()
    best_allocation = decide_opt_alloc(videos_proba)

    cache_nodes = topology_manager.get_cache_nodes()
    if cache_nodes:
        initial_allocation = {}
        node_cache_capacity = single_cache_capacity

        for i, node in enumerate(cache_nodes):
            node_allocation = [0] * nSP
            actual_capacity = node_cache_capacity

            avg_per_sp = actual_capacity // nSP
            rem_per_sp = actual_capacity % nSP

            for sp in range(nSP):
                node_allocation[sp] = avg_per_sp + (1 if sp < rem_per_sp else 0)

            initial_allocation[node] = node_allocation

        print(f"最佳分配: {best_allocation}")
        print(f"初始分配: {initial_allocation}")

        if should_i_simulate:
            request_rate = config['simulation']['request_rate']
            interval_size = config['simulation']['interval_size']
            delta = config['simulation']['delta']

            if method is None:
                method = config['simulation']['method']

            D = config['simulation']['D']
            nb_interval = int(simulation_time / interval_size)

            sp_names_list = real_data_config.get('sp_names', ['youtube', 'netflix', 'douyin'])
            requests_per_interval_val = real_data_config.get('requests_per_interval', 100)

            print(f"\n生成真实数据请求序列...")
            print(f"SP名称: {sp_names_list}")
            print(f"每interval请求数: {requests_per_interval_val}")
            print(f"总intervals: {nb_interval}")

            fixed_requests = create_real_data_requests(sp_names_list, requests_per_interval_val, nb_interval)
            print(f"请求序列生成完成，共 {len(fixed_requests)} 个interval")

            if isinstance(method, list):
                print("使用对比模式，同时运行多种分配策略")
                print(f"要对比的方法: {method}")

                compare_results = {}

                for compare_method in method:
                    print(f"\n{'='*50}")
                    print(f"运行方法: {compare_method}")
                    print(f"{'='*50}")

                    try:
                        result = optimize_nSP(
                            initial_allocation, videos_proba, best_allocation,
                            request_rate, nb_interval, interval_size, 0.9, delta, D, compare_method,
                            fixed_requests=fixed_requests, debug_interval=debug_interval
                        )
                        compare_results[compare_method] = result
                    except Exception as e:
                        import traceback
                        print(f"方法 {compare_method} 执行失败: {e}")
                        print(traceback.format_exc())
                        continue

                return compare_results

            else:
                print(f"运行方法: {method}")
                result = optimize_nSP(
                    initial_allocation, videos_proba, best_allocation,
                    request_rate, nb_interval, interval_size, 0.9, delta, D, method,
                    fixed_requests=fixed_requests, debug_interval=debug_interval
                )
                return result

    return None


if __name__ == "__main__":
    result = main()