"""
拓扑管理模块
支持不同类型的拓扑结构，包括树型、路径型、环型、网状型等
"""

import networkx as nx
import random
import os

class TopologyManager:
    """
    拓扑管理类，用于创建和管理不同类型的拓扑结构
    """
    
    def __init__(self, topology_type, parameters, cache_nodes_count, sp_nodes_count, router_nodes_count=None):
        """
        初始化拓扑管理器

        参数:
        topology_type: 拓扑类型 (TREE, PATH, RING, MESH)
        parameters: 拓扑参数
        cache_nodes_count: 缓存节点数目或比例（如果是小数且在0-1之间，则为比例）
        sp_nodes_count: SP节点数目
        router_nodes_count: 路由器节点数目，默认为None，表示剩余的节点全部作为路由器节点
        """
        self.topology_type = topology_type
        self.parameters = parameters
        self.cache_nodes_ratio = None

        # 检查cache_nodes_count是否是比例（0-1之间的小数）
        if isinstance(cache_nodes_count, float) and 0 < cache_nodes_count <= 1:
            self.cache_nodes_ratio = cache_nodes_count
            self.cache_nodes_count = None  # 稍后在创建拓扑后计算
        else:
            self.cache_nodes_count = int(cache_nodes_count)
            self.cache_nodes_ratio = None

        self.sp_nodes_count = sp_nodes_count
        self.router_nodes_count = router_nodes_count
        self.graph = None
        self.cache_nodes = []
        self.sp_nodes = []
        self.router_nodes = []
        self.receiver_nodes = []
        
    def create_topology(self):
        """
        创建拓扑结构
        """
        if self.topology_type == "TREE":
            self._create_tree_topology()
        elif self.topology_type == "PATH":
            self._create_path_topology()
        elif self.topology_type == "RING":
            self._create_ring_topology()
        elif self.topology_type == "MESH":
            self._create_mesh_topology()
        elif self.topology_type == "GEANT":
            self._load_real_topology("Geant2012.graphml")
        elif self.topology_type == "TISCALI":
            self._load_real_topology("3257.r0.cch")
        elif self.topology_type == "WIDE":
            self._load_real_topology("WideJpn.graphml")
        elif self.topology_type == "GARR":
            self._load_real_topology("Garr201201.graphml")
        else:
            raise ValueError(f"不支持的拓扑类型: {self.topology_type}")
        
        # 分配节点类型
        self._assign_node_types()
        
    def _create_tree_topology(self):
        """
        创建树型拓扑
        """
        k = self.parameters.get('k', 2)  # 分支因子
        h = self.parameters.get('h', 3)  # 高度
        
        # 计算节点总数
        total_nodes = sum(k**i for i in range(h+1))
        
        # 创建树型图
        self.graph = nx.DiGraph()
        
        # 添加节点和边
        node_id = 0
        for level in range(h+1):
            nodes_in_level = k**level
            for i in range(nodes_in_level):
                self.graph.add_node(node_id)
                # 添加父节点连接
                if level > 0:
                    parent_id = (node_id - nodes_in_level) // k
                    self.graph.add_edge(parent_id, node_id)
                node_id += 1
        
    def _create_path_topology(self):
        """
        创建路径型拓扑
        """
        # 处理 router_nodes_count 为 None 的情况
        router_nodes = self.router_nodes_count if self.router_nodes_count is not None else 0
        n = self.cache_nodes_count + self.sp_nodes_count + router_nodes + 1  # 加上一个接收器节点
        self.graph = nx.path_graph(n)
        
    def _create_ring_topology(self):
        """
        创建环型拓扑
        """
        # 处理 router_nodes_count 为 None 的情况
        router_nodes = self.router_nodes_count if self.router_nodes_count is not None else 0
        n = self.cache_nodes_count + self.sp_nodes_count + router_nodes
        self.graph = nx.cycle_graph(n)
        
    def _create_mesh_topology(self):
        """
        创建网状拓扑
        """
        # 处理 router_nodes_count 为 None 的情况
        router_nodes = self.router_nodes_count if self.router_nodes_count is not None else 0
        n = self.cache_nodes_count + self.sp_nodes_count + router_nodes
        
        # 创建一个随机正则图，每个节点有4个邻居（更合理的网状拓扑）
        # 确保n >= 4，否则使用完全图
        if n >= 4:
            # 每个节点有4个邻居
            self.graph = nx.random_regular_graph(4, n)
        else:
            # 节点数不足时使用完全图
            self.graph = nx.complete_graph(n)
        
    def _load_real_topology(self, filename):
        """
        加载真实网络拓扑
        
        参数:
        filename: 拓扑文件名
        """
        # 构建拓扑文件路径
        icarus_topology_dir = os.path.join(os.path.dirname(__file__), "..", "..", "icarus-master", "resources", "topologies")
        topology_path = os.path.join(icarus_topology_dir, filename)
        
        # 检查文件是否存在
        if not os.path.exists(topology_path):
            print(f"警告: 拓扑文件不存在: {topology_path}")
            print("使用默认的 MESH 拓扑作为替代")
            # 使用默认的 MESH 拓扑
            self._create_mesh_topology()
            return
        
        # 加载拓扑
        if filename.endswith(".graphml"):
            # 加载 GraphML 格式的拓扑
            self.graph = nx.read_graphml(topology_path)
        elif filename.endswith(".cch"):
            # 加载 RocketFuel 格式的拓扑
            self.graph = self._parse_rocketfuel_topology(topology_path)
        else:
            print(f"警告: 不支持的文件格式: {filename}")
            print("使用默认的 MESH 拓扑作为替代")
            self._create_mesh_topology()
            return
        
        # 转换为无向图
        self.graph = self.graph.to_undirected()
        
        # 获取最大连通分量
        if not nx.is_connected(self.graph):
            largest_cc = max(nx.connected_components(self.graph), key=len)
            self.graph = self.graph.subgraph(largest_cc).copy()
        
        # 重新映射节点ID为整数
        mapping = {node: i for i, node in enumerate(self.graph.nodes())}
        self.graph = nx.relabel_nodes(self.graph, mapping)
        
    def _parse_rocketfuel_topology(self, file_path):
        """
        解析 RocketFuel 格式的拓扑文件
        
        参数:
        file_path: RocketFuel 格式文件的路径
        
        返回:
        nx.Graph: 解析后的图
        """
        graph = nx.Graph()
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # 解析行 - 先按制表符分割
                parts = line.split('\t')
                if len(parts) < 1:
                    continue
                
                # 提取节点ID和连接
                node_info = parts[0]
                
                # 提取节点ID - 第一个空格之前的部分
                node_id_str = node_info.split(' ')[0]
                try:
                    node_id = int(node_id_str)
                except ValueError:
                    continue
                
                # 添加节点
                graph.add_node(node_id)
                
                # 查找连接部分
                connections = ''
                for part in parts:
                    if '->' in part:
                        connections = part
                        break
                
                # 提取连接的节点
                if '->' in connections:
                    # 提取 -> 后面的部分，直到 = 号
                    connected_part = connections.split('->')[1]
                    if '=' in connected_part:
                        connected_part = connected_part.split('=')[0]
                    
                    # 提取所有 <node_id> 格式的节点
                    import re
                    connected_nodes = re.findall(r'<(\d+)>', connected_part)
                    
                    # 添加边
                    for node_str in connected_nodes:
                        try:
                            connected_node_id = int(node_str)
                            graph.add_edge(node_id, connected_node_id)
                        except ValueError:
                            continue
        
        # 如果图为空，返回一个简单的 mesh 拓扑
        if graph.number_of_nodes() == 0:
            print("警告: 解析 RocketFuel 拓扑失败，图为空")
            temp_graph = nx.Graph()
            # 添加一些默认节点和边
            for i in range(10):
                temp_graph.add_node(i)
            for i in range(9):
                temp_graph.add_edge(i, i+1)
            return temp_graph
        
        return graph
        
    def _assign_node_types(self):
        """
        分配节点类型
        """
        all_nodes = list(self.graph.nodes())
        random.shuffle(all_nodes)

        # 如果使用了比例参数，则根据拓扑节点总数计算缓存节点数量
        if self.cache_nodes_ratio is not None:
            # 计算缓存节点数量：总节点数 * 比例
            total_nodes = len(all_nodes)
            self.cache_nodes_count = max(1, int(total_nodes * self.cache_nodes_ratio))

        # 分配SP节点
        self.sp_nodes = all_nodes[:self.sp_nodes_count]
        remaining_nodes = all_nodes[self.sp_nodes_count:]

        # 分配缓存节点
        self.cache_nodes = remaining_nodes[:self.cache_nodes_count]
        remaining_nodes = remaining_nodes[self.cache_nodes_count:]

        # 分配路由器节点
        if self.router_nodes_count is None:
            # 剩余的节点全部作为路由器节点
            self.router_nodes = remaining_nodes
            self.receiver_nodes = []
        else:
            self.router_nodes = remaining_nodes[:self.router_nodes_count]
            remaining_nodes = remaining_nodes[self.router_nodes_count:]

            # 剩余节点作为接收器节点
            self.receiver_nodes = remaining_nodes
        
    def get_cache_nodes(self):
        """
        获取缓存节点列表
        """
        return self.cache_nodes
    
    def get_sp_nodes(self):
        """
        获取SP节点列表
        """
        return self.sp_nodes
    
    def get_router_nodes(self):
        """
        获取路由器节点列表
        """
        return self.router_nodes
    
    def get_receiver_nodes(self):
        """
        获取接收器节点列表
        """
        return self.receiver_nodes
    
    def get_shortest_path(self, source, target):
        """
        获取两个节点之间的最短路径
        """
        try:
            return nx.shortest_path(self.graph, source, target)
        except nx.NetworkXNoPath:
            return None
    
    def get_path_length(self, source, target):
        """
        获取两个节点之间的路径长度
        """
        try:
            return nx.shortest_path_length(self.graph, source, target)
        except nx.NetworkXNoPath:
            return float('inf')
    
    def get_latency(self, source, target):
        """
        获取两个节点之间的时延
        
        Args:
            source: 源节点
            target: 目标节点
            
        Returns:
            两个节点之间的时延（毫秒）
        """
        path_length = self.get_path_length(source, target)
        # 获取链路时延参数，默认为1毫秒
        link_delay = self.parameters.get('delay', 1)
        # 总时延 = 路径长度 * 链路时延
        return path_length * link_delay
    
    def get_neighbors(self, node):
        """
        获取节点的邻居列表
        """
        return list(self.graph.neighbors(node))
    
    def visualize(self, fig=None, show=True):
        """
        可视化拓扑结构
        
        Args:
            fig: 可选的 matplotlib Figure 对象，如果提供则使用它，否则创建新的
            show: 是否显示图表，默认为 True
        """
        import matplotlib.pyplot as plt
        
        # 为不同类型的节点设置不同的颜色
        node_colors = []
        for node in self.graph.nodes():
            if node in self.sp_nodes:
                node_colors.append('red')  # SP node
            elif node in self.cache_nodes:
                node_colors.append('green')  # Cache node
            elif node in self.router_nodes:
                node_colors.append('blue')  # Router node
            else:
                node_colors.append('gray')  # Receiver node
        
        # 绘制拓扑
        if fig is None:
            plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, node_color=node_colors, with_labels=True, node_size=500)
        plt.title(f"{self.topology_type} Topology")
        
        # 添加图例
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='SP Node'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Cache Node'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Router Node'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Receiver Node')
        ]
        plt.legend(handles=legend_elements, loc='best')
        
        if show:
            plt.show()
