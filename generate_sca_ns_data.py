#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成SCA_NS方法的分配历史数据
"""

import os
import sys
from pathlib import Path

# 添加Simulations目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Simulations'))

import numpy as np
import simulation_code as sc

def generate_sca_ns_data():
    """
    生成SCA_NS方法的分配历史数据
    """
    print("="*80)
    print("生成 SCA_NS 方法的分配历史数据")
    print("="*80)
    
    # 使用optimize_nSP函数运行SCA_neighborhood_search方法
    # 这个函数会返回分配历史记录
    allocation_history = sc.optimize_nSP(
        method='SCA_neighborhood_search',
        debug_interval=None
    )
    
    # 获取配置参数
    config = sc.config
    single_cache_capacity = config['simulation']['single_cache_capacity']
    request_rate = config['simulation']['request_rate']
    
    # 保存分配历史
    output_file = f"results/allocations_single_cache_capacity{single_cache_capacity}_request_rate{request_rate}_SCA_NS.txt"
    with open(output_file, 'w') as f:
        for alloc in allocation_history:
            f.write(f"{alloc}\n")
    
    print(f"\nSCA_NS 分配历史已保存到: {output_file}")
    print(f"共记录了 {len(allocation_history)} 个分配状态")
    print("\n" + "="*80)

if __name__ == '__main__':
    generate_sca_ns_data()