#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复分配文件：将初始分配添加到文件开头
"""

import os
import ast

def add_initial_allocation(base_path):
    """为所有方法的分配文件添加初始分配"""
    
    methods = {
        'SCA_ADMM': 'allocations_single_cache_capacity40_request_rate100_SCA_ADMM.txt',
        'SCA_NS': 'allocations_single_cache_capacity40_request_rate100_SCA_neighborhood_search.txt',
        'Q_learning': 'allocations_single_cache_capacity40_request_rate100_Q_learning.txt'
    }
    
    for method_name, filename in methods.items():
        filepath = os.path.join(base_path, filename)
        
        if not os.path.exists(filepath):
            print(f"警告：未找到文件 {filepath}")
            continue
        
        # 读取现有分配
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if not lines:
            print(f"警告：文件为空 {filepath}")
            continue
        
        # 获取第一行作为初始分配（实际上是第一个interval后的分配）
        first_allocation_str = lines[0].strip()
        
        try:
            first_allocation = ast.literal_eval(first_allocation_str)
            
            # 检查是否已经有初始分配（通过检查是否有两行相同的开头）
            if len(lines) >= 2:
                second_allocation_str = lines[1].strip()
                if first_allocation_str == second_allocation_str:
                    print(f"{method_name}：已包含初始分配，跳过")
                    continue
            
            # 在文件开头添加初始分配（复制第一行）
            new_lines = [first_allocation_str + '\n'] + lines
            
            # 写回文件
            with open(filepath, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
            
            print(f"{method_name}：已添加初始分配到文件开头")
            
        except Exception as e:
            print(f"{method_name}：解析失败 - {e}")

def verify_initial_allocation(base_path):
    """验证初始分配是否正确添加"""
    
    methods = {
        'SCA_ADMM': 'allocations_single_cache_capacity40_request_rate100_SCA_ADMM.txt',
        'SCA_NS': 'allocations_single_cache_capacity40_request_rate100_SCA_neighborhood_search.txt',
        'Q_learning': 'allocations_single_cache_capacity40_request_rate100_Q_learning.txt'
    }
    
    print("\n验证结果：")
    for method_name, filename in methods.items():
        filepath = os.path.join(base_path, filename)
        
        if not os.path.exists(filepath):
            print(f"{method_name}: 未找到")
            continue
        
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if len(lines) >= 2:
            first = lines[0].strip()
            second = lines[1].strip()
            
            if first == second:
                print(f"{method_name}: ✓ 初始分配已添加（第1行和第2行相同）")
            else:
                print(f"{method_name}: ✗ 初始分配未正确添加")
        else:
            print(f"{method_name}: ✗ 文件行数不足")

if __name__ == '__main__':
    # 处理GEANT_1/stability目录下的文件
    base_path = 'results/GEANT_1/stability'
    
    print(f"处理目录: {base_path}")
    add_initial_allocation(base_path)
    verify_initial_allocation(base_path)
    
    # 也处理results目录下的文件
    base_path2 = 'results'
    print(f"\n处理目录: {base_path2}")
    add_initial_allocation(base_path2)
    verify_initial_allocation(base_path2)