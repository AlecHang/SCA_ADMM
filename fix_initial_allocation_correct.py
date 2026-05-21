#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
正确修复分配文件：生成真正的均匀初始分配并添加到文件开头
"""

import os
import ast

def generate_uniform_initial_allocation(cache_nodes, capacity, nSP):
    """生成均匀初始分配"""
    allocation = {}
    base_allocation = capacity // nSP
    remainder = capacity % nSP
    
    for node in cache_nodes:
        node_allocation = []
        for sp in range(nSP):
            if sp < remainder:
                node_allocation.append(base_allocation + 1)
            else:
                node_allocation.append(base_allocation)
        allocation[node] = node_allocation
    
    return allocation

def extract_nodes_from_file(filepath):
    """从现有分配文件中提取节点列表"""
    with open(filepath, 'r', encoding='utf-8') as f:
        first_line = f.readline()
    
    if not first_line:
        return None
    
    try:
        allocation = ast.literal_eval(first_line.strip())
        return list(allocation.keys())
    except:
        return None

def fix_initial_allocation(base_path):
    """正确修复初始分配"""
    
    methods = {
        'SCA_ADMM': 'allocations_single_cache_capacity40_request_rate100_SCA_ADMM.txt',
        'SCA_NS': 'allocations_single_cache_capacity40_request_rate100_SCA_neighborhood_search.txt',
        'Q_learning': 'allocations_single_cache_capacity40_request_rate100_Q_learning.txt'
    }
    
    cache_capacity = 40
    nSP = 3  # 从数据中可以看到每个节点有3个SP
    
    for method_name, filename in methods.items():
        filepath = os.path.join(base_path, filename)
        
        if not os.path.exists(filepath):
            print(f"警告：未找到文件 {filepath}")
            continue
        
        # 提取节点列表
        nodes = extract_nodes_from_file(filepath)
        if nodes is None:
            print(f"{method_name}：无法提取节点列表")
            continue
        
        # 生成均匀初始分配
        initial_allocation = generate_uniform_initial_allocation(nodes, cache_capacity, nSP)
        initial_line = str(initial_allocation) + '\n'
        
        # 读取现有内容
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 如果第一行已经是初始分配（通过检查是否均匀分配），跳过
        if lines:
            first_line = lines[0].strip()
            try:
                first_alloc = ast.literal_eval(first_line)
                # 检查是否已经是均匀分配
                is_uniform = True
                for node in first_alloc:
                    alloc = first_alloc[node]
                    expected = [14, 13, 13] if nSP == 3 else [cache_capacity // nSP + (1 if i < cache_capacity % nSP else 0) for i in range(nSP)]
                    if alloc != expected:
                        is_uniform = False
                        break
                
                if is_uniform:
                    print(f"{method_name}：已包含正确的初始分配，跳过")
                    continue
            except:
                pass
        
        # 检查是否有重复的第一行（之前错误的修复）
        if len(lines) >= 2 and lines[0].strip() == lines[1].strip():
            # 移除之前错误添加的行
            lines = lines[1:]
        
        # 在开头添加正确的初始分配
        new_lines = [initial_line] + lines
        
        # 写回文件
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        
        print(f"{method_name}：已修复初始分配")
        print(f"  初始分配: {initial_allocation}")

def verify_initial_allocation(base_path):
    """验证初始分配是否正确"""
    
    methods = {
        'SCA_ADMM': 'allocations_single_cache_capacity40_request_rate100_SCA_ADMM.txt',
        'SCA_NS': 'allocations_single_cache_capacity40_request_rate100_SCA_neighborhood_search.txt',
        'Q_learning': 'allocations_single_cache_capacity40_request_rate100_Q_learning.txt'
    }
    
    cache_capacity = 40
    nSP = 3
    expected_alloc = [14, 13, 13]  # 40 = 14 + 13 + 13
    
    print("\n验证结果：")
    for method_name, filename in methods.items():
        filepath = os.path.join(base_path, filename)
        
        if not os.path.exists(filepath):
            print(f"{method_name}: 未找到")
            continue
        
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if not lines:
            print(f"{method_name}: 文件为空")
            continue
        
        first_line = lines[0].strip()
        second_line = lines[1].strip() if len(lines) > 1 else ""
        
        try:
            first_alloc = ast.literal_eval(first_line)
            
            # 检查是否为均匀分配
            is_correct = True
            for node in first_alloc:
                if first_alloc[node] != expected_alloc:
                    is_correct = False
                    break
            
            # 检查第一行和第二行是否不同
            is_different = first_line != second_line
            
            if is_correct and is_different:
                print(f"{method_name}: ✓ 初始分配正确（均匀分配，与第二行不同）")
            elif is_correct:
                print(f"{method_name}: ⚠ 初始分配正确但与第二行相同")
            else:
                print(f"{method_name}: ✗ 初始分配不正确")
                
        except Exception as e:
            print(f"{method_name}: ✗ 解析失败 - {e}")

if __name__ == '__main__':
    base_path = 'results/GEANT_1/stability'
    
    print(f"处理目录: {base_path}")
    fix_initial_allocation(base_path)
    verify_initial_allocation(base_path)