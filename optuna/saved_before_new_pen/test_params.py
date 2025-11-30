#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 num_envs 和 horizon_length 参数调用链的正确性
"""

import sys
import os
import hydra
from omegaconf import DictConfig, OmegaConf

# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

@hydra.main(config_path="../configs", config_name="config", version_base="1.1")
def test_params(cfg: DictConfig):
    """测试参数调用链"""
    print("\n" + "="*80)
    print("测试参数调用链")
    print("="*80)
    
    # 测试 task.env.numEnvs
    print(f"\n1. task.env.numEnvs:")
    print(f"   值: {cfg.task.env.numEnvs}")
    print(f"   类型: {type(cfg.task.env.numEnvs)}")
    
    # 测试 train.ppo.num_actors (应该引用 task.env.numEnvs)
    print(f"\n2. train.ppo.num_actors:")
    print(f"   值: {cfg.train.ppo.num_actors}")
    print(f"   类型: {type(cfg.train.ppo.num_actors)}")
    print(f"   是否等于 numEnvs: {cfg.train.ppo.num_actors == cfg.task.env.numEnvs}")
    
    # 测试 train.ppo.horizon_length
    print(f"\n3. train.ppo.horizon_length:")
    print(f"   值: {cfg.train.ppo.horizon_length}")
    print(f"   类型: {type(cfg.train.ppo.horizon_length)}")
    
    # 计算 batch_size
    batch_size = cfg.train.ppo.horizon_length * cfg.train.ppo.num_actors
    print(f"\n4. 计算 batch_size:")
    print(f"   batch_size = horizon_length × num_actors")
    print(f"   {batch_size} = {cfg.train.ppo.horizon_length} × {cfg.train.ppo.num_actors}")
    
    # 测试参数覆盖
    print("\n" + "="*80)
    print("测试参数覆盖 (task.env.numEnvs=4096, train.ppo.horizon_length=16)")
    print("="*80)
    print("\n完整配置:")
    print(OmegaConf.to_yaml(cfg))
    
    return True

if __name__ == "__main__":
    # 测试1: 默认配置
    print("\n测试1: 默认配置")
    test_params()
    
    # 测试2: 覆盖参数
    print("\n\n测试2: 覆盖参数")
    sys.argv = [
        "test_params.py",
        "task=LinkerHandHora",
        "task.env.numEnvs=4096",
        "train.ppo.horizon_length=16"
    ]
    test_params()
