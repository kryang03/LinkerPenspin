#!/usr/bin/env python
"""快速测试Hydra配置是否正常工作"""

import os
import sys

# 切换到项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

print(f"当前工作目录: {os.getcwd()}")
print(f"Python路径: {sys.path[:3]}")

# 测试导入train模块
try:
    import train
    print("✅ train模块导入成功")
except Exception as e:
    print(f"❌ train模块导入失败: {e}")
    sys.exit(1)

# 测试Hydra配置
try:
    import hydra
    from omegaconf import DictConfig
    
    @hydra.main(config_name='config', config_path='configs', version_base=None)
    def test_config(cfg: DictConfig):
        print("✅ Hydra配置加载成功")
        print(f"任务名称: {cfg.task_name}")
        return True
    
    # 模拟命令行参数
    sys.argv = ["test.py", "task=LinkerHandHora"]
    result = test_config()
    
    if result:
        print("\n" + "="*60)
        print("✅ 所有测试通过！Hydra配置正常工作")
        print("="*60)
    
except Exception as e:
    print(f"❌ Hydra配置测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
