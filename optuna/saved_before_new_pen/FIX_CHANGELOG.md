# HPO 系统修复记录

## 修复的问题

### 1. Isaac Gym Foundation 重复创建问题 ✅

**问题描述**：
- 在运行第二个试验时出现段错误（核心已转储）
- 错误信息：`Foundation object exists already. Only one instance per process can be created.`
- 退出码：139

**根本原因**：
Isaac Gym/PhysX Foundation 是全局单例对象，在同一个 Python 进程中不能重复创建。原始实现在同一进程中运行所有试验，导致第二个试验尝试重新初始化 Isaac Gym 时崩溃。

**解决方案**：
修改 `optuna/tune_teacher.py`，使用 `subprocess` 让每个试验在独立的子进程中运行：

```python
# 旧方案（有问题）
import train
composite_score = train.main()  # 在同一进程中运行

# 新方案（已修复）
result = subprocess.run(
    ["python", "train.py"] + args,
    capture_output=True,
    text=True
)
# 从输出中提取评分
```

**优势**：
- ✅ 完全隔离每个试验，避免资源冲突
- ✅ 每个试验结束后自动清理 GPU 内存
- ✅ 更稳定，一个试验崩溃不影响其他试验
- ✅ 支持真正的多GPU并行（之前可能有冲突）

---

### 2. Hydra 配置模块未找到 ✅

**问题描述**：
```
Primary config module 'configs' not found.
Check that it's correct and contains an __init__.py file
```

**解决方案**：
创建 `configs/__init__.py` 文件，让 Python 识别 configs 为模块。

---

### 3. 数据库中失败试验导致的错误 ✅

**问题描述**：
```
ValueError: Record does not exist.
```
在尝试访问 `study.best_value` 时出现。

**解决方案**：
- 修改代码添加错误处理，检查是否有成功完成的试验
- 提供清理工具：`optuna/clean_failed_trials.py`

---

## 更新的文件

### 核心文件

1. **optuna/tune_teacher.py** - 主要修改
   - 使用 subprocess 运行每个试验
   - 从子进程输出中提取评分
   - 添加更完善的错误处理

2. **train.py** - 小修改
   - 在结束时输出 `OPTUNA_SCORE: <score>` 供父进程读取

3. **configs/__init__.py** - 新增
   - 空文件，让 Python 识别 configs 为模块

### 工具脚本

4. **optuna/clean_failed_trials.py** - 新增
   - 清理数据库中失败的试验

5. **optuna/test_foundation_fix.sh** - 新增
   - 快速测试修复是否有效（运行2个试验）

6. **optuna/verify_setup.sh** - 新增
   - 验证系统配置是否正确

### 文档

7. **optuna/TROUBLESHOOTING.md** - 更新
   - 添加 Foundation 问题的说明
   - 添加其他常见问题的解决方案

---

## 验证修复

### 快速测试（推荐）

```bash
# 运行2个快速试验验证修复
bash optuna/test_foundation_fix.sh
```

预期结果：
- 第一个试验完成
- 第二个试验也能正常完成（之前会崩溃）
- 退出码为0

### 完整测试

```bash
# 验证系统配置
bash optuna/verify_setup.sh

# 运行实际优化（3-5个试验测试）
bash optuna/run_hpo.sh 0 5 10000000
```

---

## 性能影响

### 子进程方案 vs 同进程方案

**优点**：
- ✅ 完全避免 Foundation 冲突
- ✅ 更好的内存管理（每个试验结束后完全释放）
- ✅ 更稳定（隔离故障）
- ✅ 真正支持多GPU并行

**缺点**：
- ⚠️ 启动开销略大（每个试验需要启动新进程）
- ⚠️ 无法共享内存（但对于 Isaac Gym 来说不适用）

**实际影响**：
- 启动开销：~5-10秒/试验（相比1-3小时的训练时间可忽略）
- 内存使用：更好（每次完全清理）
- 稳定性：显著提升

---

## 使用指南

### 首次使用

```bash
# 1. 验证配置
bash optuna/verify_setup.sh

# 2. 测试修复
bash optuna/test_foundation_fix.sh

# 3. 如果测试通过，开始正式优化
bash optuna/run_hpo.sh 0 50
```

### 继续之前的优化

```bash
# 如果之前有失败的试验，先清理
python optuna/clean_failed_trials.py --dry_run  # 查看
python optuna/clean_failed_trials.py            # 清理

# 继续优化
bash optuna/run_hpo.sh 0 20
```

### 多GPU并行

现在完全支持真正的多GPU并行：

```bash
# 终端1 - GPU 0
bash optuna/run_hpo.sh 0 25

# 终端2 - GPU 1
bash optuna/run_hpo.sh 1 25

# 终端3 - GPU 2
bash optuna/run_hpo.sh 2 25
```

每个GPU一个进程，互不干扰。

---

## 常见问题

### Q: 为什么不在同一进程中清理 Isaac Gym？

A: Isaac Gym/PhysX 的 Foundation 对象设计为全局单例，没有提供官方的"清理并重新初始化"API。尝试手动清理可能导致其他问题。

### Q: 子进程方案会影响性能吗？

A: 影响微乎其微。进程启动开销（~5-10秒）相比训练时间（1-3小时）可忽略不计。

### Q: 可以共享GPU吗？

A: 不建议。每个GPU运行一个优化进程即可。如果GPU内存充足，理论上可以，但可能导致资源竞争。

### Q: 如何调试子进程中的错误？

A: 查看训练输出目录中的日志：
```bash
tail -f outputs/LinkerHandHora/optuna_trial_XXXX/training.log
```

---

## 技术细节

### 进程间通信

父进程（Optuna）通过解析子进程的标准输出获取评分：

```python
# 子进程（train.py）输出
print(f"OPTUNA_SCORE: {composite_score}")

# 父进程解析
for line in result.stdout.split('\n'):
    if line.startswith('OPTUNA_SCORE:'):
        composite_score = float(line.split(':')[1].strip())
```

### 环境变量传递

父进程设置 CUDA_VISIBLE_DEVICES 确保子进程使用正确的GPU：

```python
env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
subprocess.run(..., env=env)
```

---

## 总结

通过将每个试验隔离到独立的子进程中，我们：

1. ✅ 完全解决了 Isaac Gym Foundation 重复创建问题
2. ✅ 提高了系统稳定性
3. ✅ 改善了内存管理
4. ✅ 支持真正的多GPU并行优化
5. ✅ 性能开销几乎可以忽略

系统现在已经可以稳定地运行大规模超参数优化了！🎉

---

## 更新日期

2025年11月18日

## 版本

v2.0 - Foundation 问题修复版
