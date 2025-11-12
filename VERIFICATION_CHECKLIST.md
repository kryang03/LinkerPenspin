# 功能验证清单

## 在运行训练之前，请确认以下检查项

### 环境准备
- [ ] IsaacGym已安装
- [ ] 所有Python依赖已安装（见requirements.txt）
- [ ] CUDA和GPU驱动正常
- [ ] 有足够的磁盘空间存储checkpoints和日志

### 初始化抓取状态
- [ ] cache/3pose/pencil/ 目录存在且包含初始化文件
- [ ] linker_hand_hora.py中的CHECKLIST已根据需要调整
- [ ] 确认grasp_cache_name设置正确（默认：3pose）

### 功能1测试：纯PPO Teacher训练

#### 快速测试（小规模）
```bash
# 用少量环境快速测试流程
bash scripts/train_rl_teacher.sh 0 42 quick_test \
  task.env.numEnvs=512 \
  train.ppo.max_agent_steps=100000
```

#### 检查点
- [ ] 训练启动无错误
- [ ] outputs/LinkerHandHora/quick_test/teacher_nn/ 目录已创建
- [ ] outputs/LinkerHandHora/quick_test/teacher_tb/ 目录已创建
- [ ] TensorBoard显示正常：`tensorboard --logdir outputs/LinkerHandHora/quick_test/teacher_tb`
- [ ] 观察指标：
  - [ ] episode_rewards/step 在更新
  - [ ] losses/actor_loss 有合理值
  - [ ] losses/critic_loss 有合理值
  - [ ] info/kl 在合理范围
- [ ] checkpoint正常保存（last.pth）

#### 完整训练（如快速测试通过）
```bash
bash scripts/train_rl_teacher.sh 0 42 teacher_baseline
```

---

### 功能2测试：RL+BC Teacher训练（可选）

#### 前置条件
- [ ] 功能1已完成训练，有可用的Teacher checkpoint
- [ ] 记录Teacher模型路径，例如：
  ```
  outputs/LinkerHandHora/teacher_baseline/teacher_nn/best_reward_10.50.pth
  ```

#### 快速测试
```bash
# 替换<TEACHER_PATH>为实际路径
bash scripts/train_rl_bc_teacher.sh 0 42 quick_test_bc_teacher \
  outputs/LinkerHandHora/teacher_baseline/teacher_nn/best_reward_xxx.pth \
  task.env.numEnvs=512 \
  train.ppo.max_agent_steps=100000
```

#### 检查点
- [ ] Teacher模型成功加载（查看日志中的"loading demonstration checkpoint"）
- [ ] 训练启动无错误
- [ ] outputs/LinkerHandHora/quick_test_bc_teacher/stage1_nn/ 目录已创建
- [ ] TensorBoard显示正常
- [ ] 观察指标：
  - [ ] losses/bc_loss 不为0且有合理值
  - [ ] 其他PPO loss正常
- [ ] checkpoint正常保存

#### 完整训练（如快速测试通过）
```bash
bash scripts/train_rl_bc_teacher.sh 0 42 teacher_finetuned \
  outputs/LinkerHandHora/teacher_baseline/teacher_nn/best_reward_xxx.pth
```

---

### 功能3测试：RL+BC Student训练

#### 前置条件
- [ ] 功能1已完成训练，有可用的Teacher checkpoint
- [ ] 确认将使用的Teacher模型路径

#### 快速测试
```bash
# 替换<TEACHER_PATH>为实际路径
bash scripts/train_rl_bc_student.sh 0 42 quick_test_student \
  outputs/LinkerHandHora/teacher_baseline/teacher_nn/best_reward_xxx.pth \
  task.env.numEnvs=512 \
  train.ppo.max_agent_steps=100000
```

#### 检查点
- [ ] Teacher模型成功加载
- [ ] 训练启动无错误
- [ ] Student使用的是proprio_hist输入（查看日志确认）
- [ ] outputs/LinkerHandHora/quick_test_student/stage1_nn/ 目录已创建
- [ ] TensorBoard显示正常
- [ ] 观察指标：
  - [ ] losses/bc_loss 不为0且有合理值
  - [ ] episode_rewards 逐渐上升
- [ ] checkpoint正常保存

#### 完整训练（如快速测试通过）
```bash
bash scripts/train_rl_bc_student.sh 0 42 student_deploy \
  outputs/LinkerHandHora/teacher_baseline/teacher_nn/best_reward_xxx.pth
```

---

## 常见问题排查

### 问题1: 找不到模型类
**错误**: `NameError: name 'PPOTeacher' is not defined`

**解决**:
```bash
# 检查train.py中的导入
grep "from penspin.algo.ppo" train.py
# 应该能看到所有三个类的导入
```

### 问题2: demon_path参数缺失
**错误**: `必须提供DEMON_PATH参数`

**解决**:
- 确保在命令中提供了第4个参数（Teacher模型路径）
- 检查路径是否存在且可访问

### 问题3: bc_loss_coef未定义
**错误**: `KeyError: 'bc_loss_coef'`

**解决**:
- 检查configs/train/LinkerHandHora.yaml是否包含bc_loss_coef
- 或者在命令行中显式指定：`train.ppo.bc_loss_coef=1.0`

### 问题4: CUDA OOM
**错误**: `CUDA out of memory`

**解决**:
```bash
# 减少环境数量
task.env.numEnvs=4096  # 或更小

# 减少batch size
train.ppo.minibatch_size=8192  # 或更小

# 减少horizon length
train.ppo.horizon_length=8  # 或更小
```

### 问题5: 训练不收敛
**现象**: reward长时间不增长

**检查**:
- [ ] 学习率是否合适（默认5e-3）
- [ ] bc_loss_coef是否过大（尝试0.5-2.0）
- [ ] 是否有足够的探索（检查entropy）
- [ ] 环境奖励函数是否合理

---

## 性能基准

### 预期训练时间（8192环境）
- **功能1 (PPO Teacher)**: 
  - FPS: ~5000-8000
  - 达到合理性能: ~500M-1B steps
  
- **功能2 (RL+BC Teacher)**:
  - FPS: ~4000-6000 (稍慢因为Teacher推理)
  - 达到合理性能: ~200M-500M steps
  
- **功能3 (RL+BC Student)**:
  - FPS: ~5000-7000
  - 达到合理性能: ~300M-800M steps

### 预期奖励范围
- 初始: -5 ~ 0
- 中期: 0 ~ 5
- 良好: 5 ~ 10
- 优秀: >10

（具体值取决于奖励函数设计）

---

## 部署前测试

### 测试Teacher模型
```bash
python train.py task=LinkerHandHora \
  train.algo=PPOTeacher \
  test=True \
  headless=False \
  train.load_path=outputs/LinkerHandHora/teacher_baseline/teacher_nn/best_reward_xxx.pth \
  task.env.numEnvs=16
```

### 测试Student模型
```bash
python train.py task=LinkerHandHora \
  train.algo=PPO_RL_BC_Student \
  test=True \
  headless=False \
  train.load_path=outputs/LinkerHandHora/student_deploy/stage1_nn/best.pth \
  task.env.numEnvs=16
```

### 可视化检查
- [ ] 手指运动流畅
- [ ] 物体稳定抓握
- [ ] 转笔动作连贯
- [ ] 无异常碰撞或抖动

---

## 完成标准

### 功能1完成标准
- [x] 训练至少500M steps
- [x] episode_reward稳定在合理范围
- [x] 保存了best checkpoint
- [x] 测试模式运行正常

### 功能2完成标准（如使用）
- [x] 成功加载Teacher模型
- [x] bc_loss正常计算
- [x] 性能达到或超过功能1
- [x] 保存了best checkpoint

### 功能3完成标准
- [x] 成功加载Teacher模型
- [x] Student仅使用proprio_hist
- [x] bc_loss正常计算
- [x] 性能接近Teacher（可能略低）
- [x] 测试模式运行正常
- [x] 可以部署到真机测试

---

## 文档参考

训练过程中可参考：
- `QUICKSTART.md` - 快速命令参考
- `TRAINING_GUIDE.md` - 详细训练说明
- `MODIFICATION_SUMMARY.md` - 代码修改记录

## 最后检查

在开始大规模训练之前：
- [ ] 所有快速测试都通过
- [ ] TensorBoard日志正常
- [ ] 磁盘空间充足
- [ ] GPU监控正常（nvidia-smi）
- [ ] 确定训练计划和预期时间
- [ ] 设置好日志和checkpoint备份

祝训练顺利！🚀
