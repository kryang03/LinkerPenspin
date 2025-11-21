# 参数清理记录

**清理日期**: 2025年11月17日  
**清理目的**: 简化代码架构，删除冗余的参数传递，提高代码可维护性

---

## 一、概述

本次清理主要解决两类问题：
1. **Scripts中的通用参数冗余**: `.sh`文件中包含大量不常改变的通用配置参数，应统一在`configs`中定义
2. **未使用的参数**: 大量参数在代码中初始化但从未实际使用

---

## 二、从Scripts移至Configs的通用参数

### 2.1 修改的配置文件

#### `configs/task/LinkerHandHora.yaml`
以下参数从scripts移至configs作为默认值：

| 参数名 | 修改前 | 修改后 | 说明 |
|--------|--------|--------|------|
| `rotation_axis` | `'-z'` | `'+z'` | 旋转轴方向，统一使用+z |
| `genGraspCategory` | `'hora'` | `'pencil'` | 抓取类别，统一使用pencil |
| `forceScale` | `0.0` | `2.0` | 力的缩放比例 |
| `randomForceProbScalar` | `0.0` | `0.25` | 随机力概率标量 |
| `episodeLength` | `400` | `400` (添加注释) | 训练时400，测试时4000 |
| `object.type` | `'cylinder_pencil-5-7'` | 保持不变 | 物体类型 |
| `hora.point_cloud_sampled_dim` | `0` | `100` | 点云采样维度，训练使用100 |

#### `configs/train/LinkerHandHora.yaml`
| 参数名 | 修改前 | 修改后 | 说明 |
|--------|--------|--------|------|
| `horizon_length` | `8` | `12` | PPO训练的horizon长度 |

### 2.2 简化的Scripts文件

#### `scripts/train_rl_teacher.sh`
**删除的参数**（已在configs中定义）:
- `train.ppo.minibatch_size=16384`
- `train.ppo.horizon_length=12`
- `task.env.numEnvs=8192`
- `task.env.object.type=cylinder_pencil-5-7`
- `task.env.randomForceProbScalar=0.25`
- `task.env.rotation_axis=+z`
- `task.env.genGraspCategory=pencil`
- `task.env.privInfo.enable_obj_orientation=True`
- `task.env.privInfo.enable_ft_pos=True`
- `task.env.privInfo.enable_obj_angvel=True`
- `task.env.privInfo.enable_tactile=True`
- `task.env.randomization.randomizeScaleList=[0.3]`
- `train.ppo.priv_info=True`
- `task.env.hora.point_cloud_sampled_dim=100`
- `task.env.numObservations=126`
- `task.env.forceScale=2.0`
- `task.env.enable_obj_ends=True`

**保留的参数**（经常变化）:
- `train.algo=PPOTeacher` (不同训练阶段需要改变)
- `task.env.grasp_cache_name=3pose` (可能使用不同pose)
- `task.env.reward.angvelClipMax/Min` (reward调优参数)
- `task.env.reward.angvelPenaltyThres` (reward调优参数)
- `train.ppo.max_agent_steps` (训练步数控制)
- `task.env.initPoseMode=low` (初始化模式可能改变)
- `task.env.reset_height_threshold=0.12` (可能需要调整)

#### `scripts/train_rl_bc_teacher.sh`
删除的参数与`train_rl_teacher.sh`类似，另外保留：
- `train.demon_path` (Teacher模型路径)
- `train.ppo.bc_loss_coef=1.0` (BC损失系数)
- `train.ppo.enable_latent_loss=False` (是否启用latent loss)

#### `scripts/train_rl_bc_student.sh`
删除的参数与上述类似，另外保留：
- `train.ppo.priv_info=False` (Student不使用特权信息)
- `train.ppo.proprio_mode=True` (Student使用本体感觉模式)
- `train.ppo.input_mode=proprio` (输入模式)
- `train.ppo.proprio_len=30` (本体感觉历史长度)
- `train.ppo.use_l1=True` (使用L1损失)
- `train.ppo.learning_rate=1e-3` (Student特定的学习率)

#### `scripts/visualize.sh`
**删除的参数**:
- `task.env.object.type=cylinder_pencil-5-7`
- `task.env.randomForceProbScalar=0.25`
- `train.algo=PPO`
- `task.env.rotation_axis=+z`
- `task.env.genGraspCategory=pencil`
- `task.env.privInfo.enable_obj_orientation=True`
- `task.env.privInfo.enable_ft_pos=True`
- `task.env.privInfo.enable_obj_angvel=True`
- `task.env.randomization.randomizeScaleList=[0.3]`
- `task.env.asset.handAsset=assets/linker_hand/L25_dof_urdf.urdf`
- `task.env.privInfo.enable_tactile=True`
- `train.ppo.priv_info=True`
- `task.env.hora.point_cloud_sampled_dim=100`
- `task.env.numObservations=126`
- `task.env.forceScale=2.0`
- `task.env.enable_obj_ends=True`

**保留的参数**（可视化特定）:
- `task.env.episodeLength=4000` (可视化需要更长episode)
- `task.env.reward.angvelClipMax=0.5` (可视化时的reward设置)
- `task.env.reward.angvelPenaltyThres=1.0` (可视化时的penalty阈值)
- `task.env.reset_height_threshold=0.14` (可视化时的阈值)

---

## 三、删除的未使用参数

### 3.1 从`penspin/tasks/linker_hand_hora.py`删除

| 行号 | 参数名 | 初始化代码 | 删除原因 |
|------|--------|-----------|----------|
| 90 | `obs_with_binary_contact` | `self.obs_with_binary_contact = config['env']['obs_with_binary_contact']` | 代码中明确注释"这个参数暂未使用"，且全文仅初始化从未调用 |
| 1325 | `contact_input_dim` | `self.contact_input_dim = p_config['contact_input_dim']` | 仅初始化，全文无其他使用 |
| 1328 | `contact_binarize_threshold` | `self.contact_binarize_threshold = p_config['contact_binarize_threshold']` | 仅初始化，全文无其他使用 |
| 1336 | `enable_priv_hand_scale` | `self.enable_priv_hand_scale = p_config['enable_hand_scale']` | 仅初始化，全文无其他使用 |

**重要修复**: 删除`enable_priv_hand_scale`后，需要修改`_setup_priv_option_config`和`_update_priv_buf`方法，使用`hasattr()`安全检查属性是否存在，避免访问已删除的参数导致`AttributeError`。

**修改的方法**:
- `_setup_priv_option_config` (第1364行): 改为 `if hasattr(self, f'enable_priv_{name}') and eval(f'self.enable_priv_{name}'):`
- `_update_priv_buf` (第1370行): 改为 `if hasattr(self, f'enable_priv_{name}') and eval(f'self.enable_priv_{name}'):`

**注意**: 以下参数保留因为有实际使用：
- `noisy_rpy_scale`: 在第648行用于生成物体姿态的欧拉角噪声
- `noisy_pos_scale`: 在第654行用于生成物体位置的高斯噪声
- `enable_priv_obj_restitution`: 在第1456行的条件判断中使用

### 3.2 从`configs/task/LinkerHandHora.yaml`删除

| 参数名 | 原配置位置 | 删除原因 |
|--------|-----------|----------|
| `obs_with_binary_contact` | `env.obs_with_binary_contact: False` | 对应代码中未使用的参数 |
| `contact_input_dim` | `privInfo.contact_input_dim: 4` | 对应代码中未使用的参数，配置中也有注释"这个参数没有用到" |
| `contact_binarize_threshold` | `privInfo.contact_binarize_threshold: 0.1` | 对应代码中未使用的参数 |
| `enable_hand_scale` | `privInfo.enable_hand_scale: False` | 对应代码中未使用的参数 |
| `enableNetContactF` | `privInfo.enableNetContactF: False` | 配置中注释"这个参数没有用到"，代码中完全未引用 |

**保留的参数**:
- `contact_form`: 在代码中有使用
- `contact_input`: 在代码中有使用
- 所有`enable_priv_*`参数（除了`enable_hand_scale`）均在代码中有实际使用

---

## 四、清理效果

### 4.1 代码简化程度

| 文件 | 清理前行数 | 清理后减少 | 说明 |
|------|-----------|-----------|------|
| `linker_hand_hora.py` | 1668行 | 减少4行参数初始化 | 提高代码可读性 |
| `train_rl_teacher.sh` | 57行 | 减少约18行命令行参数 | 脚本更简洁 |
| `train_rl_bc_teacher.sh` | 65行 | 减少约16行命令行参数 | 脚本更简洁 |
| `train_rl_bc_student.sh` | 65行 | 减少约16行命令行参数 | 脚本更简洁 |
| `visualize.sh` | ~40行 | 减少约15行命令行参数 | 脚本更简洁 |

### 4.2 维护性提升

1. **统一配置管理**: 通用参数统一在configs中维护，修改时只需改一处
2. **代码清晰度**: 删除未使用参数后，代码意图更明确
3. **脚本可读性**: scripts文件只包含经常变化的参数，一目了然
4. **减少错误**: 避免在多个scripts中重复定义相同参数时出现不一致

### 4.3 使用建议

**修改通用参数时**: 
- 直接修改`configs/task/LinkerHandHora.yaml`或`configs/train/LinkerHandHora.yaml`
- 所有scripts会自动使用新的默认值

**运行训练时需要临时覆盖参数**:
- 在scripts的`${EXTRA_ARGS}`位置传入，例如：
  ```bash
  scripts/train_rl_teacher.sh 0 42 test_output task.env.numEnvs=4096
  ```

**新增参数时**:
1. 先在configs中定义默认值
2. 在代码中初始化并使用
3. 仅当参数经常需要改变时，才考虑加入scripts

---

## 五、兼容性说明

### 5.1 向后兼容
本次清理不影响现有功能，所有训练和测试流程保持不变。configs中的新默认值与原scripts中的值一致。

### 5.2 迁移指南
如果您之前通过修改scripts中的参数来自定义训练：
- 现在应该修改configs文件中的对应参数
- 或者通过命令行`${EXTRA_ARGS}`传入参数覆盖

### 5.3 已验证的配置组合
清理后的配置已确保以下场景正常工作：
- ✅ Teacher模型训练 (PPO)
- ✅ Teacher模型训练 (RL+BC)
- ✅ Student模型训练 (RL+BC)
- ✅ 模型可视化和评估

---

## 六、后续优化建议

1. **进一步参数审查**: 定期检查configs中的参数是否都有实际使用
2. **配置模板化**: 考虑为不同实验场景创建配置模板
3. **参数验证**: 添加参数有效性检查，避免配置错误
4. **文档完善**: 为每个重要参数添加详细说明文档

---

**修改记录**:
- 2025-11-17 初始版本: 完成参数清理和文档编写
- 2025-11-17 修复1: 修复删除`enable_priv_hand_scale`后的AttributeError问题，在相关方法中添加`hasattr()`安全检查
- 2025-11-17 修复2: 修复`point_cloud_sampled_dim`默认值错误(0→100)，避免点云维度不匹配导致的IndexError
