# train.py
# Script to train policies in Isaac Gym
#
# Copyright (c) 2018-2021, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import isaacgym

import os
import hydra
import datetime
from termcolor import cprint
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

from penspin.algo.ppo.demon import DemonTrain
from penspin.algo.ppo.ppo_rl_teacher import PPOTeacher
from penspin.algo.ppo.ppo_rl_bc_teacher import PPO_RL_BC_Teacher
from penspin.algo.ppo.ppo_rl_bc_student import PPO_RL_BC_Student

from penspin.tasks import isaacgym_task_map
from penspin.utils.reformat import omegaconf_to_dict, print_dict
from penspin.utils.misc import set_np_formatting, set_seed, git_hash, git_diff_config

# OmegaConf & Hydra Config
# Resolvers used in hydra configs (see https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#resolvers)
OmegaConf.register_new_resolver('eq', lambda x, y: x.lower() == y.lower())
OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver('if', lambda pred, a, b: a if pred else b)
# allows us to resolve default arguments which are copied in multiple places in the config.
# used primarily for num_env
OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg == '' else arg)


@hydra.main(config_name='config', config_path='configs')
def main(config: DictConfig):
    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if config.checkpoint:
        # 如果包含通配符，则从 glob 模块导入 glob 函数。
        # glob 函数用于查找匹配特定模式的文件路径名。
        if '*' in config.checkpoint:
            from glob import glob
            _ckpt = glob(config.checkpoint)
            assert len(_ckpt) == 1
            config.checkpoint = _ckpt[0]
        config.checkpoint = to_absolute_path(config.checkpoint)

    # set numpy formatting for printing only
    set_np_formatting()

    cfg_dict = omegaconf_to_dict(config)
    print_dict(cfg_dict)

    config.seed = set_seed(config.seed)

    cprint('Start Building the Environment', 'green', attrs=['bold'])
    env = isaacgym_task_map[config.task_name](
        config=omegaconf_to_dict(config.task),
        sim_device=config.sim_device,
        graphics_device_id=config.graphics_device_id,
        headless=config.headless,
    )
    # env = LinkerHandHora(config=omegaconf_to_dict(config.task),...)

    output_dif = os.path.join('outputs', config.train.ppo.output_name)
    os.makedirs(output_dif, exist_ok=True)
    agent = eval(config.train.algo)(env, output_dif, full_config=config)
    # agent = PPO(env, output_dif, full_config=config)
    
    if config.test:
        assert config.train.load_path
        agent.restore_test(config.train.load_path)
        agent.test()
    else:
        date = str(datetime.datetime.now().strftime('%m%d%H'))
        # print(git_diff_config('./'))
        # gitdiff_suffix = ''
        # os.system(f'git diff HEAD > {output_dif}/gitdiff{gitdiff_suffix}.patch')
        # with open(os.path.join(output_dif, f'config_{date}_{git_hash()}.yaml'), 'w') as f:
        #     f.write(OmegaConf.to_yaml(config))
        agent.restore_train(config.train.load_path) #这里就是config.checkpoint
        best_reward = agent.train()
        
        # 输出评分供Optuna使用（通过标准输出）
        if best_reward is not None:
            print(f"\nOPTUNA_SCORE: {best_reward}")
        
        return best_reward  # 返回最佳奖励值供Optuna使用


if __name__ == '__main__':
    main()
