conda create -n penspin python=3.8
conda activate penspin
conda install -c conda-forge urllib3
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install hydra-core termcolor gym tensorboardX trimesh numpy==1.22.4 wandb

Download IsaacGym Preview 4.0 ([Download](https://developer.nvidia.com/isaac-gym)),

cd ../isaacgym/python/   
pip install -e .

cd ../../LinkerPenspin/|
////可以修改.envrc: export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/user/anaconda3/envs/penspin/lib/