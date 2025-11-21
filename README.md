# A collection of simulation benchmarks for robot policy research

Currently we included `robocasa`, `robomimic`, `rlbench`, `metaworld`, `libero`. They all supports asynchronous vectorized policy evaluation. Find more instructions on each environment below. Also check `conda_yaml` for installation commands.

## For MetaWorld

1. Add the following lines to `~/.bashrc`
```
# mujoco
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export MUJOCO_GL=egl
```
2. Make it count
```
source ~/.bashrc
```
3. Download and install mujoco binary to location `${HOME}/.mujoco`
```
$ cd ~/.mujoco

$ wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco210.tar.gz --no-check-certificate

$ tar -xvzf mujoco210.tar.gz
```
4. Generate dataset with
```
python scripts/gen_data.py metaworld --help
```

## For RLBench

1. Add the following lines to `~/.bashrc`
```
# coppelia sim
export COPPELIASIM_ROOT=${HOME}/.coppeliasim
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```
2. Make it count
```
source ~/.bashrc
```
3. Download and install CoppeliaSim binary to location `${HOME}/.coppeliasim`
```
$ wget https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz

$ mkdir -p $COPPELIASIM_ROOT && tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz -C $COPPELIASIM_ROOT --strip-components 1

$ rm -rf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
```
4. Generate dataset with
```
python scripts/gen_data.py rlbench --help
```

## For LIBERO
1. find a desired path and clone the repo
```
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
```
2. install
```
cd LIBERO
pip install -e .
```
3. Download datasets
```
python benchmark_scripts/download_libero_datasets.py --datasets libero_[spatial/object/goal/100]
```
4. Check `scripts/convert_libero_dataset.py`

## For RoboCasa
1. Download assets with 
```
python -m robocasa.scripts.download_kitchen_assets
```
2. Configure macros 
```
python -m robocasa.scripts.setup_macros
```
3. Edit `DATASET_BASE_PATH` in `robocasa/macros_private.py` to desired data folder for data downloading
4. Download datasets 
```
python -m robocasa.scripts.download_datasets --tasks [...] --ds_types [mg_im/human_im/human_raw]
```
5. Check `scripts/convert_robocasa_dataset.py`

## For RoboMimic
1. Download datasets 
```
python -m robomimic.scripts.download_datasets --download_dir data/robomimic/hdf5_datasets --tasks [...] --dataset_types ph --hdf5_types raw
```
2. Check `scripts/convert_robomimic_dataset.py`

## Notes
* RoboCasa requires nightly `robosuite`, install this version
```
git+https://github.com/Chaoqi-LIU/robosuite.git@trajtok#egg=robosuite
```
* LIBERO requires `robosuite==1.4.0`, install with
```
pip install robosuite==1.4.0
```
