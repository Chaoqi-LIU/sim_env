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

## Notes
* RoboCasa requires nightly `robosuite`, install this version
```
git+https://github.com/Chaoqi-LIU/robosuite.git@trajtok#egg=robosuite
```
* LIBERO requires `robosuite==1.4.0`, install with
```
pip install robosuite==1.4.0
```