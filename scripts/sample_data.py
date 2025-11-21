"""
Usage:
python sample_data.py -p path -s savepath -c samplenum ...
"""

if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import click
import numpy as np
import zarr
import re

from policy_research.common.input_util import wait_user_input
from policy_research.common.replay_buffer import ReplayBuffer


@click.command()
@click.option('-p', '--path', required=True, type=str)
@click.option('-s', '--save_path', type=str, default=None)
@click.option('-c', '--num', required=True, type=int)
def sample_data(path: str, save_path: str, num: int):
    assert os.path.exists(path), f"Path {path} does not exist."

    # load all the data
    buffer = ReplayBuffer.create_from_path(path)

    # if `path` is in the format of '..._{buffer.n_episodes}.zarr', 
    # then we set `save_path` to '..._{num}.zarr'
    if (
        save_path is None and
        int(re.search(r'_[0-9]+\.zarr', path).group(0)[1:-5]) == buffer.n_episodes
    ):
        save_path = re.sub(r'_[0-9]+\.zarr', f'_{num}.zarr', path)
    
    # check if the save path exists
    if os.path.exists(save_path):
        keypress = wait_user_input(
            valid_input=lambda key: key in ['', 'y', 'n'],
            prompt=f"{save_path} already exists. Overwrite? [y/`n`]: ",
            default='n'
        )
        if keypress == 'n':
            print("Abort")
            return
        else:
            os.system(f"rm -rf {save_path}")

    sampled_buffer = ReplayBuffer.create_empty_zarr()

    # take a glance at the original data
    print(f"{'-' * 50}\nsampling data ... \n{buffer}")

    # sample the data
    num = min(num, buffer.n_episodes)
    sample_indics = np.random.choice(buffer.n_episodes, num, replace=False)
    for i in sample_indics:
        sampled_buffer.add_episode(buffer.get_episode(i, copy=True))

    # report and save
    print(f"{'-' * 50}\nsaving sampled data ...")
    print(f"{save_path}: \n{sampled_buffer}")
    compressor = zarr.Blosc(cname='zstd', clevel=5, shuffle=1)
    sampled_buffer.save_to_path(save_path, compressors=compressor)


if __name__ == '__main__':
    sample_data()
