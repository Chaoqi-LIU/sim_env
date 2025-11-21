"""
Usage:
python merge_data.py -p path1 -p path2 -p path3 ...
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

from policy_research.common.input_util import wait_user_input
from policy_research.common.replay_buffer import ReplayBuffer

from typing import Tuple


@click.command()
@click.option('-p', '--paths', multiple=True, required=True, type=str)
@click.option('-s', '--save_path', required=True, type=str)
@click.option('-r', '--shuffle', is_flag=True)
def merge_data(paths: Tuple[str], save_path: str, shuffle: bool):
    if len(paths) <= 1:
        print("Only one path provided, directly copying the data.")
    for path in paths:
        assert os.path.exists(path), f"Path {path} does not exist."
    
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

    merged_buffer = ReplayBuffer.create_empty_zarr()

    # load all the data
    buffers = [ReplayBuffer.create_from_path(path) for path in paths]
    buffer_lens = [buffer.n_episodes for buffer in buffers]
    buffer_idx = [0] * len(buffers)

    # take a glance
    for path, buff in zip(paths, buffers):
        print("-" * 50)
        print(f"{path}: \n{buff}")

    # merge the data    
    total_len = sum(buffer_lens)
    for _ in range(total_len):
        if shuffle:
            idx = np.random.randint(0, len(buffers))
        else:
            idx = 0     # always take the first buffer
        
        # load and place
        eps = buffers[idx].get_episode(buffer_idx[idx], copy=True)
        merged_buffer.add_episode(eps)

        # update pool
        buffer_idx[idx] += 1
        if buffer_idx[idx] >= buffer_lens[idx]:
            buffers.pop(idx)
            buffer_lens.pop(idx)
            buffer_idx.pop(idx)
    
    assert len(buffers) == 0, "Some buffers are not fully merged."

    # report and save
    print(f"{'-' * 50}\nsaving merged data ...")
    print(f"{save_path}: \n{merged_buffer}")
    compressor = zarr.Blosc(cname='zstd', clevel=5, shuffle=1)
    merged_buffer.save_to_path(save_path, compressors=compressor)


if __name__ == '__main__':
    merge_data()
