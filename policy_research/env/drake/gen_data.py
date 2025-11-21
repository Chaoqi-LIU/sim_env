if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import pathlib
import click
import numpy as np
import copy
import zarr
import tqdm

from policy_research.common.replay_buffer import ReplayBuffer
from policy_research.common.input_util import wait_user_input
from policy_research.env.drake.util import generate_one_episode


@click.command()
@click.option('-t', '--task_name', type=str, required=True)
@click.option('-c', '--num_episodes', type=int, required=True)
@click.option('-l', '--num_waypoints_per_episode', type=int, default=5)
@click.option('-dt', '--time_step', type=float, default=0.05)
@click.option('-d', '--root_data_dir', type=str, default='data/drake')
@click.option('-s', '--save_dir', type=str, default=None)
def gen_drake_data(
    task_name: str,
    num_episodes: int,
    num_waypoints_per_episode: int,
    time_step: float,
    root_data_dir: str,
    save_dir: str,
):
    replay_buffer = ReplayBuffer.create_empty_zarr()
    for _ in tqdm.tqdm(
        range(num_episodes),
        desc=f"Generating drake-{task_name} episodes",
    ):
        positions = generate_one_episode(
            env=task_name,
            num_waypoints=num_waypoints_per_episode,
            time_step=time_step,
            waypoints_bank=os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                f'workspace_feasible_points/drake-{task_name}_N1000.npy'
            )
        ).astype(np.float32)

        if task_name == 'franka':
            # standardize franka action space:
            # 9d position -> 8d position (gripper 2d -> 1d)
            # since drake franka data always has gripper open, 
            # meaning gripper state is always 0.
            positions = positions[:, :8]

        this_data_collected = {
            'action': positions,
            'delta_action': np.diff(positions, axis=0, append=positions[-1:])
        }
        replay_buffer.add_episode(copy.deepcopy(this_data_collected))

    # construct save path
    if save_dir is None:
        num_actions = replay_buffer.episode_ends[-1]
        save_dir = os.path.join(root_data_dir, 
            f'drake_{task_name}_N{num_episodes}_T{num_actions}_t{time_step:.2f}.zarr')
    if os.path.exists(save_dir):
        keypress = wait_user_input(
            valid_input=lambda key: key in ['', 'y', 'n'],
            prompt=f"{save_dir} already exists. Overwrite? [y/`n`]: ",
            default='n'
        )
        if keypress == 'n':
            print("Abort")
            return
        else:
            os.system(f"rm -rf {save_dir}")
    pathlib.Path(save_dir).mkdir(parents=True)

    # report
    print('-' * 50)
    print(f"{save_dir}: \n{replay_buffer}")

    # save data collected
    compressor = zarr.Blosc(cname='zstd', clevel=5, shuffle=1)
    replay_buffer.save_to_path(save_dir, compressor=compressor)


if __name__ == "__main__":
    gen_drake_data()
