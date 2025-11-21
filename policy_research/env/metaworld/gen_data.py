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
import hydra

from policy_research.common.replay_buffer import ReplayBuffer
from policy_research.common.input_util import wait_user_input
from policy_research.env.metaworld.factory import get_metaworld_env, get_subtasks
from typing import List, Tuple


def load_expert_policy(task_name) -> List:
    subtask_names = get_subtasks(task_name)
    experts = []
    for name in subtask_names:
        if name == 'peg-insert-side':
            expert_name = 'PegInsertionSide'
        else:
            expert_name = ''.join([s.capitalize() for s in name.split('-')])
        expert = hydra.utils.get_class(f"metaworld.policies.Sawyer{expert_name}V2Policy")
        experts.append(expert())
    return experts


@click.command()
@click.option('-t', '--task_name', type=str, required=True)
@click.option('-d', '--root_data_dir', type=str, default='data/metaworld')
@click.option('-s', '--save_dir', type=str, default=None)
@click.option('-c', '--num_episodes', type=int, default=10)
@click.option('-o', '--sensors', multiple=True, type=str, default=(
    'topview', 'corner', 'corner2', 'corner3', 'behindGripper', 'gripperPOV'))
@click.option('--save_subtasks', is_flag=True)
def gen_metaworld_data(
    task_name: str,
    root_data_dir: str,
    save_dir: str,
    num_episodes: int,
    sensors: Tuple[str],
    save_subtasks: bool
):
    if save_dir is None:
        save_dir = os.path.join(root_data_dir, f'{task_name}_N{num_episodes}.zarr')

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
    save_dir = [save_dir,]

    envs = get_metaworld_env(
        task_name=task_name,
        image_size=128,
        camera_names=sensors,
        device='cuda:0',
        oracle=True
    )
    experts = load_expert_policy(task_name)
    assert len(envs) == len(experts)
    assert not save_subtasks or len(envs) > 1

    # create subtask directories
    if save_subtasks:
        for senv in envs:
            sname = senv.task_name
            sdir = os.path.join(root_data_dir, f'{sname}_N{num_episodes // len(envs)}.zarr')
            if os.path.exists(sdir):
                keypress = wait_user_input(
                    valid_input=lambda key: key in ['', 'y', 'n'],
                    prompt=f"{sdir} already exists. Overwrite? [y/`n`]: ",
                    default='n'
                )
                if keypress == 'n':
                    print("Abort")
                    return
                else:
                    os.system(f"rm -rf {sdir}")
            os.mkdir(sdir)
            save_dir.append(sdir)
    save_dir = save_dir[1:] + save_dir[:1]

    # distribute episodes to different envs
    num_envs = len(envs)
    env_indices = [i % num_envs for i in range(num_episodes)]
    assert len(env_indices) == num_episodes
    assert num_episodes % num_envs == 0

    replay_buffers = [
        ReplayBuffer.create_empty_zarr() for _ in range(
            (num_envs + 1) if save_subtasks else 1
        )
    ]
    data_dtype = {
        **{f"{sensor}_rgb": 'uint8' for sensor in sensors},
        **{f"{sensor}_depth": 'float32' for sensor in sensors},
    }

    episode_idx = 0
    while episode_idx < num_episodes:
        # pick env and expert
        env_idx = env_indices[episode_idx]
        env = envs[env_idx]
        expert = experts[env_idx]

        obs_dict = env.reset()
        done = False
        episode_reward = 0.0
        episode_success = False
        episode_success_count = 0

        this_data_collected = {'action': []}
        while not done:
            action = expert.get_action(obs_dict['full_state'])

            # collect data
            for k, v in obs_dict.items():
                if k not in this_data_collected:
                    this_data_collected[k] = []
                this_data_collected[k].append(v)
            this_data_collected['action'].append(action)

            obs_dict, reward, done, info = env.step(action)
            episode_reward += reward
            episode_success = episode_success or info['success']
            episode_success_count += info['success']
    
        if not episode_success or episode_success_count < 5:
            print(
                f"Episode {episode_idx + 1}, task: {env.task_name}, failed with "
                f"reward={episode_reward:.3f}, success_count={episode_success_count}, success={episode_success}"
            )
        else:
            episode_idx += 1
            # dtype conversion
            for key in this_data_collected.keys():
                this_data_collected[key] = np.array(this_data_collected[key], dtype=data_dtype.get(key, 'float32'))
            # copy trial data to total data collected
            replay_buffers[-1].add_episode(copy.deepcopy(this_data_collected))
            # copy trial data to subtask data collected if needed
            if save_subtasks:
                replay_buffers[env_idx].add_episode(copy.deepcopy(this_data_collected))
            print(f"Episode {episode_idx}, task: {env.task_name}, reward: {episode_reward}, success count: {episode_success_count}")

    # report
    for buff, sdir in zip(replay_buffers, save_dir):
        print('-' * 50)
        print(f"{sdir}: \n{buff}")

    # save data collected
    compressor = zarr.Blosc(cname='zstd', clevel=5, shuffle=1)
    for i, buff in enumerate(replay_buffers):
        buff.save_to_path(save_dir[i], compressors=compressor)


if __name__ == '__main__':
    gen_metaworld_data()
