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

from policy_research.common.input_util import wait_user_input
from policy_research.env.drake.util import sample_feasible_waypoints

@click.command()
@click.option('-t', '--task_name', type=str, required=True)
@click.option('-n', '--num_waypoints', type=int, required=True)
def gen_env_feasible_waypoints(
    task_name: str,
    num_waypoints: int,
):
    save_path=os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'workspace_feasible_points',
        f'drake-{task_name}_N{num_waypoints}.npy'
    )
    if os.path.exists(save_path):
        keypress = wait_user_input(
            valid_input=lambda key: key in ['', 'y', 'n'],
            prompt=f"{save_path} already exists. Overwrite? [y/`n`]: ",
            default='n',
        )
        if keypress == 'n':
            print("Exiting without overwriting.")
            return
        else:
            os.system(f'rm {save_path}')

    waypoints = sample_feasible_waypoints(
        env=task_name,
        num_waypoints=num_waypoints,
        verbose=False,
        tqdm_bar=True,
    )
    np.save(save_path, waypoints)


if __name__ == "__main__":
    gen_env_feasible_waypoints()
