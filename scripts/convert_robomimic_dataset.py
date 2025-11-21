"""
Convert downloaded RoboMimic dataset to desired zarr format.
"""

if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import click
import zarr
import pathlib

from policy_research.common.input_util import wait_user_input
from policy_research.env.robomimic.dataset_conversion import convert_robomimic_hdf5_to_zarr

@click.command()
@click.option('--root_dir', type=str, default="data/robomimic")
@click.option('--hdf5_dir_name', type=str, default="hdf5_datasets")
@click.option('-ds', '--ds_name', type=click.Choice(['ph', 'mh']), default='ph')
@click.option('--hdf5_type', type=click.Choice(['demo', 'low_dim', 'image']), default='demo')
@click.option('-n', '--num_sample_demo', type=int, default=None)
def convert_all_robomimic_datasets(
    root_dir: str,
    hdf5_dir_name: str,
    ds_name: str,
    hdf5_type: str,
    num_sample_demo,
):
    hdf5_root = os.path.join(root_dir, hdf5_dir_name)
    task_dirs = [name for name in os.listdir(hdf5_root)
        if os.path.isdir(os.path.join(hdf5_root, name))]
    
    for task_dir in task_dirs:
        print(f"Converting {task_dir} ({ds_name})...")

        try:
            replay_buffer = convert_robomimic_hdf5_to_zarr(
                hdf5_path=os.path.join(hdf5_root, task_dir, ds_name, f'{hdf5_type}.hdf5'),
                sample_ndemo=num_sample_demo,
                desired_image_size=128,
                desired_camera_names=[
                    'agentview',
                    'robot0_eye_in_hand',
                ],
                desired_state_ports=[
                    'robot0_joint_pos',
                    'robot0_eef_pos',
                    'robot0_eef_quat',
                    'robot0_gripper_qpos',
                ],
            )

            # check if save path exists
            n_demo = replay_buffer.n_episodes
            save_path = f"{root_dir}/{task_dir}_{ds_name}_N{n_demo}.zarr"
            if os.path.exists(save_path):
                keypress = wait_user_input(
                    valid_input=lambda key: key in ['', 'y', 'n'],
                    prompt=f"{save_path} already exists. Overwrite? [y/`n`]: ",
                    default='n'
                )
                if keypress == 'n':
                    print("Abort")
                    continue
                else:
                    os.system(f"rm -rf {save_path}")
            
            # save
            pathlib.Path(save_path).mkdir(parents=True)
            compressor = zarr.Blosc(cname='zstd', clevel=5, shuffle=1)
            replay_buffer.save_to_path(save_path, compressor=compressor)
            print(f"Saved to {save_path}")

        except Exception as e:
            print(f"Failed to convert {task_dir} ({ds_name}): {e}")
            continue

    print("All done.")



if __name__ == "__main__":
    convert_all_robomimic_datasets()
