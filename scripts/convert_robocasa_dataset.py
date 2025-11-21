"""
Convert downloaded RoboCasa dataset to desired zarr format.
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
from robocasa.utils.dataset_registry import SINGLE_STAGE_TASK_DATASETS

from policy_research.common.input_util import wait_user_input
from policy_research.env.robocasa.dataset_conversion import convert_robocasa_hdf5_to_zarr

@click.command()
@click.option('--root_dir', type=str, default="data/robocasa")
@click.option('-ds', '--ds_type', type=click.Choice(['human_im', 
    'human_raw', 'mg_im']), default='human_im')
@click.option('-n', '--num_sample_demo', type=int, default=None)
def convert_all_robocasa_datasets(
    root_dir: str,
    ds_type: str = "human_im",
    num_sample_demo: int = None
):
    for task_name in list(SINGLE_STAGE_TASK_DATASETS.keys()):
        try:
            print(f"Converting {task_name} ({ds_type})...")
            replay_buffer = convert_robocasa_hdf5_to_zarr(
                task_name=task_name,
                ds_type=ds_type,
                sample_ndemo=num_sample_demo,
            )

            # check if save path exists
            n_demo = replay_buffer.n_episodes
            save_path = f"{root_dir}/{task_name}_{ds_type}_N{n_demo}.zarr"
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
            print(f"Failed to convert {task_name} ({ds_type}): {e}")
            continue
    print("All done.")


if __name__ == "__main__":
    convert_all_robocasa_datasets()
