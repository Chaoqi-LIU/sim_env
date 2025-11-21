"""
Convert downloaded Libero dataset to desired zarr format.
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
from policy_research.env.libero.dataset_conversion import convert_libero_hdft_to_zarr


@click.command()
@click.option('--root_dir', type=str, default="data/libero")
@click.option('--hdf5_dir_name', type=str, default="hdf5_datasets")
@click.option('-n', '--num_sample_demo', type=int, default=None)
def convert_all_libero_datasets(
    root_dir: str,
    hdf5_dir_name: str,
    num_sample_demo: int,
):
    hdf5_root = os.path.join(root_dir, hdf5_dir_name)
    hdf5_paths = [os.path.join(hdf5_root, f)
        for f in os.listdir(hdf5_root)
        if f.endswith('.hdf5')
    ]
    
    for task_hdf5_path in hdf5_paths:
        print(f"Converting {task_hdf5_path}...")

        try:
            replay_buffer = convert_libero_hdft_to_zarr(
                hdf5_path=task_hdf5_path,
                sample_ndemo=num_sample_demo,
            )

            # check if save path exists
            n_demo = replay_buffer.n_episodes
            task_name = pathlib.Path(task_hdf5_path).stem[:-len('_demo')]
            save_path = f"{root_dir}/{task_name}_N{n_demo}.zarr"
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
            print(f"Failed to convert {task_hdf5_path}: {e}")
            continue

    print("All done.")


if __name__ == "__main__":
    convert_all_libero_datasets()
    