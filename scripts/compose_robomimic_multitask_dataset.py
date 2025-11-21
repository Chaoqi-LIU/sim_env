if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)


import click
import os
import re
import glob
from policy_research.env.robomimic.factory import MT_TASKS

TASK_NAME_TO_DIR_NAME = {
    'Lift': 'lift',
    'PickPlaceCan': 'can',
    'ToolHang': 'tool_hang',
    'NutAssemblySquare': 'square',
}

@click.command()
@click.option('-mt', '--multitask_name', type=click.Choice(list(MT_TASKS.keys())), required=True)
@click.option('--root_dir', type=str, default="data/robomimic")
@click.option('-ds', '--ds_type', type=click.Choice(['ph', 'mh']), default='ph')
def compose_robomimic_multitask_datasets(
    multitask_name: str,
    root_dir: str,
    ds_type: str,
):
    num_demo = 0
    zarr_paths = []
    for task_name in MT_TASKS[multitask_name]:
        pattern = f"{TASK_NAME_TO_DIR_NAME[task_name]}_{ds_type}_N*.zarr"
        matching_files = glob.glob(os.path.join(root_dir, pattern))
        if not matching_files:
            raise FileNotFoundError(f"No files found matching pattern: {pattern}")
        zarr_file = matching_files[0]  # Take the first match
        match = re.search(
            rf"{TASK_NAME_TO_DIR_NAME[task_name]}_{ds_type}_N(\d+)\.zarr", 
            os.path.basename(zarr_file)
        )
        if not match:
            raise ValueError(f"Could not extract demo count from filename: {zarr_file}")
        this_num_demo = int(match.group(1))
        num_demo += this_num_demo
        assert os.path.exists(zarr_file), f"Zarr path {zarr_file} does not exist."
        zarr_paths.append(zarr_file)

    # construct command
    cmd = (
        f"python {ROOT_DIR}/scripts/merge_data.py "
        f"{' '.join(['-p ' + p for p in zarr_paths])} "
        f"-s {os.path.join(root_dir, f'{multitask_name}_N{num_demo}.zarr')} "
        f"--shuffle"
    )
    print(f"Running command: {cmd}")
    os.system(cmd)


if __name__ == "__main__":
    compose_robomimic_multitask_datasets()
