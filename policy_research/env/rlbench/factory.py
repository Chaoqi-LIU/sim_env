import numpy as np
from policy_research.env.rlbench.env import RlbenchEnv
from typing import List, Optional


MT_TASKS = {
    'mt4': [
        'close_box',
        'toilet_seat_down',
        'close_microwave',
        'open_window',
    ],
}


def is_multitask(task_name: str) -> bool:
    return task_name in MT_TASKS


def get_subtasks(task_name: str) -> List[str]:
    if is_multitask(task_name):
        return MT_TASKS[task_name]
    else:
        return [task_name]


def take_a_glance(task_names: List[str], 
    camera_name: Optional[str]) -> List[np.ndarray]:
    env = RlbenchEnv(image_size=512)
    obs = []
    for task_name in task_names:
        env.set_task(task_name)
        if camera_name is None:
            obs.append(env.render())
        else:
            obs.append(env.reset()[0][camera_name + '_rgb'])
    env.close()
    return obs
