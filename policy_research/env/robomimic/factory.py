from typing import List

MT_TASKS = {
    'mt3': [
        'Lift',
        'PickPlaceCan',
        # 'ToolHang',   # tool_hang has no mh demo
        'NutAssemblySquare',
    ],
    'mt4': [
        'Lift',
        'PickPlaceCan',
        'ToolHang',
        'NutAssemblySquare',
    ]
}

def is_multitask(task_name: str) -> bool:
    return task_name in MT_TASKS


def get_subtasks(task_name: str) -> List[str]:
    if is_multitask(task_name):
        return MT_TASKS[task_name]
    else:
        return [task_name]
