"""
...
"""

# external
import os
import torch
import warnings

# gpu infos
def get_num_gpus():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ignore *all* warnings in this block
        try:
            return torch.cuda.device_count()
        except Exception as exception:
            print(f'[Exception]: {exception}')
            return 0


def get_available_gpus():
    gpus = os.environ.get("CUDA_VISIBLE_DEVICES")
    if gpus is None:
        return list(range(get_num_gpus()))
    else:
        return sorted([int(id) for id in gpus.split(",") if id != ""])
