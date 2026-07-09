import random
import numpy as np
import torch


def get_module_recursively(model, name):
    splits = name.split(".")
    module = model
    for split in splits:
        new_module = getattr(module, split)
        if new_module is None:
            raise ValueError(f"{module} has no attribute {split}.")
        module = new_module
    return module

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
