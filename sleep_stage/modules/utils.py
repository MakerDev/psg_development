#%%
"[TensorBoard]"

import torch
from torch.utils.tensorboard import SummaryWriter
from os import path, mkdir
from datetime import datetime

class TensorBoard(SummaryWriter):

    def __init__(self, root):

        if not path.isdir(root): mkdir(root)
        super(TensorBoard, self).__init__(root)

    # def __tensorboard_dir_checker(self, root_dir, code):

    #     ftime = "%Y-%m-%dT%H-%M-%S"
    #     tensorboard_dir = path.join(root_dir,
    #         f"[{datetime.now().strftime(ftime)}][{code}]")
        
    #     if not path.isdir(tensorboard_dir):
    #         mkdir(tensorboard_dir)
    #     else:
    #         raise ValueError(f"Invalid directory - {tensorboard_dir}")
        
    #     return tensorboard_dir

    def write_epoch(self, idx, history):
        
        for key, val in history.items():
            if len(val) != 0: 
                self.add_scalar(key, val[-1], idx)


class ContextManager:

    def __init__(self, root):

        if not path.isdir(root): mkdir(root)
        self.root = root

    def load(self, name):

        torch_object = torch.load(path.join(self.root, name))

        return torch_object

    def save(self, torch_object, name):

        torch.save(torch_object, path.join(self.root, name))

        pass

