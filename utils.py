import os, glob
import torch
import numpy as np
import random
import re
import json

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def load_checkpoint(checkpoint_file=None, fold=None, config=None):
    checkpoint = None
    if checkpoint_file:
        checkpoint = torch.load(checkpoint_file)
    elif config and fold is not None:
        checkpoint_files = glob.glob(f"{config.checkpoint_dir}/fold=*-epoch=*.pth")
        checkpoint_files = {f: int(re.search('epoch=(\d+)', f).group(1)) for f in checkpoint_files 
                            if int(re.search('fold=(\d+)', f).group(1)) == fold}
        if len(checkpoint_files) > 0:
            checkpoint_file = max(checkpoint_files, key=checkpoint_files.get)
            checkpoint = torch.load(checkpoint_file)
    return checkpoint

def save_checkpoint(checkpoint, save_path, config):
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    torch.save(checkpoint, save_path)

def log_to_file(log_stats, log_file, config):
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    with open(f"{config.checkpoint_dir}/{log_file}", mode="a", encoding="utf-8") as f:
        f.write(json.dumps(log_stats) + "\n")