import os, glob
import cv2
import torch
import pandas as pd
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


def draw_annotation(image, df_annot, img_col, img_id, color_map):
    """
    Draws the annotations on the image

    Args:
    -----
    image: numpy array
        Image to draw the annotations on
    df_annot: pandas dataframe
        Dataframe containing the annotations
    img_col: str
        Name of the column containing the image id
    img_id: str
        Image id
    color_map: dict
        Dictionary containing the color for each class
    """
    df = df_annot.query(f"{img_col} == '{img_id}'")
    for index, annot_row in df.iterrows():
        data = eval(annot_row['data'])
        label = annot_row["label"].split(' - ')
        cv2.polylines(image, np.int32([data]), isClosed=False, 
                      color=color_map[label[0]], thickness=15, lineType=16)
        if len(label) > 1 and label[1] != 'Incompletely Imaged':
            # x_center, y_center = image.shape[1]/2, image.shape[0]/2
            # x, y = min([data[0], data[-1]], key=lambda x: (x[0]-x_center)**2 + (x[1]-y_center)**2)
            # cv2.circle(image, (x, y), radius=15, 
            #             color=color_map[label[1]], thickness=25)
            cv2.circle(image, tuple(data[0]), radius=15, 
                        color=color_map[label[1]], thickness=25)
            cv2.circle(image, tuple(data[-1]), radius=15, 
                        color=color_map[label[1]], thickness=25)
    return image


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


def load_checkpoint(checkpoint_path=None, fold=None, checkpoint_dir=None, postfix=''):
    """
    Loads the checkpoint from the checkpoint_path or the latest checkpoint from the checkpoint_dir
    
    Args:
    -----
    checkpoint_path: str
        Path to the checkpoint
    fold: int
        Fold number
    checkpoint_dir: str
        Path to the checkpoint directory
    postfix: str
        Postfix to add to the checkpoint name
    """
    checkpoint = None
    if checkpoint_path:
        # Load checkpoint given by the path
        if checkpoint_path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(checkpoint_path, 
                                                            map_location='cpu', 
                                                            check_hash=True)
        else:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"Loaded checkpoint from {checkpoint_path}")
    elif checkpoint_dir and fold is not None:
        # Load checkpoint from the latest one
        checkpoint_files = glob.glob(f"{checkpoint_dir}/fold=*-epoch=*{postfix}.pth")
        checkpoint_files = {f: int(re.search('epoch=(\d+)', f).group(1)) for f in checkpoint_files 
                            if int(re.search('fold=(\d+)', f).group(1)) == fold}
        if len(checkpoint_files) > 0:
            checkpoint_file = max(checkpoint_files, key=checkpoint_files.get)
            checkpoint = torch.load(checkpoint_file, map_location='cpu')
    return checkpoint

def save_checkpoint(checkpoint, save_path):
    """Saves the checkpoint to the save_path"""
    os.makedirs(os.path.split(save_path)[0], exist_ok=True)
    torch.save(checkpoint, save_path)

def log_to_file(log_stats, log_file, checkpoint_dir):
    """Saves the log to the log_file"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(f"{checkpoint_dir}/{log_file}", mode="a", encoding="utf-8") as f:
        f.write(json.dumps(log_stats) + "\n")