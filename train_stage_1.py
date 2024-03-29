import os, glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import timm
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from dataset import *
from model import *
from utils import *


class CFG:
    
    ## Dataset
    input_dir = '.'
    img_col = 'StudyInstanceUID'
    label_cols = [
        'ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal', 
        'NGT - Abnormal', 'NGT - Borderline', 'NGT - Incompletely Imaged', 'NGT - Normal', 
        'CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal', 
        'Swan Ganz Catheter Present',
    ]
    color_map = {
        'ETT': (255, 255, 0),
        'NGT': (255, 0, 255),
        'CVC': (0, 255, 255),
        'Swan Ganz Catheter Present': (0, 128, 128),
        'Normal': (255, 0, 0),
        'Borderline': (0, 255, 0),
        'Abnormal': (0, 0, 255),
    }
    draw_type = 'line' 
    draw_endpoint = True 
    drop_bg = False
    batch_size = 32
    image_size = 512
    num_workers = 2
    pin_memory = True
    kfold = 5
    seed = 42

    ## Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = 'convnext_tiny'
    in_chans = 3
    num_classes = len(label_cols)
    drop_path_rate = 0.1
    pretrained = False                       # True: load pretrained model, False: train from scratch
    checkpoint_path = 'convnext_tiny_22k_1k_384_altered.pth'                    # Path to model's pretrained weights

    ## Training
    n_epochs = 10
    optimizer = 'AdamW'
    learning_rate = 1e-4
    weight_decay = 1e-5
    lr_scheduler = 'CosineAnnealingWarmRestarts' # 'CosineAnnealingLR' #
    lr_scheduler_params = {
        # 'T_max': 5,
        'T_0': 5, 'T_mult': 1,
        'eta_min': 1e-6,
    }
    resume = True                           # Resume training if True
    checkpoint_dir = 'teacher_checkpoint'   # Directory to save new checkpoints
    save_freq = 2                           # Number of checkpoints to save after each epoch
    debug = False                           # Get a few samples for debugging


class CFG2(CFG):
    drop_bg = True
    checkpoint_dir = 'teacher_checkpoint-2'   # Directory to save new checkpoints


class CFGLine(CFG):
    draw_endpoint = False 
    drop_bg = False
    n_epochs = 20
    checkpoint_dir = 'teacher_checkpoint-line'   # Directory to save new checkpoints


class CFGSquare(CFG):
    color_map = {
        'ETT - Abnormal': (255, 0, 0),
        'ETT - Borderline': (0, 255, 0),
        'ETT - Normal': (0, 0, 255),
        'NGT - Abnormal': (255, 255, 0),
        'NGT - Borderline': (255, 0, 255),
        'NGT - Incompletely Imaged': (0, 255, 255),
        'NGT - Normal': (128, 0, 0),
        'CVC - Abnormal': (0, 128, 0),
        'CVC - Borderline': (0, 0, 128),
        'CVC - Normal': (128, 128, 0),
        'Swan Ganz Catheter Present': (128, 0, 128),
    }
    draw_type = 'square' 
    draw_endpoint = False 
    drop_bg = False
    checkpoint_dir = 'teacher_checkpoint-square'   # Directory to save new checkpoints


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1. - pt)**self.gamma * bce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def train_one_epoch(model, loader, criterion, optimizer, scaler, config):
    model.train()
    running_loss = AverageMeter()
    tepoch = tqdm(loader)
    # with tqdm(total=len(loader)) as tepoch:
    for batch_idx, (data, targets) in enumerate(tepoch):
        data = data.to(config.device) 
        targets = targets.to(config.device)

        # Forward pass
        with torch.cuda.amp.autocast():
            _, outputs = model(data)
            loss = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
        # Update progress bar
        running_loss.update(loss.item(), data.size(0))
        # tepoch.set_postfix(loss=running_loss.avg)
        tepoch.set_description_str(f'Loss: {running_loss.avg:.4f}')

    return running_loss.avg


def valid_one_epoch(model, loader, criterion, config):
    model.eval()
    running_loss = AverageMeter()
    preds = []
    gts = []
    # with tqdm(total=len(loader)) as tepoch:
    tepoch = tqdm(loader)
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(tepoch):
            data, targets = data.to(config.device), targets.to(config.device)

            # Forward pass
            _, outputs = model(data)
            loss = criterion(outputs, targets)
            preds.append(outputs.cpu())
            gts.append(targets.cpu())

            # Update progress bar
            running_loss.update(loss.item(), data.size(0))
            # tepoch.set_postfix(loss=running_loss.avg)
            tepoch.set_description_str(f'Loss: {running_loss.avg:.4f}')

    # Calculate AUC score         
    auc_scores = {}
    preds = torch.cat(preds).sigmoid()
    gts = torch.cat(gts)
    for i, col in enumerate(config.label_cols):
        auc_scores[col] = roc_auc_score(gts[:, i], preds[:, i])
    auc = np.mean(list(auc_scores.values()))
    return running_loss.avg, auc


def run(fold, config):
    # Prepare train and val set
    full_train_df = pd.read_csv(f'{config.input_dir}/train_fold{config.kfold}.csv')
    df_annot = pd.read_csv(f'{config.input_dir}/train_annotations.csv')
    full_train_df = full_train_df.set_index(config.img_col).loc[df_annot[config.img_col].unique()].reset_index()
    train_df = full_train_df.query(f"fold!={fold}")
    val_df = full_train_df.query(f"fold=={fold}")

    if config.debug:
        train_df = train_df.sample(80)
        val_df = val_df.sample(640)

    train_transform = [
        build_transform(None, adjust_color=True, is_train=False, include_top=False),
        build_transform(config.image_size, adjust_color=False, is_train=True, include_top=True)
    ]
    val_transform = build_transform(config.image_size, adjust_color=False, is_train=False, include_top=True)
    
    train_dataset = RANZCRDataset(image_dir=f"{config.input_dir}/train", df=train_df, 
                                  img_col=config.img_col, label_cols=config.label_cols,
                                  df_annot=df_annot, color_map=config.color_map, 
                                  draw_type=config.draw_type, draw_endpoint=config.draw_endpoint, 
                                  drop_bg=config.drop_bg, transform=train_transform[1], 
                                  prev_transform=train_transform[0], return_img='annot_image')
    val_dataset = RANZCRDataset(image_dir=f"{config.input_dir}/train", df=val_df,
                                img_col=config.img_col, label_cols=config.label_cols,
                                df_annot=df_annot, color_map=config.color_map, 
                                draw_type=config.draw_type, draw_endpoint=config.draw_endpoint, 
                                drop_bg=config.drop_bg, transform=val_transform, return_img='annot_image')
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                              num_workers=config.num_workers, pin_memory=config.pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,
                            num_workers=config.num_workers, pin_memory=config.pin_memory)

    # Initialize model
    model = RANZCRClassifier(config.model_name, pretrained=config.pretrained,
                             checkpoint_path=config.checkpoint_path, 
                             in_chans=config.in_chans, num_classes=config.num_classes,
                             drop_path_rate=config.drop_path_rate)
    model = model.to(config.device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of params: {n_parameters}")
    # print(f"LR: {config.learning_rate}")

    # Set up training
    start_epoch = 0
    criterion = FocalLoss() # nn.BCEWithLogitsLoss()
    optimizer = eval(f"optim.{config.optimizer}(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)")
    scheduler = eval(f"optim.lr_scheduler.{config.lr_scheduler}(optimizer, **config.lr_scheduler_params)")
    scaler = torch.cuda.amp.GradScaler()
    writer = SummaryWriter()

    # Load checkpoint
    if config.resume:
        checkpoint = load_checkpoint(fold=fold, checkpoint_dir=config.checkpoint_dir)
        if checkpoint:
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            scaler.load_state_dict(checkpoint['scaler'])
            start_epoch = checkpoint['epoch'] + 1
    print(f"LR: {scheduler.get_last_lr()[0]}")

    best_auc = 0
    for epoch in range(start_epoch, config.n_epochs):
        print(f'Fold: {fold}  Epoch: {epoch}')
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, config)
        val_loss, auc = valid_one_epoch(model, val_loader, criterion, config)
        scheduler.step()
        print(f'AUC: {auc:.4f}')
        print(f'New LR: {scheduler.get_last_lr()[0]}')

        # Log to file
        log_stats = {
            'fold': fold, 'epoch': epoch, 'n_parameters': n_parameters,
            'train_loss': train_loss, 'val_loss': val_loss, 'val_auc': auc,
        }
        log_to_file(log_stats, "log.txt", config.checkpoint_dir)

        # Tensorboard logging
        if writer:
            writer.add_scalar('train/loss', train_loss, global_step=epoch)
            writer.add_scalar('val/loss', val_loss, global_step=epoch)
            writer.add_scalar('val/auc', auc, global_step=epoch)

        # Save checkpoint
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'epoch': epoch,
        }            
        save_path = f"{config.checkpoint_dir}/fold={fold}-epoch={epoch}-auc={auc:.4f}.pth"
        save_checkpoint(checkpoint, save_path)
        if auc > best_auc:
            best_auc = auc
            checkpoint['auc'] = auc
            save_path = f"{config.checkpoint_dir}/fold={fold}-best.pth"
            print('--> Saving checkpoint')
            save_checkpoint(checkpoint, save_path)

        # Delete old checkpoint
        for fname in glob.glob(f"{config.checkpoint_dir}/fold={fold}-epoch={epoch-config.save_freq}*"):
            os.remove(fname)


def main():
    config = CFG
    set_seed(config.seed)
    if os.path.exists('kaggle/input'):
        config.input_dir = '../input/ranzcr-clip-catheter-line-classification'

    # Train model
    for fold in range(config.kfold):
        run(fold, config)

if __name__ == "__main__":
    main()