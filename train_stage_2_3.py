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


class config:
    
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
    n_stages = 2
    n_epochs = {
        'pre': 15,
        'post': 10,
        'full': 15,
    }
    weights = [0.5, 1]
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
    teacher_dir = 'teacher_checkpoint'      # Path to teacher's checkpoint
    checkpoint_dir = 'student_checkpoint'   # Directory to save new checkpoints
    save_freq = 2                           # Number of checkpoints to save after each epoch
    debug = False                           # Get a few samples for debugging


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


class CustomLoss(nn.Module):
    def __init__(self, weights=[1, 1]):
        super(CustomLoss, self).__init__()
        self.weights = weights
        
    def forward(self, features, teacher_features, y_pred, labels):
        distill_loss = nn.MSELoss()(features.reshape(-1), teacher_features.reshape(-1))
        cls_loss = FocalLoss()(y_pred, labels) # nn.BCEWithLogitsLoss()(y_pred, labels) #
        loss = self.weights[0] * distill_loss + self.weights[1] * cls_loss
        return loss

    
def initialize_loader(fold, config, stage='pre'):
    full_train_df = pd.read_csv(f'{config.input_dir}/train_fold{config.kfold}.csv')
    df_annot = pd.read_csv(f'{config.input_dir}/train_annotations.csv')

    if stage in ['pre', 'full']:
        if stage == 'pre':
            full_train_df = full_train_df.set_index(config.img_col).loc[df_annot[config.img_col].unique()].reset_index()
        train_df = full_train_df.query(f"fold!={fold}")
        val_df = full_train_df.query(f"fold=={fold}")

        if config.debug:
            train_df = train_df.sample(80)
            val_df = val_df.sample(640)

        train_transform = [
            build_transform(None, adjust_color=True, is_train=False, include_top=False),
            build_transform(config.image_size, adjust_color=False, is_train=True, 
                            include_top=True, additional_targets={'annot_image': 'image'})
        ]
        val_transform = build_transform(config.image_size, adjust_color=False, is_train=False, include_top=True)
        
        train_dataset = RANZCRDataset(image_dir=f"{config.input_dir}/train", df=train_df, 
                                    img_col=config.img_col, label_cols=config.label_cols,
                                    df_annot=df_annot, color_map=config.color_map, 
                                    transform=train_transform[1], prev_transform=train_transform[0], 
                                    return_img='both')
        val_dataset = RANZCRDataset(image_dir=f"{config.input_dir}/train", df=val_df,
                                    img_col=config.img_col, label_cols=config.label_cols,
                                    df_annot=df_annot, color_map=config.color_map, 
                                    transform=val_transform, return_img='image')
    elif stage == 'post':
        train_df = full_train_df.query(f"fold!={fold}")
        val_df = full_train_df.query(f"fold=={fold}")

        if config.debug:
            train_df = train_df.sample(80)
            val_df = val_df.sample(640)
            
        train_transform = build_transform(config.image_size, adjust_color=True, is_train=True, include_top=True)
        val_transform = build_transform(config.image_size, adjust_color=False, is_train=False, include_top=True)
        
        train_dataset = RANZCRDataset(image_dir=f"{config.input_dir}/train", df=train_df, 
                                    img_col=config.img_col, label_cols=config.label_cols,
                                    transform=train_transform, return_img='image')
        val_dataset = RANZCRDataset(image_dir=f"{config.input_dir}/train", df=val_df,
                                    img_col=config.img_col, label_cols=config.label_cols,
                                    transform=val_transform, return_img='image')
    else:
        raise ValueError(f'Stage must be either pre or post or full, not {stage}')

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                            num_workers=config.num_workers, pin_memory=config.pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,
                            num_workers=config.num_workers, pin_memory=config.pin_memory)
    return train_loader, val_loader
        

def initialize_model(fold, config, stage='pre'):
    model, teacher_model = None, None
    if stage in ['pre', 'full']:
        teacher_model = RANZCRClassifier(config.model_name, in_chans=config.in_chans, 
                                     num_classes=config.num_classes, drop_path_rate=config.drop_path_rate)
        teacher_model.load_state_dict(load_checkpoint(f"{config.teacher_dir}/fold={fold}-best.pth")['model'])
        for param in teacher_model.parameters():
            param.requires_grad = False
        teacher_model.to(config.device)
        teacher_model.eval()

        model = RANZCRClassifier(config.model_name, pretrained=config.pretrained,
                                checkpoint_path=config.checkpoint_path, 
                                in_chans=config.in_chans, num_classes=config.num_classes,
                                drop_path_rate=config.drop_path_rate)
        model = model.to(config.device)
    elif stage == 'post':
        model = RANZCRClassifier(config.model_name, pretrained=config.pretrained,
                             checkpoint_path=config.checkpoint_path, 
                             in_chans=config.in_chans, num_classes=config.num_classes,
                             drop_path_rate=config.drop_path_rate)
        model.load_state_dict(load_checkpoint(f"{config.checkpoint_dir}/fold={fold}-best-pre.pth")['model'])
        model = model.to(config.device)
    else:
        raise ValueError(f'Stage must be either pre or post or full, not {stage}')
    return model, teacher_model


def initialize_criterion(config, stage='pre'):
    train_criterion, val_criterion = None, None
    if stage in ['pre', 'full']:
        train_criterion = CustomLoss(weights=config.weights)
        val_criterion = FocalLoss() # nn.BCEWithLogitsLoss() #
    elif stage == 'post':
        train_criterion = FocalLoss() # nn.BCEWithLogitsLoss() #
        val_criterion = FocalLoss() # nn.BCEWithLogitsLoss() #
    else:
        raise ValueError(f'Stage must be either pre or post or full, not {stage}')
    return train_criterion, val_criterion


def train_one_epoch(model, teacher_model, loader, criterion, optimizer, scaler, config, stage='pre'):
    model.train()
    running_loss = AverageMeter()
    tepoch = tqdm(loader)
    # with tqdm(total=len(loader)) as tepoch:
    for batch_idx, data in enumerate(tepoch):
        if stage == 'pre':
            data, annot_data, targets = data
            data = data.to(config.device)
            annot_data = annot_data.to(config.device) 
            targets = targets.to(config.device)
            
            # Get teacher model's features
            with torch.no_grad():
                teacher_features, _ = teacher_model(annot_data)

            # Forward pass
            with torch.cuda.amp.autocast():
                features, outputs = model(data)
                loss = criterion(features, teacher_features, outputs, targets)
        elif stage == 'post':
            data, targets = data
            data = data.to(config.device)
            targets = targets.to(config.device)
            
            # Forward pass
            with torch.cuda.amp.autocast():
                _, outputs = model(data)
                loss = criterion(outputs, targets)
        elif stage == 'full':
            data, annot_data, targets = data
            data = data.to(config.device)
            annot_data = annot_data.to(config.device) 
            targets = targets.to(config.device)
            has_annot = torch.any(torch.ne(data.reshape((data.shape[0], -1)), 
                                           annot_data.reshape((annot_data.shape[0], -1))), dim=1)
            annot_data = annot_data[has_annot]
            
            # Get teacher model's features
            with torch.no_grad():
                teacher_features, _ = teacher_model(annot_data)

            # Forward pass
            with torch.cuda.amp.autocast():
                features, outputs = model(data)
                loss = criterion(features[has_annot], teacher_features, outputs, targets)
        else:
            raise ValueError(f'Stage must be either pre or post or full, not {stage}')

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


def run(fold, config, stage='pre'):
    # Prepare train and val set
    train_loader, val_loader = initialize_loader(fold, config, stage)

    # Initialize model
    model, teacher_model = initialize_model(fold, config, stage)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of params: {n_parameters}")
    # print(f"LR: {config.learning_rate}")

    # Set up training
    start_epoch = 0
    train_criterion, val_criterion = initialize_criterion(config, stage)
    optimizer = eval(f"optim.{config.optimizer}(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)")
    scheduler = eval(f"optim.lr_scheduler.{config.lr_scheduler}(optimizer, **config.lr_scheduler_params)")
    scaler = torch.cuda.amp.GradScaler()
    writer = SummaryWriter()
    postfix = f'-{stage}'

    # Load checkpoint
    if config.resume:
        checkpoint = load_checkpoint(fold=fold, checkpoint_dir=config.checkpoint_dir, postfix=postfix)
        if checkpoint:
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            scaler.load_state_dict(checkpoint['scaler'])
            start_epoch = checkpoint['epoch'] + 1
    print(f"LR: {scheduler.get_last_lr()[0]}")

    best_auc = 0
    for epoch in range(start_epoch, config.n_epochs[stage]):
        print(f'Fold: {fold}  Epoch: {epoch}')
        train_loss = train_one_epoch(model, teacher_model, train_loader, train_criterion, optimizer, scaler, config, stage)
        val_loss, auc = valid_one_epoch(model, val_loader, val_criterion, config)
        scheduler.step()
        print(f'AUC: {auc:.4f}')
        print(f'New LR: {scheduler.get_last_lr()[0]}')

        # Log to file
        log_stats = {
            'fold': fold, 'epoch': epoch, 'n_parameters': n_parameters,
            'train_loss': train_loss, 'val_loss': val_loss, 'val_auc': auc,
        }
        log_to_file(log_stats, f"log{postfix}.txt", config.checkpoint_dir)

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
        save_path = f"{config.checkpoint_dir}/fold={fold}-epoch={epoch}-auc={auc:.4f}{postfix}.pth"
        save_checkpoint(checkpoint, save_path)
        if auc > best_auc:
            best_auc = auc
            checkpoint['auc'] = auc
            save_path = f"{config.checkpoint_dir}/fold={fold}-best{postfix}.pth"
            print('--> Saving checkpoint')
            save_checkpoint(checkpoint, save_path)

        # Delete old checkpoint
        for fname in glob.glob(f"{config.checkpoint_dir}/fold={fold}-epoch={epoch-config.save_freq}*{postfix}.pth"):
            os.remove(fname)


def main():
    set_seed(config.seed)
    if os.path.exists('kaggle/input'):
        config.input_dir = '../input/ranzcr-clip-catheter-line-classification'

    # Train model
    for fold in range(config.kfold):
        if config.n_stages == 1:
            run(fold, config, stage='full')
        elif config.n_stages == 2:
            run(fold, config, stage='pre')
            run(fold, config, stage='post')

if __name__ == "__main__":
    main()