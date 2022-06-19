import torch
from torch.utils.data import DataLoader
import timm
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import sys
sys.path.append('.')

from dataset import RANZCRDataset, build_transform
from model import RANZCRClassifier
from utils import load_checkpoint

class config:
    # Data
    input_dir = '..'
    image_dir = f'{input_dir}/train'
    df_path = f'{input_dir}/train_fold5.csv' # 'sample_submission.csv' #
    use_annotation = True
    df_annot_path = f'{input_dir}/train_annotations.csv'
    fold = 0
    img_col = 'StudyInstanceUID'
    label_cols = [
        'ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal', 
        'NGT - Abnormal', 'NGT - Borderline', 'NGT - Incompletely Imaged', 'NGT - Normal', 
        'CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal', 
        'Swan Ganz Catheter Present',
    ]
    mode = 'compare' # 'predict' #
    batch_size = 32
    image_size = 512
    num_workers = 2
    pin_memory = True
    seed = 42

    # Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    in_chans = 3
    num_classes = len(label_cols)
    drop_path_rate = 0.1
    pretrained = False                       # True: load pretrained model, False: train from scratch
    checkpoint_path = ''                    # Path to model's pretrained weights
    checkpoint_dirs = {'convnext_tiny': [
                            'student_checkpoint/fold=0-best-post.pth',
                            # 'student_checkpoint/fold=1-best-post.pth',
                            # 'student_checkpoint/fold=2-best-post.pth',
                            # 'student_checkpoint/fold=3-best-post.pth',
                            # 'student_checkpoint/fold=4-best-post.pth',
                        ]}
    # checkpoint_dirs = {'convnext_tiny': [
    #                         'student_checkpoint/fold=0-best-full.pth',
    #                         'student_checkpoint/fold=1-best-full.pth',
    #                         'student_checkpoint/fold=2-best-full.pth',
    #                         'student_checkpoint/fold=3-best-full.pth',
    #                         'student_checkpoint/fold=4-best-full.pth',
    #                     ]}
    debug = True


def predict(model, loader, config):
    model.eval()
    preds = []
    tepoch = tqdm(loader)
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(tepoch):
            data = data.to(config.device)
            _, outputs = model(data)
            preds.append(outputs)
    return torch.cat(preds).sigmoid().cpu().numpy()


def main():
    # Load data
    df = pd.read_csv(config.df_path)
    trainsform = build_transform(config.image_size, adjust_color=False, is_train=False, include_top=True)

    # Draw annotation if specified
    df_annot, color_map, return_img = None, None, 'image'
    if config.use_annotation:
        df_annot = pd.read_csv(config.df_annot_path)
        df = df.set_index(config.img_col).loc[df_annot[config.img_col].unique()].reset_index()
        color_map = {
            'ETT': (255, 255, 0),
            'NGT': (255, 0, 255),
            'CVC': (0, 255, 255),
            'Swan Ganz Catheter Present': (0, 128, 128),
            'Normal': (255, 0, 0),
            'Borderline': (0, 255, 0),
            'Abnormal': (0, 0, 255),
        }
        return_img = 'annot_image'

    # Select fold if specified
    if config.fold is not None:
        df = df.query(f'fold=={config.fold}')

    dataset = RANZCRDataset(image_dir=config.image_dir, df=df,
                            img_col=config.img_col, label_cols=config.label_cols,
                            df_annot=df_annot, color_map=color_map, 
                            transform=trainsform, return_img=return_img)
    loader = DataLoader(dataset, batch_size=config.batch_size,
                        num_workers=config.num_workers, pin_memory=config.pin_memory,
                        shuffle=False)

    # Load model
    models = []
    for model_name, checkpoint_dirs in config.checkpoint_dirs.items():
        for checkpoint_dir in checkpoint_dirs:
            # Initialize model
            model = RANZCRClassifier(model_name, pretrained=config.pretrained,
                                     checkpoint_path=config.checkpoint_path, 
                                     in_chans=config.in_chans, num_classes=config.num_classes,
                                     drop_path_rate=config.drop_path_rate)
            model = model.to(config.device)

            # Load weights
            checkpoint = load_checkpoint(checkpoint_dir)
            if 'auc' in checkpoint:
                print(f"AUC: {checkpoint['auc']}")
            model.load_state_dict(checkpoint['model'])
            models.append(model)

    # Predict
    preds = []
    for model in models:
        preds.append(predict(model, loader, config))
    preds = np.mean(preds, axis=0)
    preds = (preds > 0.5).astype('int')
    if config.mode == 'compare':
        # Compare
        preds = pd.DataFrame(preds, columns=config.label_cols, index=df[config.img_col].values)
        df_compare = df.set_index(config.img_col)[config.label_cols].compare(preds)
        df_compare.columns = ["_".join(a) for a in df_compare.columns.to_flat_index()]
        df_compare.to_csv(f'fold-{config.fold}-compare.csv')
        print(df_compare)
    elif config.mode == 'predict':
        # Save predictions
        df[config.label_cols] = preds
        df.to_csv('submission.csv', index=False)
        print(df)


if __name__ == '__main__':
    main()