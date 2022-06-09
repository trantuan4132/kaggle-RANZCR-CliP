import torch
from torch.utils.data import DataLoader
import timm
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

from dataset import RANZCRDataset, build_transform
from model import RANZCRClassifier
from utils import load_checkpoint

class config:
    # Data
    input_dir = '.'
    img_col = 'StudyInstanceUID'
    label_cols = [
        'ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal', 
        'NGT - Abnormal', 'NGT - Borderline', 'NGT - Incompletely Imaged', 'NGT - Normal', 
        'CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal', 
        'Swan Ganz Catheter Present',
    ]
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
    checkpoint_dirs = {'convnext_tiny': ['student_checkpoint/fold=0-best-pre.pth',
                                         'student_checkpoint/fold=1-best-pre.pth',
                                         'student_checkpoint/fold=2-best-pre.pth',
                                         'student_checkpoint/fold=3-best-pre.pth',
                                         'student_checkpoint/fold=4-best-pre.pth']}
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
    test_df = pd.read_csv(f'{config.input_dir}/sample_submission.csv')

    if config.debug:
        test_df = test_df.iloc[:100]

    test_trainsform = build_transform(config.image_size, adjust_color=False, is_train=False, include_top=True)
    test_dataset = RANZCRDataset(image_dir=f"{config.input_dir}/test", df=test_df,
                                 img_col=config.img_col, label_cols=config.label_cols,
                                 transform=test_trainsform)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
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
        preds.append(predict(model, test_loader, config))
    preds = np.mean(preds, axis=0)
    preds = (preds > 0.5).astype('int')
    test_df[config.label_cols] = preds
    test_df.to_csv('submission.csv', index=False)
    print(test_df)


if __name__ == '__main__':
    main()