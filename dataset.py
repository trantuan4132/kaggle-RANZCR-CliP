import os
import cv2
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from utils import draw_annotation


class RANZCRDataset(Dataset):
    def __init__(self, image_dir, df, img_col, label_cols, df_annot=None, 
                 color_map=None, transform=None, prev_transform=None, return_img='image'):
        """
        Args:
        -----
        image_dir: str
            Path to the image directory
        df: pandas dataframe
            Dataframe containing the image ids and labels
        img_col: str
            Name of the column containing the image id
        label_cols: list
            List of the names of the columns containing the labels
        df_annot: pandas dataframe, optional
            Dataframe containing the annotations
        color_map: dict, optional
            Dictionary containing the color for each class
        transform: albumentations transform, optional
            Albumentations transform to apply to the image
        prev_transform: albumentations transform, optional
            Albumentations transform to apply to the image before applying the transform
        return_img: str, default 'image'
            Whether to return the image ('image') or the annotation image ('annot_image') or both ('both')
        """
        super(RANZCRDataset, self).__init__()
        self.image_dir = image_dir
        self.df = df
        self.df_annot = df_annot
        self.color_map = color_map
        self.img_col = img_col
        self.label_cols = label_cols
        self.transform = transform
        self.prev_transform = prev_transform
        self.return_img = return_img 

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if self.df is not None:
            row = self.df.iloc[index]
            image = cv2.imread(f'{self.image_dir}/{row[self.img_col]}.jpg')[:, :, ::-1].astype(np.uint8)
            label = row[self.label_cols].values.astype('float')
        else:
            image = cv2.imread(f'{self.image_dir}/{os.listdir(self.image_dir)[index]}')[:, :, ::-1].astype(np.uint8)
            label = np.zeros(len(self.label_cols))

        if self.prev_transform:
            image = self.prev_transform(image=image)['image']

        # Draw annotation on image if available
        annot_image = None
        if self.df_annot is not None and self.color_map:
            if self.return_img == 'both':
                annot_image = image.copy()
                annot_image = draw_annotation(annot_image, self.df_annot, self.img_col, 
                                              row[self.img_col], self.color_map)
            elif self.return_img == 'annot_image':
                image = draw_annotation(image, self.df_annot, self.img_col, 
                                        row[self.img_col], self.color_map)    

        if self.transform:
            if annot_image is not None:
                transformed = self.transform(image=image, annot_image=annot_image)
                image, annot_image = transformed['image'], transformed['annot_image']
            else:
                image = self.transform(image=image)['image']
        return (image, label) if annot_image is None else (image, annot_image, label)


def build_transform(image_size=None, adjust_color=True, is_train=True, include_top=True, additional_targets=None):
    """
    Builds a transformations pipeline for the data.

    Args:
    -----
    image_size: int, optional
        The size of the image to be transformed.
    adjust_color: bool, optional
        Whether to randomly adjust the color of the image.
    is_train: bool, optional
        Whether the data is being used for training.
    include_top: bool, optional
        Whether to normalize and convert to tensor.
    additional_targets: dict, optional
        A dictionary of additional targets to be applied same transformation as the image.
    """
    transform = []
    if image_size:
        transform.append(A.Resize(image_size, image_size))
    image_size = image_size or 40
    if adjust_color:
        transform.extend([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=40, sat_shift_limit=40, val_shift_limit=0, p=0.5),
        ])
    if is_train:
        transform.extend([
            A.HorizontalFlip(p=0.5),
            # A.OneOf([
            #   A.ImageCompression(),
            #   A.Downscale(scale_min=0.1, scale_max=0.15),
            # ], p=0.2),
            # A.PiecewiseAffine(p=0.2),
            # A.Sharpen(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.3, rotate_limit=30, border_mode=0, p=0.5),
            A.CoarseDropout(max_height=int(image_size*0.2), max_width=int(image_size*0.2), 
                            min_holes=1, max_holes=4, p=0.5),
        ])
    if include_top:
        transform.extend([
            A.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
            ToTensorV2(),
        ])
    return A.Compose(transform, additional_targets=additional_targets)


if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt

    img_col = 'StudyInstanceUID'
    label_cols = [
        'ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal', 
        'NGT - Abnormal', 'NGT - Borderline', 'NGT - Incompletely Imaged', 'NGT - Normal', 
        'CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal', 
        'Swan Ganz Catheter Present',
    ]
    df = pd.read_csv('train.csv').iloc[[1994, 2601, 8783]]
    df_annot = pd.read_csv('train_annotations.csv')
    # color_map = {
    #     'ETT - Abnormal': (255, 255, 0),
    #     'ETT - Borderline': (255, 255, 0),
    #     'ETT - Normal': (255, 255, 0),
    #     'NGT - Abnormal': (255, 0, 255),
    #     'NGT - Borderline': (255, 0, 255),
    #     'NGT - Incompletely Imaged': (255, 0, 255),
    #     'NGT - Normal': (255, 0, 255),
    #     'CVC - Abnormal': (0, 255, 255),
    #     'CVC - Borderline': (0, 255, 255),
    #     'CVC - Normal': (0, 255, 255),
    #     'Swan Ganz Catheter Present': (0, 0, 255),
    #     'tip': (255, 0, 0),
    # }
    color_map = {
        'ETT': (255, 255, 0),
        'NGT': (255, 0, 255),
        'CVC': (0, 255, 255),
        'Swan Ganz Catheter Present': (0, 128, 128),
        'Normal': (255, 0, 0),
        'Borderline': (0, 255, 0),
        'Abnormal': (0, 0, 255),
    }
    n_rows = 1
    n_cols = 2
    prev_transform = build_transform(image_size=None, adjust_color=True, is_train=False, include_top=False)
    transform = build_transform(image_size=512, adjust_color=False, is_train=True, 
                                include_top=False, additional_targets={'annot_image': 'image'})
    # transform = build_transform(image_size=512, is_train=True, include_top=False)
    n_samples = n_rows * n_cols
    sample_dataset = RANZCRDataset(image_dir="train", df=df.sample(n_samples), 
                                   img_col=img_col, label_cols=label_cols,
                                   df_annot=df_annot, color_map=color_map, 
                                   transform=transform, prev_transform=prev_transform, return_img='both')
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*4))
    for i in range(n_samples):
        img, annot_img, label = sample_dataset[i]
        axes.ravel()[i].imshow(annot_img)
    plt.show()