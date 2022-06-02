from operator import imod
import cv2
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class RANZCRDataset(Dataset):
    def __init__(self, image_dir, df, img_col, label_cols, transform=None):
        super(RANZCRDataset, self).__init__()
        self.image_dir = image_dir
        self.df = df
        self.img_col = img_col
        self.label_cols = label_cols
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image = cv2.imread(f'{self.image_dir}/{row[self.img_col]}.jpg')[:, :, ::-1]
        if self.transform:
            image = self.transform(image=image)['image']
        label = row[self.label_cols].values.astype('float')
        return image, label


def build_transform(is_train, config):
    if is_train:
        return A.Compose([
            A.Resize(config.image_size, config.image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            # A.OneOf([
            #   A.ImageCompression(),
            #   A.Downscale(scale_min=0.1, scale_max=0.15),
            # ], p=0.2),
            # A.PiecewiseAffine(p=0.2),
            # A.Sharpen(p=0.2),
            A.HueSaturationValue(hue_shift_limit=40, sat_shift_limit=40, val_shift_limit=0, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.3, rotate_limit=30, border_mode=0, p=0.5),
            A.CoarseDropout(max_height=int(config.image_size*0.2), max_width=int(config.image_size*0.2), 
                            min_holes=1, max_holes=4, p=0.5),
            A.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
            ToTensorV2(),                        
        ])
    else:
        return A.Compose([
            A.Resize(config.image_size, config.image_size),
            A.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
            ToTensorV2(), 
        ])