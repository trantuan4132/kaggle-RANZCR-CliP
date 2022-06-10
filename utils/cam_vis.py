import cv2
import torch
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt

from dataset import RANZCRDataset, build_transform
from model import RANZCRClassifier
from utils import load_checkpoint


class config:
    # Data
    image_paths = [
        'train/1.2.826.0.1.3680043.8.498.10010621324226224265011850078370952894.jpg'
    ]
    # input_dir = '.'
    # img_col = 'StudyInstanceUID'
    label_cols = [
        'ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal', 
        'NGT - Abnormal', 'NGT - Borderline', 'NGT - Incompletely Imaged', 'NGT - Normal', 
        'CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal', 
        'Swan Ganz Catheter Present',
    ]
    # batch_size = 32
    image_size = 512
    # num_workers = 2
    # pin_memory = True
    # seed = 42

    # Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = 'convnext_tiny'
    in_chans = 3
    num_classes = len(label_cols)
    drop_path_rate = 0.1
    pretrained = False                       # True: load pretrained model, False: train from scratch
    # weight_path = ''                         # Path to model's pretrained weights
    checkpoint_path = ''


def main():
    transform = [
        build_transform(image_size=config.image_size, adjust_color=False, is_train=False, include_top=False),
        build_transform(image_size=None, adjust_color=False, is_train=False, include_top=True),
    ]
    images, input_tensor = [], []
    for image_path in config.image_paths:
        image = cv2.imread(image_path)[:, :, ::-1]
        images.append(transform[0](image))
        input_tensor.append(transform[-1](image).unsqueeze(0))
    input_tensor = torch.cat(input_tensor, dim=0)
    input_tensor = input_tensor.to(config.device)

    # Load model
    model = RANZCRClassifier(config.model_name, pretrained=config.pretrained,
                             checkpoint_path=config.checkpoint_path, 
                             in_chans=config.in_chans, num_classes=config.num_classes,
                             drop_path_rate=config.drop_path_rate)
    model = model.to(config.device)
    model.load_state_dict(load_checkpoint(config.checkpoint_path['model']))

    # CAM Visualization
    target_layers = [model.head.norm] # [model.stages[-1].blocks[-1].norm]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=(config.device=='cuda'))
    grayscale_cams = cam(input_tensor=input_tensor, targets=None)
    n_rows = 1
    n_cols = len(images) // n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*4))
    for i, image, grayscale_cam in enumerate(zip(images, grayscale_cams)):
        visualization = show_cam_on_image(image/255, grayscale_cam, use_rgb=True)
        axes.ravel()[i].imshow(visualization)
    plt.show()