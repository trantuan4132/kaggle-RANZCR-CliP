import cv2
import torch
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('.')

from dataset import RANZCRDataset, build_transform
from model import RANZCRClassifier
from utils import load_checkpoint


class config:
    # Data
    image_paths = [
        'train/1.2.826.0.1.3680043.8.498.10010621324226224265011850078370952894.jpg',
        # 'train/1.2.826.0.1.3680043.8.498.40203216203080904288333836838469265651.jpg'
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
    weight_path = ''                         # Path to model's pretrained weights
    checkpoint_path = 'student_checkpoint/fold=0-best-full.pth'


def main():
    transform = [
        build_transform(image_size=config.image_size, adjust_color=False, is_train=False, include_top=False),
        build_transform(image_size=None, adjust_color=False, is_train=False, include_top=True),
    ]
    images, input_tensor = [], []
    for image_path in config.image_paths:
        image = cv2.imread(image_path)[:, :, ::-1]
        image = transform[0](image=image)['image']
        images.append(image)
        input_tensor.append(transform[-1](image=image)['image'].unsqueeze(0))
    input_tensor = torch.cat(input_tensor, dim=0)
    input_tensor = input_tensor.to(config.device)

    # Load model
    model = RANZCRClassifier(config.model_name, pretrained=config.pretrained,
                             checkpoint_path=config.weight_path, 
                             in_chans=config.in_chans, num_classes=config.num_classes,
                             drop_path_rate=config.drop_path_rate, return_features=False)
    model = model.to(config.device)
    model.load_state_dict(load_checkpoint(config.checkpoint_path)['model'])

    # Inference
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
    outputs = (outputs.sigmoid().cpu().numpy() > 0.5).astype('int')
    targets = [[ClassifierOutputTarget(out) for out in np.where(output==1)[0]] for output in outputs]

    # CAM Visualization
    target_layers = [model.model.head.norm] # [model.model.stages[-1].blocks[-1].norm]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=(config.device=='cuda'))
    for idx in range(len(input_tensor)):
        tensor = torch.cat([input_tensor[idx].unsqueeze(0)])*len(targets[idx])
        grayscale_cams = cam(input_tensor=tensor, targets=targets[idx])
        n_rows = 1
        n_cols = 1 + len(targets[idx]) // n_rows
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*4))
        axes.ravel()[0].imshow(images[idx])
        axes.ravel()[0].set_title('Original Image')
        axes.ravel()[0].axis('off')
        for i, grayscale_cam in enumerate(grayscale_cams, start=1):
            visualization = show_cam_on_image(images[idx]/255, grayscale_cam, use_rgb=True)
            axes.ravel()[i].imshow(visualization)
            axes.ravel()[i].set_title(config.label_cols[targets[idx][i-1].category])
            axes.ravel()[i].axis('off')
        plt.show()


if __name__ == "__main__":
    main()