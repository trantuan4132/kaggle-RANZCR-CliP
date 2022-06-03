import torch
import torch.nn as nn
import timm

class RANZCRClassifier(nn.Module):
    def __init__(self, model_name, pretrained=False, checkpoint_path='', 
                 in_chans=3, num_classes=1000, drop_path_rate=0.0):
        super(RANZCRClassifier, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained,
                                       checkpoint_path=checkpoint_path,
                                       drop_path_rate=drop_path_rate)
        self.model.reset_classifier(num_classes=num_classes)

    def forward(self, x):
        x = self.model(x)
        return x