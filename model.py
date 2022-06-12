import torch
import torch.nn as nn
import timm

class RANZCRClassifier(nn.Module):
    def __init__(self, model_name, pretrained=False, checkpoint_path='', 
                 in_chans=3, num_classes=1000, drop_path_rate=0.0, return_features=True):
        super(RANZCRClassifier, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained,
                                       checkpoint_path=checkpoint_path,
                                       drop_path_rate=drop_path_rate)
        n_features = self.model.get_classifier().in_features
        self.model.reset_classifier(num_classes=0, global_pool='')
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(n_features, num_classes)
        self.return_features = return_features

    def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        pooled_features = self.pooling(features).view(bs, -1)
        output = self.fc(pooled_features)
        return features, output if self.return_features else output


if __name__ == '__main__':
    model = RANZCRClassifier('convnext_tiny', pretrained=False, 
                             checkpoint_path='convnext_tiny_22k_1k_384_altered.pth')
    # print(model(torch.randn(32, 3, 224, 224)))
    print(model)