from .resnet import resnet18, resnet50
from torch import nn
import torch
import os


class BaseModel(nn.Module):
    def __init__(self, num_classes=9, base='resnet18'):
        super(BaseModel, self).__init__()
        if base == 'resnet50':
            self.base = resnet50(pretrained=True)
            out_feature_size = 2048
        elif base == 'resnet18':
            self.base = resnet18(pretrained=True)
            out_feature_size = 512
        self.classifier = torch.nn.Linear(out_feature_size, num_classes)

    def forward(self, x):
        x = self.base(x)
        x = self.classifier(x)
        return x

    def save_model(self, save_path):
        torch.save(self.base.state_dict(), os.path.join(save_path, 'best_model.pth'))
        torch.save(self.classifier.state_dict(), os.path.join(save_path, 'best_classifier.pth'))

    def renew_model(self, save_path):
        net_path = os.path.join(save_path, 'best_model.pth')
        classifier_path = os.path.join(save_path, 'best_classifier.pth')
        self.base.load_state_dict(torch.load(net_path))
        self.classifier.load_state_dict(torch.load(classifier_path))
