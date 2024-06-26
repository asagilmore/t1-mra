import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision.models import vgg16, VGG16_Weights
import numpy as np

vgg16 = vgg16(weights=VGG16_Weights.DEFAULT).features
for param in vgg16.parameters():
    param.requires_grad = False

class VGG16FeatureExtractor(nn.Module):
    def __init__(self):

        vgg16 = models.vgg16(pretrained=True).features

        for param in vgg16.parameters():
            param.requires_grad = False

        super(VGG16FeatureExtractor, self).__init__()
        self.features = vgg16[:16]

    def forward(self, x):
        return self.features(x)


class PerceptualLoss():
    def __init__(self, feature_extractor, loss_criterion):
        self.feature_extractor = feature_extractor
        self.loss_criterion = nn.MSELoss()

    def get_loss(self, output, target):
        rgb_output = output.expand(-1, 3, -1, -1)
        rgb_target = target.expand(-1, 3, -1, -1)
        out_features = self.feature_extractor(rgb_output)
        target_features = self.feature_extractor(rgb_target)

        return self.loss_criterion(out_features, target_features)