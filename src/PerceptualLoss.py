import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights


class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG16FeatureExtractor, self).__init__()

        vgg16_inst = vgg16(weights=VGG16_Weights.DEFAULT).features[:16]

        for param in vgg16_inst.parameters():
            param.requires_grad = False

        self.features = nn.Sequential(*list(vgg16_inst.children()))

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


class CombinedLoss(nn.Module):
    def __init__(self, criterion, FeatureExtractor, alpha=1.0, beta=1.0):
        super(CombinedLoss, self).__init__()
        self.feature_extractor = FeatureExtractor
        self.criterion = criterion
        self.alpha = alpha
        self.beta = beta

    def forward(self, output, target):

        criterion_loss = self.criterion(output, target)

        rgb_output = output.expand(-1, 3, -1, -1)
        rgb_target = target.expand(-1, 3, -1, -1)
        output_features = self.feature_extractor(rgb_output)
        target_features = self.feature_extractor(rgb_target)

        perceptual_loss = self.criterion(output_features, target_features)

        total_loss = self.alpha * criterion_loss + self.beta * perceptual_loss
        return total_loss
