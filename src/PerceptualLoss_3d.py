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


# torch.Size([1, 100, 512, 512])
class PerceptualLoss_3d():
    def __init__(self, feature_extractor, loss_criterion):
        self.feature_extractor = feature_extractor
        self.loss_criterion = nn.MSELoss()
        self.PerceptualLoss = PerceptualLoss(self.feature_extractor,
                                             self.loss_criterion)

    def _get_2d_loss(self, output, target):
        rgb_output = output.expand(-1, 3, -1, -1)
        rgb_target = target.expand(-1, 3, -1, -1)
        loss = self.PerceptualLoss.get_loss(rgb_output, rgb_target)
        return loss

    def _get_loss_of_volume(self, input_volume, mask_volume, axis):
        running_loss = 0.0
        for index in range(input_volume.shape[axis]):
            # Slice the scan based on the axis
            if axis == 1:
                input_slice = input_volume[:, index, :, :]
                mask_slice = mask_volume[:, index, :, :]
            elif axis == 2:
                input_slice = input_volume[:, :, index, :]
                mask_slice = mask_volume[:, :, index, :]
            elif axis == 3:
                input_slice = input_volume[:, :, :, index]
                mask_slice = mask_volume[:, :, :, index]
            else:
                raise ValueError("Invalid axis. Axis must be between 0 and 3.x")

            loss = self._get_2d_loss(input_slice, mask_slice)
            running_loss += loss

        return running_loss

    def get_loss(self, output, mask):

        x_loss = self._get_loss_of_volume(output, mask, 3)
        y_loss = self._get_loss_of_volume(output, mask, 2)
        z_loss = self._get_loss_of_volume(output, mask, 1)

        total_loss = x_loss + y_loss + z_loss

        return total_loss
