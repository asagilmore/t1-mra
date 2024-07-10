import torch.nn as nn
import torch


class Critic(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.GELU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512, affine=True),
            nn.GELU(),
            nn.Conv2d(512, 1, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x):
        return self.model(x).view(-1)


def gradient_penalty(critic, real_images, fake_images):
    batch_size, c, h, w = real_images.size()
    epsilon = torch.rand(batch_size, 1, 1, 1,
                         requires_grad=True).to(real_images.device)
    epsilon = epsilon.expand_as(real_images)

    interpolated_images = epsilon * real_images + (1 - epsilon) * fake_images

    mixed_scores = critic(interpolated_images)
    grad_outputs = torch.ones_like(mixed_scores, requires_grad=False)

    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
    )[0]

    gradient = gradient.view(gradient.size(0), -1)
    gradient_norm = gradient.norm(2, dim=1)
    penalty = torch.mean((gradient_norm - 1) ** 2)
    return penalty
