import torch
import torch.nn as nn
from torchvision import models
from torch.nn.functional import gelu, relu


class UNet(nn.Module):
    """
    U-Net Model Architecture

    See: https://arxiv.org/abs/1606.06650
    """

    def __init__(self, input_dim, output_dim, batch_size, input_channels=1, output_channels=1):
        super().__init__()
        conv_kwargs = {
            "kernel_size": (3, 3),
            "padding": "same",
        }

        # Decoder convolutional transpose keyword arguments
        conv_trans_kwargs = {
            "kernel_size": (2, 2),
            "stride": 2,
            "padding": "same"
        }

        # encoder
        self.e1a = nn.Conv2d(input_channels, 64, **conv_kwargs)
        self.e1b = nn.Conv2d(64, 64, **conv_kwargs)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 284x284x64

        self.e2a = nn.Conv2d(64, 128, **conv_kwargs)
        self.e2b = nn.Conv2d(128, 128, **conv_kwargs)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e3a = nn.Conv2d(128, 256, **conv_kwargs)
        self.e3b = nn.Conv2d(256, 256, **conv_kwargs)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e4a = nn.Conv2d(256, 512, **conv_kwargs)
        self.e4b = nn.Conv2d(512, 512, **conv_kwargs)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e5a = nn.Conv2d(512, 1024, **conv_kwargs)
        self.e5b = nn.Conv2d(1024, 1024, **conv_kwargs)


        # decoder
        self.upconv1 = nn.ConvTranspose2d(1024, 512, **conv_trans_kwargs)
        self.d1a = nn.Conv2d(1024, 512, **conv_kwargs)
        self.d1b = nn.Conv2d(512, 512, **conv_kwargs)

        self.upconv2 = nn.ConvTranspose(512, 256, **conv_trans_kwargs)
        self.d2a = nn.Conv2d(512, 256, **conv_kwargs)
        self.d2b = nn.Conv2d(256, 256, **conv_kwargs)

        self.upconv3 = nn.ConvTranspose(256, 128, **conv_trans_kwargs)
        self.d3a = nn.Conv2d(256, 128, **conv_kwargs)
        self.d3b = nn.Conv2d(128, 128, **conv_kwargs)

        self.upconv4 = nn.ConvTranspose(128, 64, **conv_trans_kwargs)
        self.d4a = nn.Conv2d(128, 64, **conv_kwargs)
        self.d4b = nn.Conv2d(64, 64, **conv_kwargs)

        # output
        self.out = nn.Conv2d(64, output_channels, **conv_kwargs)

    def forward(self, x, act_funct=gelu):

        #encoding
        x = act_funct(self.e1a(x))
        x = act_funct(self.e1b(x))
        skip1 = x
        x = self.pool1(x)

        x = act_funct(self.e2a(x))
        x = act_funct(self.e2b(x))
        skip2 = x
        x = self.pool2(x)

        x = act_funct(self.e3a(x))
        x = act_funct(self.e3b(x))
        skip3 = x
        x = self.pool3(x)

        x = act_funct(self.e4a(x))
        x = act_funct(self.e4b(x))
        skip4 = x
        x = self.pool4(x)

        x = act_funct(self.e5a(x))
        x = act_funct(self.e5b(x))

        #decoding
        x = self.upconv1(x)
        x = torch.cat((x, skip4), dim=1)
        x = act_funct(self.d1a(x))
        x = act_funct(self.d1b(x))

        x = self.upconv2(x)
        x = torch.cat((x, skip3), dim=1)
        x = act_funct(self.d2a(x))
        x = act_funct(self.d2b(x))

        x = self.upconv3(x)
        x = torch.cat((x, skip2), dim=1)
        x = act_funct(self.d3a(x))
        x = act_funct(self.d3b(x))

        x = self.upconv4(x)
        x = torch.cat((x, skip1), dim=1)
        x = act_funct(self.d4a(x))
        x = act_funct(self.d4b(x))
        out = self.out(x)

        return out