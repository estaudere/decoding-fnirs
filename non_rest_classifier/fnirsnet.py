"""Implementation of fNIRSNet model, from https://github.com/wzhlearning/fNIRSNet?tab=readme-ov-file"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DWSConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(DWSConv, self).__init__()
        self.depth_conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=0, groups=in_channels)
        self.point_conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1)

    def forward(self, input):
        x = self.depth_conv(input)
        x = self.point_conv(x)
        return x


class fNIRSNet(torch.nn.Module):
    """
    fNIRSNet model

    Args:
        num_class: Number of classes.
        DHRConv_width: Width of DHRConv = width of fNIRS signals.
        DWConv_height: Height of DWConv = height of 2 * fNIRS channels, and '2' means HbO and HbR.
        num_DHRConv: Number of channels for DHRConv.
        num_DWConv: number of channels for DWConv.
    """
    def __init__(self, num_class, DHRConv_width=186, DWConv_height=42, num_DHRConv=4, num_DWConv=8):
        super(fNIRSNet, self).__init__()
        # DHR Module
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=num_DHRConv, kernel_size=(1, DHRConv_width), stride=1, padding=0)
        self.bn1 = torch.nn.BatchNorm2d(num_DHRConv)

        # Global Module
        self.conv2 = DWSConv(in_channels=num_DHRConv, out_channels=num_DWConv, kernel_size=(DWConv_height, 1))
        self.bn2 = torch.nn.BatchNorm2d(num_DWConv)

        self.fc = torch.nn.Linear(num_DWConv, num_class)
        self.act = torch.nn.Sigmoid()

    def forward(self, x):
        x = x.view(x.size()[0], 1, x.size()[1], x.size()[2]) # turn to (batch_size, 1, channels, timesteps)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x
    
    
class LabelSmoothing(torch.nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()