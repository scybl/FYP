"""
Paper:      Using DUCK-Net for Polyp Image Segmentation
Url:        https://arxiv.org/abs/2311.02239
Create by:  zh320
Date:       2024/11/08
GitHub: https://github.com/zh320/medical-segmentation-pytorch
"""

import torch.nn as nn
import torch.nn.functional as F

model_hub = {}


class Activation(nn.Module):
    def __init__(self, act_type, **kwargs):
        super().__init__()
        activation_hub = {'relu': nn.ReLU, 'relu6': nn.ReLU6,
                          'leakyrelu': nn.LeakyReLU, 'prelu': nn.PReLU,
                          'celu': nn.CELU, 'elu': nn.ELU,
                          'hardswish': nn.Hardswish, 'hardtanh': nn.Hardtanh,
                          'gelu': nn.GELU, 'glu': nn.GLU,
                          'selu': nn.SELU, 'silu': nn.SiLU,
                          'sigmoid': nn.Sigmoid, 'softmax': nn.Softmax,
                          'tanh': nn.Tanh, 'none': nn.Identity,
                          }

        act_type = act_type.lower()
        if act_type not in activation_hub.keys():
            raise NotImplementedError(f'Unsupport activation type: {act_type}')

        self.activation = activation_hub[act_type](**kwargs)

    def forward(self, x):
        return self.activation(x)


# Regular convolution -> batchnorm -> activation
class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1,
                 bias=False, act_type='relu', **kwargs):
        if isinstance(kernel_size, list) or isinstance(kernel_size, tuple):
            padding = ((kernel_size[0] - 1) // 2 * dilation, (kernel_size[1] - 1) // 2 * dilation)
        elif isinstance(kernel_size, int):
            padding = (kernel_size - 1) // 2 * dilation

        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation, groups=groups,
                      bias=bias),
            nn.BatchNorm2d(out_channels),
            Activation(act_type, **kwargs)
        )


def register_model(*other_registries):
    def decorator(model_class):
        model_hub[model_class.__name__.lower()] = model_class

        for registry in other_registries:
            if isinstance(registry, list):
                registry.append(model_class.__name__.lower())
            else:
                print(f"Model registry is not a list. Skipping registry: {registry}")

        return model_class

    return decorator


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act_type, fuse_channels=None):
        super().__init__()
        fuse_channels = in_channels if fuse_channels is None else fuse_channels
        self.duck = DUCK(in_channels, fuse_channels, act_type)
        self.conv1 = ConvBNAct(fuse_channels, out_channels, 3, 2, act_type=act_type)
        self.conv2 = ConvBNAct(in_channels, out_channels, 2, 2, act_type=act_type)

    def forward(self, x1, x2=None):
        if x2 is None:
            x2 = self.conv2(x1)
        else:
            x2 = self.conv2(x2)

        skip = self.duck(x1)
        x1 = self.conv1(skip)

        return x1, skip, x2


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act_type):
        super().__init__()
        self.duck = DUCK(in_channels, out_channels, act_type)

    def forward(self, x, residual):
        size = residual.size()[2:]
        x = F.interpolate(x, size, mode='nearest')

        x += residual
        x = self.duck(x)

        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act_type):
        super().__init__()
        self.upper_branch = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.lower_branch = nn.Sequential(
            ConvBNAct(in_channels, out_channels, 3, act_type=act_type),
            ConvBNAct(out_channels, out_channels, 3, act_type=act_type),
        )
        self.bn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            Activation(act_type)
        )

    def forward(self, x):
        x_up = self.upper_branch(x)
        x_low = self.lower_branch(x)

        x = x_up + x_low
        x = self.bn(x)

        return x


class MidscopeBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, act_type):
        super().__init__(
            ConvBNAct(in_channels, out_channels, 3, act_type=act_type),
            ConvBNAct(out_channels, out_channels, 3, dilation=2, act_type=act_type)
        )


class WidescopeBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, act_type):
        super().__init__(
            ConvBNAct(in_channels, out_channels, 3, act_type=act_type),
            ConvBNAct(out_channels, out_channels, 3, dilation=2, act_type=act_type),
            ConvBNAct(out_channels, out_channels, 3, dilation=3, act_type=act_type),
        )


class SeparatedBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, filter_size, act_type):
        super().__init__(
            ConvBNAct(in_channels, out_channels, (1, filter_size), act_type=act_type),
            ConvBNAct(out_channels, out_channels, (filter_size, 1), act_type=act_type),
        )


class DUCK(nn.Module):
    def __init__(self, in_channels, out_channels, act_type, filter_size=6 + 1):
        """
        Here I change the filter size of separated block to be odd number.
        """
        super().__init__()
        self.in_bn = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            Activation(act_type)
        )
        self.branch1 = WidescopeBlock(in_channels, out_channels, act_type)
        self.branch2 = MidscopeBlock(in_channels, out_channels, act_type)
        self.branch3 = ResidualBlock(in_channels, out_channels, act_type)
        self.branch4 = nn.Sequential(
            ResidualBlock(in_channels, out_channels, act_type),
            ResidualBlock(out_channels, out_channels, act_type),
        )
        self.branch5 = nn.Sequential(
            ResidualBlock(in_channels, out_channels, act_type),
            ResidualBlock(out_channels, out_channels, act_type),
            ResidualBlock(out_channels, out_channels, act_type),
        )
        self.branch6 = SeparatedBlock(in_channels, out_channels, filter_size, act_type)
        self.out_bn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            Activation(act_type)
        )

    def forward(self, x):
        x = self.in_bn(x)

        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        x5 = self.branch5(x)
        x6 = self.branch6(x)

        x = x1 + x2 + x3 + x4 + x5 + x6
        x = self.out_bn(x)

        return x


@register_model()
class DuckNet(nn.Module):
    def __init__(self, num_class, n_channel=3, base_channel=17, act_type='relu'):
        super().__init__()
        self.down_stage1 = DownsampleBlock(n_channel, base_channel * 2, act_type, fuse_channels=base_channel)
        self.down_stage2 = DownsampleBlock(base_channel * 2, base_channel * 4, act_type)
        self.down_stage3 = DownsampleBlock(base_channel * 4, base_channel * 8, act_type)
        self.down_stage4 = DownsampleBlock(base_channel * 8, base_channel * 16, act_type)
        self.down_stage5 = DownsampleBlock(base_channel * 16, base_channel * 32, act_type)
        self.mid_stage = nn.Sequential(
            ResidualBlock(base_channel * 32, base_channel * 32, act_type),
            ResidualBlock(base_channel * 32, base_channel * 32, act_type),
            ResidualBlock(base_channel * 32, base_channel * 16, act_type),
            ResidualBlock(base_channel * 16, base_channel * 16, act_type),
        )

        self.up_stage5 = UpsampleBlock(base_channel * 16, base_channel * 8, act_type)
        self.up_stage4 = UpsampleBlock(base_channel * 8, base_channel * 4, act_type)
        self.up_stage3 = UpsampleBlock(base_channel * 4, base_channel * 2, act_type)
        self.up_stage2 = UpsampleBlock(base_channel * 2, base_channel, act_type)
        self.up_stage1 = UpsampleBlock(base_channel, base_channel, act_type)
        self.seg_head = nn.Conv2d(base_channel, num_class, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x1, x1_skip, x = self.down_stage1(x)
        x2, x2_skip, x = self.down_stage2(x1 + x, x)
        x3, x3_skip, x = self.down_stage3(x2 + x, x)
        x4, x4_skip, x = self.down_stage4(x3 + x, x)
        x5, x5_skip, x = self.down_stage5(x4 + x, x)
        x = self.mid_stage(x5 + x)

        x = self.up_stage5(x, x5_skip)
        x = self.up_stage4(x, x4_skip)
        x = self.up_stage3(x, x3_skip)
        x = self.up_stage2(x, x2_skip)
        x = self.up_stage1(x, x1_skip)
        x = self.seg_head(x)

        return x
