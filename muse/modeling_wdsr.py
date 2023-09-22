import math

import torch
from torch import nn

from .modeling_utils import ConfigMixin, ModelMixin, register_to_config

class WDSR(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        num_channels=3,
        num_blocks=16,
        num_residual_units=32,
        width_multiplier=4,
        image_mean=0.5,
        scale=1,
        temporal_size=None,
    ):
        super(WDSR, self).__init__()
        
        self.temporal_size = temporal_size
        self.image_mean = image_mean
        kernel_size = 3
        skip_kernel_size = 5
        weight_norm = torch.nn.utils.weight_norm
        num_inputs = num_channels
        if self.temporal_size:
            num_inputs *= self.temporal_size
        num_outputs = scale * scale * num_channels

        body = []
        conv = weight_norm(nn.Conv2d(num_inputs, num_residual_units, kernel_size, padding=kernel_size // 2))
        nn.init.ones_(conv.weight_g)
        nn.init.zeros_(conv.bias)
        body.append(conv)
        
        for _ in range(num_blocks):
            body.append(
                Block(
                    num_residual_units,
                    kernel_size,
                    width_multiplier,
                    weight_norm=weight_norm,
                    res_scale=1 / math.sqrt(num_blocks),
                )
            )
        
        conv = weight_norm(nn.Conv2d(num_residual_units, num_outputs, kernel_size, padding=kernel_size // 2))
        nn.init.ones_(conv.weight_g)
        nn.init.zeros_(conv.bias)
        body.append(conv)
        
        self.body = nn.Sequential(*body)

        skip = []
        if num_inputs != num_outputs:
            conv = weight_norm(nn.Conv2d(num_inputs, num_outputs, skip_kernel_size, padding=skip_kernel_size // 2))
            nn.init.ones_(conv.weight_g)
            nn.init.zeros_(conv.bias)
            skip.append(conv)
        self.skip = nn.Sequential(*skip)

        shuf = []
        if scale > 1:
            shuf.append(nn.PixelShuffle(scale))
        self.shuf = nn.Sequential(*shuf)

    def forward(self, x):
        if self.temporal_size:
            x = x.view([x.shape[0], -1, x.shape[3], x.shape[4]])
        x -= self.image_mean
        x = self.body(x) + self.skip(x)
        x = self.shuf(x)
        x += self.image_mean
        if self.temporal_size:
            x = x.view([x.shape[0], -1, 1, x.shape[2], x.shape[3]])
        return x

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_last_layer(self):
        return self.body[-1]

class Block(nn.Module):
    def __init__(
        self, num_residual_units, kernel_size, width_multiplier=1, weight_norm=torch.nn.utils.weight_norm, res_scale=1
    ):
        super(Block, self).__init__()
        body = []
        
        conv = weight_norm(
            nn.Conv2d(
                num_residual_units, int(num_residual_units * width_multiplier), kernel_size, padding=kernel_size // 2
            )
        )
        nn.init.constant_(conv.weight_g, 2.0)
        nn.init.zeros_(conv.bias)
        body.append(conv)
        body.append(nn.ReLU(True))
        
        conv = weight_norm(
            nn.Conv2d(
                int(num_residual_units * width_multiplier), num_residual_units, kernel_size, padding=kernel_size // 2
            )
        )
        nn.init.constant_(conv.weight_g, res_scale)
        nn.init.zeros_(conv.bias)
        body.append(conv)

        self.body = nn.Sequential(*body)

    def forward(self, x):
        x = self.body(x) + x
        return x
