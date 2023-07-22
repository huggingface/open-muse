"""
Ported from lucidrian's muse maksgit repository with some qol changes
"""
from torch import nn
import torch

def leaky_relu(p=0.1):
    return nn.LeakyReLU(0.1)

def get_activation(name):
    if name == "leaky_relu":
        return leaky_relu
    elif name == "silu":
        return nn.SiLU
    else:
        raise NotImplementedError(f"Activation {name} is not implemented")

class Discriminator(nn.Module):
    def __init__(
        self,
        config
    ):
        super().__init__()
        dim = config.discriminator.dim
        discr_layers = config.discriminator.discr_layers
        layer_mults = list(map(lambda t: 2**t, range(discr_layers)))
        layer_dims = [dim * mult for mult in layer_mults]
        dims = (dim, *layer_dims)
        channels=config.discriminator.channels
        groups=config.discriminator.groups
        init_kernel_size=config.discriminator.init_kernel_size
        kernel_size=config.discriminator.kernel_size
        act=config.discriminator.act
        activation = get_activation(act)
        dim_pairs = zip(dims[:-1], dims[1:])

        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        dims[0],
                        init_kernel_size,
                        padding=init_kernel_size // 2,
                    ),
                    activation(),
                )
            ]
        )

        for dim_in, dim_out in dim_pairs:
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        dim_in,
                        dim_out,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=kernel_size // 2,
                    ),
                    nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
                    nn.GroupNorm(groups, dim_out),
                    activation(),
                )
            )

        dim = dims[-1]
        self.to_logits = nn.Sequential(  # return 5 x 5, for PatchGAN-esque training
            nn.Conv2d(dim, dim, 1), activation(), nn.Conv2d(dim, 1, 4)
        )

    def forward(self, x):
        for net in self.layers:
            x = net(x)

        return self.to_logits(x)

class PaellaDiscriminator(nn.Module):
    def __init__(self, config):
        channels=config.discriminator.channels
        cond_channels=config.discriminator.cond_channels
        hidden_channels = config.discriminator.hidden_channels
        depth = config.discriminator.depth
        super().__init__()
        d = max(depth - 3, 3)
        layers = [
            nn.utils.spectral_norm(nn.Conv2d(channels, hidden_channels // (2 ** d), kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
        ]
        for i in range(depth - 1):
            c_in = hidden_channels // (2 ** max((d - i), 0))
            c_out = hidden_channels // (2 ** max((d - 1 - i), 0))
            layers.append(nn.utils.spectral_norm(nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1)))
            layers.append(nn.InstanceNorm2d(c_out))
            layers.append(nn.LeakyReLU(0.2))
        self.encoder = nn.Sequential(*layers)
        self.shuffle = nn.Conv2d((hidden_channels + cond_channels) if cond_channels > 0 else hidden_channels, 1, kernel_size=1)
        self.logits = nn.Sigmoid()

    def forward(self, x, cond=None):
        x = self.encoder(x)
        if cond is not None:
            cond = cond.view(cond.size(0), cond.size(1), 1, 1, ).expand(-1, -1, x.size(-2), x.size(-1))
            x = torch.cat([x, cond], dim=1)
        x = self.shuffle(x)
        x = self.logits(x)
        return x