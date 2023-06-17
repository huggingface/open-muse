"""
Ported from lucidrian's muse maksgit repository with some qol changes
"""
from torch import nn

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