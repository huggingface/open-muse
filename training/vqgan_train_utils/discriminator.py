import functools

from torch import nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# class NLayerDiscriminator(nn.Module):
#     """Defines a PatchGAN discriminator as in Pix2Pix
#     --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
#     """

#     def __init__(self, input_nc=3, ndf=64, n_layers=3):
#         """Construct a PatchGAN discriminator
#         Parameters:
#             input_nc (int)  -- the number of channels in input images
#             ndf (int)       -- the number of filters in the last conv layer
#             n_layers (int)  -- the number of conv layers in the discriminator
#             norm_layer      -- normalization layer
#         """
#         super(NLayerDiscriminator, self).__init__()
#         norm_layer = nn.BatchNorm2d
#         use_bias = False  # no need to use bias as BatchNorm2d has affine parameters

#         kw = 4
#         padw = 1
#         sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, False)]
#         nf_mult = 1
#         nf_mult_prev = 1
#         for n in range(1, n_layers):  # gradually increase the number of filters
#             nf_mult_prev = nf_mult
#             nf_mult = min(2**n, 8)
#             sequence += [
#                 nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
#                 norm_layer(ndf * nf_mult),
#                 nn.LeakyReLU(0.2, False),
#             ]

#         nf_mult_prev = nf_mult
#         nf_mult = min(2**n_layers, 8)
#         sequence += [
#             nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
#             norm_layer(ndf * nf_mult),
#             nn.LeakyReLU(0.2, False),
#         ]

#         sequence += [
#             nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
#         ]  # output 1 channel prediction map
#         self.main = nn.Sequential(*sequence)

#     def forward(self, input):
#         """Standard forward."""
#         return self.main(input)


"""
Ported from lucidrian's muse maksgit repository with some qol changes
"""

def leaky_relu(p=0.1):
    return nn.LeakyReLU(0.1)

def get_activation(name):
    if name == "leaky_relu":
        return leaky_relu
    elif name == "silu":
        return nn.SiLU
    else:
        raise NotImplementedError(f"Activation {name} is not implemented")

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=3):
        super().__init__()
        layer_mults = list(map(lambda t: 2**t, range(n_layers)))
        layer_dims = [ndf * mult for mult in layer_mults]
        dims = (ndf, *layer_dims)
        init_kernel_size=5
        kernel_size=3
        activation = get_activation("leaky_relu")
        dim_pairs = zip(dims[:-1], dims[1:])

        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        input_nc,
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
                    nn.GroupNorm(32, dim_out),
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