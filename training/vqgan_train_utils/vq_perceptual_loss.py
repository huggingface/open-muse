import torch
import torch.nn as nn
import torch.nn.functional as F

from .lpips import LPIPS


def adopt_weight(weight, global_step, threshold=0, value=0.0):
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) + torch.mean(torch.nn.functional.softplus(logits_fake))
    )
    return d_loss


class VQLPIPSWithDiscriminator(nn.Module):
    def __init__(
        self,
        disc_start,
        codebook_weight=1.0,
        disc_factor=1.0,
        disc_weight=1.0,
        perceptual_weight=1.0,
        disc_loss="hinge",
    ):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.codebook_weight = codebook_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight

        self.discriminator_iter_start = disc_start
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        print(f"VQLPIPSWithDiscriminator running with {disc_loss} loss.")
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(
        self,
        discriminator,
        codebook_loss,
        inputs,
        reconstructions,
        optimizer_idx,
        global_step,
        last_layer=None,
        split="train",
    ):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss
        else:
            p_loss = torch.tensor([0.0])

        nll_loss = rec_loss
        nll_loss = torch.mean(nll_loss)

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            logits_fake = discriminator(reconstructions.contiguous())
            g_loss = -torch.mean(logits_fake)

            try:
                d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
            except RuntimeError:
                assert not self.training
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = nll_loss + d_weight * disc_factor * g_loss

            if self.codebook_weight > 0 and codebook_loss is not None:
                loss += self.codebook_weight * codebook_loss.mean()

            log = {
                "{}/total_loss".format(split): loss.clone().detach(),
                "{}/nll_loss".format(split): nll_loss.detach(),
                "{}/rec_loss".format(split): rec_loss.detach(),
                "{}/p_loss".format(split): p_loss.detach(),
                "{}/d_weight".format(split): d_weight.detach(),
                "{}/disc_factor".format(split): torch.tensor(disc_factor),
                "{}/g_loss".format(split): g_loss.detach(),
            }

            if self.codebook_weight > 0 and codebook_loss is not None:
                log["{}/codebook_loss".format(split)] = codebook_loss.detach()

            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            logits_real = discriminator(inputs.contiguous().detach())
            logits_fake = discriminator(reconstructions.contiguous().detach())

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {
                "{}/disc_loss".format(split): d_loss.clone().detach(),
                "{}/logits_real".format(split): logits_real.detach(),
                "{}/logits_fake".format(split): logits_fake.detach(),
            }

            return d_loss, log
