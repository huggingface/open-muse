# Adapted from https://github.com/lucidrains/muse-maskgit-pytorch

import math
from functools import partial

import torch


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def gumbel_noise(t, generator=None):
    noise = torch.zeros_like(t).uniform_(0, 1, generator=generator)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1.0, dim=-1, generator=None):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t, generator=generator)).argmax(dim=dim)


def top_k(logits, thres=0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = logits.topk(k, dim=-1)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(2, ind, val)
    return probs


def mask_by_random_topk(mask_len, probs, temperature=1.0, generator=None):
    confidence = log(probs) + temperature * gumbel_noise(probs, generator=generator)
    sorted_confidence = torch.sort(confidence, dim=-1).values
    cut_off = torch.gather(sorted_confidence, 1, mask_len.long())
    masking = confidence < cut_off
    return masking


def cosine_schedule(t):
    return torch.cos(t * math.pi * 0.5)


def linear_schedule(t):
    mask_ratio = 1 - t
    mask_ratio = mask_ratio.clamp(min=1e-6, max=1.0)
    return mask_ratio


def pow(t, method):
    exponent = float(method.replace("pow", ""))
    mask_ratio = 1.0 - t**exponent
    mask_ratio = mask_ratio.clamp(min=1e-6, max=1.0)
    return mask_ratio


def sigmoid_schedule(t, start=-3, end=3, tau=1.0, clip_min=1e-6):
    for item in [t, start, end, tau]:
        item = torch.tensor(item) if not torch.is_tensor(item) else item

    # A gamma function based on sigmoid function.
    v_start = torch.sigmoid(torch.tensor(start / tau))
    v_end = torch.sigmoid(torch.tensor(end / tau))
    output = torch.sigmoid((t * (end - start) + start) / tau)
    output = (v_end - output) / (v_end - v_start)
    return torch.clip(output, clip_min, 1.0)


def get_mask_chedule(method, **schedule_kwargs):
    if method == "cosine":
        return cosine_schedule
    elif method == "linear":
        return linear_schedule
    elif "pow" in method:
        return partial(pow, method=method)
    elif method == "sigmoid":
        return partial(sigmoid_schedule, **schedule_kwargs)
    else:
        raise ValueError("Unknown schedule method: {}".format(method))
