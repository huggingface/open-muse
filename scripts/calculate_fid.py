import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from argparse import ArgumentParser
from torch import nn
from torch.utils.data import Dataset
from PIL import Image
import os
import torch
import numpy as np
from tqdm import tqdm
from scipy import linalg

class ImagePathDataset(Dataset):
    def __init__(self, data_root, transforms=None):
        self.files = os.listdir(data_root)
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img

# Taken from https://github.com/GaParmar/clean-fid/blob/main/cleanfid/fid.py
def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

def fid_from_feats(feats1, feats2):
    mu1, sig1 = np.mean(feats1, axis=0), np.cov(feats1, rowvar=False)
    mu2, sig2 = np.mean(feats2, axis=0), np.cov(feats2, rowvar=False)
    return frechet_distance(mu1, sig1, mu2, sig2)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--timm_model", type=str, choices=["inception_v3"], default="inception_v3")
    parser.add_argument("--generated_folder", type=str, default="generated")
    parser.add_argument("--real_folder", type=str, default="real")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    model = timm.create_model(args.timm_model, pretrained=True).to(args.device)
    model.eval()
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)

    model.fc = nn.Identity()
    generated_dataset = ImagePathDataset(args.generated_folder, transform)
    real_dataset = ImagePathDataset(args.real_folder, transform)
    generated_dataloader = torch.utils.data.DataLoader(generated_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=args.num_workers)
    real_dataloader = torch.utils.data.DataLoader(real_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=args.num_workers)

    generated_acts, real_acts = [], []
    print("Making activations for generated dataset")
    for batch in tqdm(generated_dataloader):
        batch = batch.to(args.device)
        generated_acts.append(model(batch).cpu().detach().numpy())

    print("Making activations for real dataset")
    for batch in tqdm(real_dataloader):
        batch = batch.to(args.device)
        real_acts.append(model(batch).cpu().detach().numpy())
    generated_acts, real_acts = np.concatenate(generated_acts, axis=0), np.concatenate(real_acts, axis=0)
    fid = fid_from_feats(generated_acts, real_acts)
    print(f"FID: {fid}")