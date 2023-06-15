"""All data related utilities and loaders are defined here."""
import os
import torch
import zipfile
import torchvision
import numpy as np
import pandas as pd
from PIL import Image
from datasets import load_dataset
from pycocotools.coco import COCO

def unzip_file(zip_src, tgt_dir):
    if zipfile.is_zipfile(zip_src):
        fz = zipfile.ZipFile(zip_src, 'r')
        for file in fz.namelist():
            fz.extract(file, tgt_dir)       
    else:
        raise RuntimeError("This is not zip file.")

        
class Laion:
    def __init__(self, metadata_path, folder_path, fid='folder', key='key', caption_col='caption', transform=None):
        self.df = pd.read_parquet(metadata_path)
        self.fpath = folder_path
        self.fid = fid
        self.key = key
        self.caption_col = caption_col
        self.transform = transform
        
    def __getitem__(self, idx):
        fid = self.df[self.fid][idx]
        key = self.df[self.key][idx]
        img_path = f"{self.fpath}/{fid}/{key}.jpg"
        img = Image.open(img_path).convert('RGB')
        caption = self.df[self.caption_col][idx]
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, caption
    
    def __len__(self):
        return len(self.df)        

                 
class LaionV2:
    def __init__(self, metadata_path, folder_path, fid='folder', key='key', caption_col=['caption', 'prompt'], p=[0.2, 0.8], transform=None):
        self.df = pd.read_parquet(metadata_path)
        self.fpath = folder_path
        self.fid = fid
        self.key = key
        self.caption_col = caption_col
        self.p = p
        self.transform = transform
        
    def __getitem__(self, idx):
        fid = self.df[self.fid][idx]
        key = self.df[self.key][idx]
        img_path = f"{self.fpath}/{fid}/{key}.jpg"
        img = Image.open(img_path).convert('RGB')
        
        prompts = []
        for col in self.caption_col:
            prompts.append(self.df[col][idx])
        caption = np.random.choice(prompts, p=self.p)
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, caption
    
    def __len__(self):
        return len(self.df)
  
    
class ImageNet:
    def __init__(self, root, split='train', transform=None):
        self.dataset = torchvision.datasets.ImageNet(root=root, split=split)
        self.transform = transform
        self.prefix = ["an image of ", "a picture of "]
        
    def __getitem__(self, idx):
        image, target = self.dataset[idx]
        caption = np.random.choice(self.prefix) + np.random.choice(self.dataset.classes[target])
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image, caption
    
    def __len__(self):
        return len(self.dataset)
           
        
class Flickr30k:
    def __init__(self, img_dir, ann_file, transform=None):
        self.dataset = torchvision.datasets.Flickr30k(root=img_dir, ann_file=ann_file)
        self.transform = transform
        
    def __getitem__(self, idx):
        img, captions = self.dataset[idx]
        caption = np.random.choice(captions)
        
        if self.transform is not None:
            img = self.transform(img)
            
        return img, caption
    
    def __len__(self):
        return len(self.dataset)
        

class DiffusionDB:
    def __init__(self, version='large_random_100k', transform=None):
        self.dataset = load_dataset("poloclub/diffusiondb", version)['train']
        self.transform = transform
        
    def __getitem__(self, idx):
        data = self.dataset[idx]
        
        image = data['image']
        prompt = data['prompt']
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image, prompt
        
    def __len__(self):
        return len(self.dataset)


class CoCo:
    def __init__(self, root, dataType='train2017', annType='captions', transform=None, image_set='train'):
        self.root = root
        self.img_dir = '{}/{}'.format(root, dataType)
        annFile = '{}/annotations/{}_{}.json'.format(root, annType, dataType)
       
        self.coco = COCO(annFile)
        if image_set == "train":
            self.imgids = self.coco.getImgIds()
        elif image_set == "val":
            self.imgids = self.coco.getImgIds()[:100]
        self.transform = transform
    
    def __getitem__(self, idx):
        imgid = self.imgids[idx]
        img_name = self.coco.loadImgs(imgid)[0]['file_name']
        annid = self.coco.getAnnIds(imgIds=imgid)
        img = Image.open(os.path.join(self.img_dir, img_name)).convert('RGB')
        ann = np.random.choice(self.coco.loadAnns(annid))['caption']
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, ann     
        
    def __len__(self):
        return len(self.imgids)


class CelebA:
    def __init__(self, root, type='identity', transform=None):
        """CelebA Dataset http://personal.ie.cuhk.edu.hk/~lz013/projects/CelebA.html
        Args:
            root (str): CelebA Dataset folder path
            type (str, optional): 'identity' or 'attr'. Defaults to 'identity'.
            transform (torchvision.transforms, optional): torchvision.transforms. Defaults to None.
        """        
        ann_dir = os.path.join(root, 'Anno')
        base_dir = os.path.join(root, 'Img')
        zfile_path = os.path.join(base_dir, 'img_align_celeba.zip')
        self.img_dir = os.path.join(base_dir, 'img_align_celeba')
        if os.path.exists(self.img_dir):
            pass
        elif os.path.exists(zfile_path):
            unzip_file(zfile_path, base_dir)
        else:
            raise RuntimeError("Dataset not found.")
        self.imgs = os.listdir(self.img_dir)
        if type == 'identity':
            self.img2id = {}
            with open(os.path.join(ann_dir, 'identity_CelebA.txt'), 'r') as f:
                for line in f.readlines():
                    name, id = line.strip().split(' ')
                    self.img2id[name] = int(id)
        self.transform = transform
        
    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img = Image.open(os.path.join(self.img_dir, img_name)).convert('RGB')
        ann = self.img2id[img_name]
        if self.transform is not None:
            img = self.transform(img)
        
        ann = torch.tensor(ann)
            
        return img, ann
    
    def __len__(self):
        return len(self.imgs)
