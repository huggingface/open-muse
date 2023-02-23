# muse-open-reproduction
A repo to train the best and fastest text2image model!

## Goal
This repo is for reproduction of the [MUSE](https://arxiv.org/abs/2301.00704) model. The goal is to create a simple and scalable repo, to reproduce MUSE and build knowedge about VQ + transformers at scale.
We will use deduped LAION-2B + COYO-700M dataset for training.

Project stages:
1. Setup the codebase and train a class-conditional model on imagenet.
2. Conduct text2image experiments on CC12M.
3. Train improved VQGANs models.
4. Train the full (base-256) model on LAION + COYO.
5. Train the full (base-512) model on LAION + COYO.


## Steps
### Setup the codebase and train a class-conditional model no imagenet.
- [x] Setup repo-structure
- [x] Add transformers and VQGAN model.
- [x] Add a generation support for the model.
- [x] Port the VQGAN from [maskgit](https://github.com/google-research/maskgit) repo for imagenet experiment.
- [ ] Finish and verify masking utils.
- [ ] Add the masking arccos scheduling function from MUSE.
- [x] Add EMA.
- [ ] Suport OmegaConf for training configuration.
- [ ] Add W&B logging utils.
- [ ] Add WebDataset support. Not really needed for imagenet experiment but can work on this parallelly. (LAION is already available in this format so will be easier to use it).
- [ ] Add a training script for class conditional generation using imagenet. (WIP)
- [ ] Make the codebase ready for the cluster training.

### Conduct text2image experiments on CC12M.
- [ ] Finish data loading, pre-processing utils.
- [ ] Add CLIP and T5 support.
- [ ] Add text2image training script.
- [ ] Add eavluation scripts (FiD, CLIP score).
- [ ] Train on CC12M. Here we could conduct different experiments:
    - [ ] Train on CC12M with T5 conditioning.
    - [ ] Train on CC12M with CLIP conditioning.
    - [ ] Train on CC12M with CLIP + T5 conditioning (probably costly during training and experiments).
    - [ ] Self conditioning from Bit Diffusion paper.
- [ ] Collect different prompts for intermmediate evaluations (Can reuse the prompts for dalle-mini, parti-prompts).
- [ ] Setup a space where people can play with the model and provide feedback, compare with other models etc.

### Train improved VQGANs models.
- [ ] Add training component models for VQGAN (EMA, discriminator, LPIPS etc).
- [ ] VGQAN training script.


### Misc tasks
- [ ] Create a space for visualizing exploring dataset
- [ ] Create a space where people can try to find their own images and can opt-out of the dataset.


## Repo structure (WIP)
```
├── README.md
├── configs                   -> All training config files.
│   └── dummy_config.yaml
├── muse
│   ├── __init__.py
│   ├── data.py              -> All data related utils. Can create a data folder if needed.
│   ├── logging.py           -> Misc logging utils.
│   ├── maskgit_vqgan.py     -> VQGAN model from maskgit repo.
│   ├── modeling_utils.py    -> All model related utils, like save_pretrained, from_pretrained from hub etc
│   ├── sampling.py          -> Sampling/Generation utils.
│   ├── taming_vqgan.py      -> VQGAN model from taming repo.
│   ├── training_utils.py    -> Common training utils.
│   └── transformer.py       -> The main transformer model.
├── pyproject.toml
├── setup.cfg
├── setup.py
├── test.py
└── training                 -> All training scripts.
    ├── train_muse.py
    └── train_vqgan.py