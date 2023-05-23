# open-muse
An open-reproduction effortto reproduce the transformer based [MUSE](https://muse-model.github.io/) model for fast text2image generation.

## Goal
This repo is for reproduction of the [MUSE](https://arxiv.org/abs/2301.00704) model. The goal is to create a simple and scalable repo, to reproduce MUSE and build knowedge about VQ + transformers at scale.
We will use deduped LAION-2B + COYO-700M dataset for training.

Project stages:
1. Setup the codebase and train a class-conditional model on imagenet.
2. Conduct text2image experiments on CC12M.
3. Train improved VQGANs models.
4. Train the full (base-256) model on LAION + COYO.
5. Train the full (base-512) model on LAION + COYO.

All the artifacts of this project will be uploaded to the [openMUSE](https://huggingface.co/openMUSE) organization on the huggingface hub.

## Usage

### Installation

First create a virtual environment and install the repo using:

```bash
git clone https://github.com/huggingface/open-muse.git
cd open-muse
pip install -e ".[extra]"
```

You'll need to install `PyTorch` and `torchvision` manually. We are using `torch==1.13.1` with `CUDA11.7` for training.

For distributed data parallel training we use `accelerate` library, although this may change in the future. For dataset loading, we use `webdataset` library. So the dataset should be in the `webdataset` format.

### Models

At the momemnt we support following models:
- `MaskGitTransformer` - The main transformer model from the paper.
- `MaskGitVQGAN` - The VQGAN model from the [maskgit](https://github.com/google-research/maskgit) repo.
- `VQGANModel` - The VQGAN model from the [taming transformers](https://github.com/CompVis/taming-transformers) repo.

The models are implemented under `muse` directory. All models implement the familiar `transformers` API. So you can use `from_pretrained` and `save_pretrained` methods to load and save the models. The model can be saved and loaded from the huggingface hub.

#### VQGAN example:

```python
import torch
from torchvision import transforms
from PIL import Image
from muse import MaskGitVQGAN

# Load the pre-trained vq model from the hub
vq_model = MaskGitVQGAN.from_pretrained("openMUSE/maskgit-vqgan-imagenet-f16-256")

# encode and decode images using
encode_transform = = transforms.Compose(
    [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ]
)
image = Image.open("...") #
pixel_values = encode_transform(image).unsqueeze(0)
image_tokens = vq_model.encode(pixel_values)
rec_image = vq_model.decode(image_tokens)

# Convert to PIL images
rec_image = 2.0 * rec_image - 1.0
rec_image = torch.clamp(rec_image, -1.0, 1.0)
rec_image = (rec_image + 1.0) / 2.0
rec_image *= 255.0
rec_image = rec_image.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
pil_images = [Image.fromarray(image) for image in rec_image]
```

#### MaskGitTransformer example for class-conditional generation:

```python
import torch
from muse import MaskGitTransformer, MaskGitVQGAN
from muse.sampling import cosine_schedule

# Load the pre-trained vq model from the hub
vq_model = MaskGitVQGAN.from_pretrained("openMUSE/maskgit-vqgan-imagenet-f16-256")

# Initialize the MaskGitTransformer model
maskgit_model = MaskGitTransformer(
    vocab_size=2025, #(1024 + 1000 + 1 = 2025 -> Vq_tokens + Imagenet class ids + <mask>)
    max_position_embeddings=257, # 256 + 1 for class token
    hidden_size=512,
    num_hidden_layers=8,
    num_attention_heads=8,
    intermediate_size=2048,
    codebook_size=1024,
    num_vq_tokens=256,
    num_classes=1000,
)

# prepare the input batch
images = torch.randn(4, 3, 256, 256)
class_ids = torch.randint(0, 1000, (4,)) # random class ids
# encode the images
image_tokens = vq_model.encode(images)
batch_size, seq_len = image_tokens.shape
# Sample a random timestep for each image
timesteps = torch.rand(batch_size, device=image_tokens.device)
# Sample a random mask probability for each image using timestep and cosine schedule
mask_prob = cosine_schedule(timesteps)
mask_prob = mask_prob.clip(min_masking_rate)
# creat a random mask for each image
num_token_masked = (seq_len * mask_prob).round().clamp(min=1)
batch_randperm = torch.rand(batch_size, seq_len, device=image_tokens.device).argsort(dim=-1)
mask = batch_randperm < num_token_masked.unsqueeze(-1)
# mask images and create input and labels
input_ids = torch.where(mask, mask_id, image_tokens)
labels = torch.where(mask, image_tokens, -100)

# shift the class ids by codebook size
class_ids = class_ids + vq_model.num_embeddings
# prepend the class ids to the image tokens
input_ids = torch.cat([class_ids.unsqueeze(-1), input_ids], dim=-1)
# prepend -100 to the labels as we don't want to predict the class ids
labels = torch.cat([-100 * torch.ones_like(class_ids).unsqueeze(-1), labels], dim=-1)

# forward pass
logits, loss = maskgit_model(input_ids, labels=labels)

loss.backward()

# to generate images
class_ids = torch.randint(0, 1000, (4,)) # random class ids
generated_tokens = maskgit_model.generate(class_ids=class_ids)
rec_images = vq_model.decode(generated_tokens)
```

___Note___:
- The vq model and transformer model are kept separate to be able to scale the transformer model independently. And we may pre-encode the images for faster training.
- The masking is also done outside the model to be able to use different masking strategies without affecting the modeling code.

## Basic explanation of MaskGit Generation Process

1. Maskgits is a transformer that outputs logits given a sequence of tokens of both vq and class-conditioned label token

2. The way the denoising process is done is to mask out with mask token ids and gradually denoise

3. In the original implementation, this is done by first using a softmax on the last dim and randomly sampling as a categorical distribution. This will give our predicted tokens for each maskid. Then we get the probabilities for those tokens to be chosen. Finally, we get the topk highest confidence probabilities when gumbel*temp is added to it. Gumbel distribution is like a shifted normal distribution towards 0 which is used to model extreme events. So in extreme scenarios, we will like to see a different token being chosen from the default one

4. For the lucidrian implementation, it first removes the highest-scoring (lowest probability) tokens by masking them with a given masking ratio. Then, except for the highest 10% of the logits that we get, we set it to -infinity so when we do the gumbel distribution on it, they will be ignored. Then update the input ids and the scores where the scores are just 1-the probability given by the softmax of the logits at the predicted ids interestingly

## Training
For class-conditional imagenet we are using `accelerate` for DDP training and `webdataset` for data loading. The training script is available in `training/train_maskgit_imagenet.py`.

We use OmegaConf for configuration management. See `configs/template_config.yaml` for the configuration template. Below we explain the configuration parameters.

```yaml
wandb:
  entity: ???

experiment:
    name: ???
    project: ???
    output_dir: ???
    max_train_examples: ???
    save_every: 1000
    eval_every: 500
    generate_every: 1000
    log_every: 50
    log_grad_norm_every: 100
    resume_from_checkpoint: latest

model:
    vq_model:
        pretrained: "openMUSE/maskgit-vqgan-imagenet-f16-256"

    transformer:
        vocab_size: 2048 # (1024 + 1000 + 1 = 2025 -> Vq + Imagenet + <mask>, use 2048 for even division by 8)
        max_position_embeddings: 264 # (256 + 1 for class id, use 264 for even division by 8)
        hidden_size: 768
        num_hidden_layers: 12
        num_attention_heads: 12
        intermediate_size: 3072
        codebook_size: 1024
        num_vq_tokens: 256
        num_classes: 1000
        initializer_range: 0.02
        layer_norm_eps: 1e-6
        use_bias: False
        use_normformer: True
        use_encoder_layernorm: True
        hidden_dropout: 0.0
        attention_dropout: 0.0

    gradient_checkpointing: True
    enable_xformers_memory_efficient_attention: False


dataset:
    params:
        train_shards_path_or_url: ???
        eval_shards_path_or_url: ???
        batch_size: ${training.batch_size}
        shuffle_buffer_size: ???
        num_workers: ???
        resolution: 256
        pin_memory: True
        persistent_workers: True
    preprocessing:
        resolution: 256
        center_crop: True
        random_flip: False

optimizer:
    name: adamw # Can be adamw or lion or fused_adamw. Install apex for fused_adamw
    params: # default adamw params
        learning_rate: ???
        scale_lr: False # scale learning rate by total batch size
        beta1: 0.9
        beta2: 0.999
        weight_decay: 0.01
        epsilon: 1e-8

lr_scheduler:
    scheduler: "constant_with_warmup"
    params:
        learning_rate: ${optimizer.params.learning_rate}
        warmup_steps: 500

training:
    gradient_accumulation_steps: 1
    batch_size: 128
    mixed_precision: "no"
    enable_tf32: True
    use_ema: False
    seed: 42
    max_train_steps: ???
    overfit_one_batch: False
    min_masking_rate: 0.0
    label_smoothing: 0.0
    max_grad_norm: null
```

Arguments with ??? are required.

__wandb__:
- `wandb.entity`: The wandb entity to use for logging.

__experiment__:
- `experiment.name`: The name of the experiment.
- `experiment.project`: The wandb project to use for logging.
- `experiment.output_dir`: The directory to save the checkpoints.
- `experiment.max_train_examples`: The maximum number of training examples to use.
- `experiment.save_every`: Save a checkpoint every `save_every` steps.
- `experiment.eval_every`: Evaluate the model every `eval_every` steps.
- `experiment.generate_every`: Generate images every `generate_every` steps.
- `experiment.log_every`: Log the training metrics every `log_every` steps.
- `log_grad_norm_every`: Log the gradient norm every `log_grad_norm_every` steps.
- `experiment.resume_from_checkpoint`: The checkpoint to resume training from. Can be `latest` to resume from the latest checkpoint or path to a saved checkpoint. If `None` or the path does not exist then training starts from scratch.

__model__:
- `model.vq_model.pretrained`: The pretrained vq model to use. Can be a path to a saved checkpoint or a huggingface model name.
- `model.transformer`: The transformer model configuration.
- `model.gradient_checkpointing`: Enable gradient checkpointing for the transformer model.
- `enable_xformers_memory_efficient_attention`: Enable memory efficient attention or flash attention for the transformer model. For flash attention we need to use `fp16` or `bf16`. [xformers](https://github.com/facebookresearch/xformers) needs to be installed for this to work.

__dataset__:
- `dataset.params.train_shards_path_or_url`: The path or url to the `webdataset` training shards.
- `dataset.params.eval_shards_path_or_url`: The path or url to the `webdataset` evaluation shards.
- `dataset.params.batch_size`: The batch size to use for training.
- `dataset.params.shuffle_buffer_size`: The shuffle buffer size to use for training.
- `dataset.params.num_workers`: The number of workers to use for data loading.
- `dataset.params.resolution`: The resolution of the images to use for training.
- `dataset.params.pin_memory`: Pin the memory for data loading.
- `dataset.params.persistent_workers`: Use persistent workers for data loading.
- `dataset.preprocessing.resolution`: The resolution of the images to use for preprocessing.
- `dataset.preprocessing.center_crop`: Whether to center crop the images. If `False` then the images are randomly cropped to the `resolution`.
- `dataset.preprocessing.random_flip`: Whether to randomly flip the images. If `False` then the images are not flipped.

__optimizer__:
- `optimizer.name`: The optimizer to use for training.
- `optimizer.params`: The optimizer parameters.

__lr_scheduler__:
- `lr_scheduler.scheduler`: The learning rate scheduler to use for training.
- `lr_scheduler.params`: The learning rate scheduler parameters.

__training__:
- `training.gradient_accumulation_steps`: The number of gradient accumulation steps to use for training.
- `training.batch_size`: The batch size to use for training.
- `training.mixed_precision`: The mixed precision mode to use for training. Can be `no`, `fp16` or `bf16`.
- `training.enable_tf32`: Enable TF32 for training on Ampere GPUs.
- `training.use_ema`: Enable EMA for training. Currently not supported.
- `training.seed`: The seed to use for training.
- `training.max_train_steps`: The maximum number of training steps.
- `training.overfit_one_batch`: Whether to overfit one batch for debugging.
- `training.min_masking_rate`: The minimum masking rate to use for training.
- `training.label_smoothing`: The label smoothing value to use for training.
- `max_grad_norm`: Max gradient norm.

___Notes about training and dataset.___:

We randomly resample the shards (with replacement) and sample examples in buffer for training every time we resume/start the training run. This means our data loading is not determinitsic. We also don't do epoch based training but just using this for book keeping and being able to reuse the same training loop with other datasets/loaders.

### Running experiments:
So far we are running experiments on single node. To launch a training run on a single node, run the following steps:

1. Prepare the dataset in `webdataset` format. You can use the `scripts/convert_imagenet_to_wds.py` script to convert the imagenet dataset to `webdataset` format.
2. First configure your training env using `accelerate config`.
3. Create a `config.yaml` file for your experiment.
4. Launch the training run using `accelerate launch`.

```bash
accelerate launch python -u training/train_maskgit_imagenet.py config=path/to/yaml/config
```

With OmegaConf, commandline overrides are done in dot-notation format. E.g. if you want to override the dataset path, you would use the command `python -u train.py config=path/to/config dataset.params.path=path/to/dataset`.

The same command can be used to launch the training locally.

## Steps
### Setup the codebase and train a class-conditional model no imagenet.
- [x] Setup repo-structure
- [x] Add transformers and VQGAN model.
- [x] Add a generation support for the model.
- [x] Port the VQGAN from [maskgit](https://github.com/google-research/maskgit) repo for imagenet experiment.
- [x] Finish and verify masking utils.
- [ ] Add the masking arccos scheduling function from MUSE.
- [x] Add EMA.
- [x] Suport OmegaConf for training configuration.
- [x] Add W&B logging utils.
- [x] Add WebDataset support. Not really needed for imagenet experiment but can work on this parallelly. (LAION is already available in this format so will be easier to use it).
- [x] Add a training script for class conditional generation using imagenet.
- [x] Make the codebase ready for the cluster training. Add SLURM scripts.

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
├── configs                        -> All training config files.
│   └── template_config.yaml
├── muse
│   ├── __init__.py
│   ├── data.py                    -> All data related utils. Can create a data folder if needed.
│   ├── logging.py                 -> Misc logging utils.
|   ├── lr_schedulers.py           -> All lr scheduler related utils.
│   ├── modeling_maskgit_vqgan.py  -> VQGAN model from maskgit repo.
│   ├── modeling_taming_vqgan.py   -> VQGAN model from taming repo.
│   └── modeling_transformer.py    -> The main transformer model.
│   ├── modeling_utils.py          -> All model related utils, like save_pretrained, from_pretrained from hub etc
│   ├── sampling.py                -> Sampling/Generation utils.
│   ├── training_utils.py          -> Common training utils.
├── pyproject.toml
├── setup.cfg
├── setup.py
├── test.py
└── training                       -> All training scripts.
    ├── __init__.py
    ├── data.py                    -> All data related utils. Can create a data folder if needed.
    ├── optimizer.py               -> All optimizer related utils and any new optimizer not available in PT.
    ├── train_maskgit_imagenet.py
    ├── train_muse.py
    └── train_vqgan.py
```

## Acknowledgments

This project is hevily based on the following open-source repos. Thanks to all the authors for their amazing work.
- [muse-maskgit-pytorch](https://github.com/lucidrains/muse-maskgit-pytorch) .  A big thanks to @lucidrains for this amazing work ❤️
- [maskgit](https://github.com/google-research/maskgit) 
- [taming-transformers](https://github.com/CompVis/taming-transformers)
- [open-clip](https://github.com/mlfoundations/open_clip)
- [open-diffusion](https://github.com/mlfoundations/open-diffusion)
- [dalle-mini](https://github.com/borisdayma/dalle-mini): ❤️
- [transformers](https://github.com/huggingface/transformers)
- [accelerate](https://github.com/huggingface/accelerate)
- [diffusers](https://github.com/huggingface/diffusers)
- [webdatset](https://github.com/webdataset/webdataset)

And obivioulsy to PyTorch team for this amazing framework ❤️
