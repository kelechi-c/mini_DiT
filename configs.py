import torch


class dit_config:
    lr = 2e-4
    timesteps = 1000
    attn_heads = 6
    patch_size = 4
    transformer_blocks = 12
    hidden_dim = 384
    batch_size = 8
    dtype = torch.float16
    split = 10_000_000
    img_size = 256
    imagenet_id = "ILSVRC/imagenet-1k"
    vae_id = "stabilityai/sd-vae-ft-mse"
    clip_id = "apple/DFN5B-CLIP-ViT-H-14-378"
    vlm_prompt = "Describe this image correctly in a brief sentence"


class train_configs:
    outpath = "snow_dit"
    epochs = 50
    lr = 2e-4


class data_config:
    img_size = 256
    dtype = torch.float16
