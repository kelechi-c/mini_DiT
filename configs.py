import torch


class dit_config:
    timesteps = 1000
    attn_heads = 6
    patch_size = 4
    transformer_blocks = 12
    hidden_dim = 384
    dtype = torch.float16
    split = 1_000_000
    img_size = 256
    imagenet_id = "ILSVRC/imagenet-1k"
    vae_id = "stabilityai/sd-vae-ft-mse"
    clip_id = "apple/DFN5B-CLIP-ViT-H-14-378"


class train_configs:
    outpath = "snow_dit"
    epochs = 50
    lr = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class data_config:
    img_size = 256
    dtype = torch.float16
    batch_size = 16
