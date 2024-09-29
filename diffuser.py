from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
import torch
from .dit_model.vanilla_dit import Snowflake_DiT
from .configs import dit_config


class Diffusion:
    def __init__(self, image_size=dit_config.img_size):
        self.dit_model = Snowflake_DiT()
        self.dit_model.eval()

        self.vae = AutoencoderKL.from_pretrained(dit_config.vae_id)

        self.timesteps = torch.linspace(1, 999)
        self.latent_size = image_size // 8  # SD VAE uses 8x downsampling
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def sampler(self, imagenet_class: list = [21, 44, 33], prompt="a snowy landscape"):
        class_count = len(labels)
        # latent noise
        z_noise = torch.randn(
            class_count, 4, self.latent_size, self.latent_size)
        y_class = torch.tensor(imagenet_class, device=self.device)

        z_noise = torch.cat([z_noise, z_noise], 0)
