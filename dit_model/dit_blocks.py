import torch
from torch import embedding, nn, numel
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from ..configs import dit_config
from transformers import LlavaPreTrainedModel
from timm.models.vision_transformer import Attention, PatchEmbed, Mlp


sd_vae = AutoencoderKL.from_pretrained(dit_config.vae_id)


class DiTBlock(nn.Module):
    def __init__(self, attn_heads=6, hidden_size=384, config=dit_config):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attention = Attention(hidden_size, num_heads=attn_heads)
        self.adaLN = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


# final mlp layer for diffusion transformer
class FInalMlp(nn.Module):
    def __init__(self, embed_dim, patch_size, out_channels):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.linear = nn.Linear(embed_dim, patch_size)
        self.ada_ln = nn.Sequential(
            nn.SiLU(), nn.Linear(embed_dim, 2 * embed_dim, bias=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


# main diffusion transformer block
class Snowflake_DiT(nn.Module):
    def __init__(self, embed_dim, num_layers, config=dit_config):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.patch_embed = PatchEmbed()
        # dit blocks
        dit_blocks = [DiTBlock() for _ in range(num_layers)]
        self.dit_layers = nn.Sequential(*dit_blocks)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x
