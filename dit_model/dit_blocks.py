import torch
from torch import nn
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from ..configs import dit_config
from timm.models.vision_transformer import Attention, PatchEmbed, Mlp


sd_vae = AutoencoderKL.from_pretrained(dit_config.vae_id)


class DiTBlock(nn.Module):
    """
    input - layer norm - scale, shift - self_attention -
    concat(w/input) -
    layernorm - scale, shift - pointwise_feedforward - scale -
    concat(w/attention_tokens)
    """

    def __init__(self, attn_heads=6, hidden_size=384, mlp_ratio=4):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attention = Attention(hidden_size, num_heads=attn_heads)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_ratio * hidden_size,
            act_layer=nn.GELU(approximate="tanh"),
            drop=0.0,
        )
        self.adaLN = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def _modulate(self, x, shift_value, scale_value):
        x = x * (1 + scale_value.unsqueeze(1))
        x = x + shift_value.unsqueeze(1)

        return x

    def forward(self, x: torch.Tensor, c_label: torch.IntTensor) -> torch.Tensor:
        input = x  # keep input for residual concat
        # apply linear module on class condition, extract gating, shift and scale parameters for adaLN modulation
        shift_attn, scale_attn, gated_attn, shift_linear, scale_linear, gated_linear = (
            self.adaLN(c_label).chunk(6, dim=1)
        )

        # modulate with scale+shift params
        x = self._modulate(self.layer_norm(x), shift_attn, scale_attn)

        # first residual connection
        attn_x = input + gated_attn.unsqueeze(1) * self.attention(x)

        x = self._modulate(self.layer_norm(attn_x), shift_linear, scale_linear)
        # feedforward mlp, concat with attention tensors
        x = attn_x + gated_linear.unsqueeze(1) * self.mlp(x)

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
