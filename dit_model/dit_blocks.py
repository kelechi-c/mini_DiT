import torch
import math
from torch import nn
from timm.models.vision_transformer import Attention, Mlp
from ..utils import modulate


class DiTBlock(nn.Module):
    """
    Diffusion transformer block from the paper.
    layout =>
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


# final linear layer for diffusion transformer
class FinalMlp(nn.Module):
    def __init__(self, embed_dim, patch_size, out_channels):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)  # layer nrmalization
        self.linear = nn.Linear(  # mlp layer
            embed_dim, patch_size * patch_size * out_channels, bias=True
        )
        self.ada_ln = nn.Sequential(  # adaptive layernorm mlp
            nn.SiLU(), nn.Linear(embed_dim, 2 * embed_dim, bias=True)
        )

    def forward(self, x: torch.Tensor, c_class: torch.Tensor) -> torch.Tensor:
        shift, scale = self.ada_ln(c_class).chunk(
            2, dim=1
        )  # shift and scale parameters

        x = self.layer_norm(x)
        x = modulate(x, shift, scale)  # apply adaLN for the last layer
        x = self.linear(x)

        return x


class TimestepEmbedder(nn.Module):
    def __init__(self, freq_embed_size=256, hidden_size=384):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(freq_embed_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.freqembed_size = freq_embed_size

    @staticmethod
    def timestep_embedding(t_step: torch.Tensor, dim: int, max_value: int = 10000):
        half_dim = dim // 2
        scale_factor = -math.log(max_value)
        freqs = scale_factor * torch.arange(start=0, end=half_dim, dtype=torch.float32)
        freqs = (freqs / half_dim).to(t_step.device)
        args = t_step[:, None].float() * freqs[None]
        sincos_embed = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        return (
            sincos_embed
            if dim % 2 == 0
            else torch.cat(
                [sincos_embed, torch.zeros_like(sincos_embed[:, :1])], dim=-1
            )
        )

    def forward(self, t_step):
        time_freq = self.timestep_embedding(t_step, self.freqembed_size)
        time_embed = self.linear(time_freq)

        return time_embed
