import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PositionEmbeddingCoordsSine(nn.Module):
    """Similar to transformer's position encoding, but generalizes it to
    arbitrary dimensions and continuous coordinates.

    Args:
        n_dim: Number of input dimensions, e.g. 2 for image coordinates.
        d_model: Number of dimensions to encode into
        temperature:
        scale:
    """
    def __init__(self, n_dim: int = 1, d_model: int = 64, temperature=10000, scale=None):
        super().__init__()

        self.n_dim = n_dim
        self.num_pos_feats = d_model // n_dim // 2 * 2
        self.temperature = temperature
        self.padding = d_model - self.num_pos_feats * self.n_dim

        if scale is None:
            scale = 1.0
        self.scale = scale * 2 * math.pi

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz: Point positions (*, d_in)

        Returns:
            pos_emb (*, d_out)
        """
        assert xyz.shape[-1] == self.n_dim

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=xyz.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats)

        xyz = xyz * self.scale
        pos_divided = xyz.unsqueeze(-1) / dim_t
        pos_sin = pos_divided[..., 0::2].sin()
        pos_cos = pos_divided[..., 1::2].cos()
        pos_emb = torch.stack([pos_sin, pos_cos], dim=-1).reshape(*xyz.shape[:-1], -1)

        # Pad unused dimensions with zeros
        pos_emb = F.pad(pos_emb, (0, self.padding))
        return pos_emb


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, n_dim: int = 1, d_model: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, d_model)
        )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        return self.mlp(xyz)
    

'''
    Geotransformer Position Encoding 
'''
class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model):
        super(SinusoidalPositionalEmbedding, self).__init__()
        if d_model % 2 != 0:
            raise ValueError(f'Sinusoidal positional encoding with odd d_model: {d_model}')
        self.d_model = d_model
        div_indices = torch.arange(0, d_model, 2).float()
        div_term = torch.exp(div_indices * (-np.log(10000.0) / d_model))
        self.register_buffer('div_term', div_term)

    def forward(self, emb_indices):
        r"""Sinusoidal Positional Embedding.
        Args:
            emb_indices: torch.Tensor (*)
        Returns:
            embeddings: torch.Tensor (*, D)
        """
        input_shape = emb_indices.shape
        omegas = emb_indices.view(-1, 1, 1) * self.div_term.view(1, -1, 1)  # (-1, d_model/2, 1)
        sin_embeddings = torch.sin(omegas)
        cos_embeddings = torch.cos(omegas)
        embeddings = torch.cat([sin_embeddings, cos_embeddings], dim=2)  # (-1, d_model/2, 2)
        embeddings = embeddings.view(*input_shape, self.d_model)  # (*, d_model)
        embeddings = embeddings.detach()
        return embeddings
    

# def build_dropout_layer(p: Optional[float], **kwargs) -> nn.Module:
#     r"""Factory function for dropout layer."""
#     if p is None or p == 0:
#         return nn.Identity()
#     else:
#         return nn.Dropout(p=p, **kwargs)
    

class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(LearnablePositionalEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)  # (L, D)
        self.norm = nn.LayerNorm(embedding_dim)
        # self.dropout = build_dropout_layer(dropout)

    def forward(self, emb_indices):
        r"""Learnable Positional Embedding.
        `emb_indices` are truncated to fit the finite embedding space.
        Args:
            emb_indices: torch.LongTensor (*)
        Returns:
            embeddings: torch.Tensor (*, D)
        """
        input_shape = emb_indices.shape
        emb_indices = emb_indices.view(-1)
        max_emd_indices = torch.full_like(emb_indices, self.num_embeddings - 1)
        emb_indices = torch.minimum(emb_indices, max_emd_indices)
        embeddings = self.embeddings(emb_indices)  # (*, D)
        embeddings = self.norm(embeddings)
        # embeddings = self.dropout(embeddings)
        embeddings = embeddings.view(*input_shape, self.embedding_dim)
        return embeddings
    

'''
    LoFTR Position Encoding
'''
class PositionEncodingSine(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 2-dimensional images
    """

    def __init__(self, d_model, max_shape=(256, 256), temp_bug_fix=True):
        """
        Args:
            max_shape (tuple): for 1/8 featmap, the max length of 256 corresponds to 2048 pixels
            temp_bug_fix (bool): As noted in this [issue](https://github.com/zju3dv/LoFTR/issues/41),
                the original implementation of LoFTR includes a bug in the pos-enc impl, which has little impact
                on the final performance. For now, we keep both impls for backward compatability.
                We will remove the buggy impl after re-training all variants of our released models.
        """
        super().__init__()

        pe = torch.zeros((d_model, *max_shape))
        y_position = torch.ones(max_shape).cumsum(0).float().unsqueeze(0)
        x_position = torch.ones(max_shape).cumsum(1).float().unsqueeze(0)
        if temp_bug_fix:
            div_term = torch.exp(torch.arange(0, d_model//2, 2).float() * (-math.log(10000.0) / (d_model//2)))
        else:  # a buggy implementation (for backward compatability only)
            div_term = torch.exp(torch.arange(0, d_model//2, 2).float() * (-math.log(10000.0) / d_model//2))
        div_term = div_term[:, None, None]  # [C//4, 1, 1]
        pe[0::4, :, :] = torch.sin(x_position * div_term)
        pe[1::4, :, :] = torch.cos(x_position * div_term)
        pe[2::4, :, :] = torch.sin(y_position * div_term)
        pe[3::4, :, :] = torch.cos(y_position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)  # [1, C, H, W]

    def forward(self, x):
        """
        Args:
            x: [N, C, H, W]
        """
        return x + self.pe[:, :, :x.size(2), :x.size(3)]