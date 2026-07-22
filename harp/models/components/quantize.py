from typing import Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .layers import WNConv1d

class VectorQuantize(nn.Module):

    def __init__(
        self, 
        input_dim: int, 
        codebook_size: int, 
        codebook_dim: int,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.ema_decay = ema_decay

        self.in_proj = WNConv1d(input_dim, codebook_dim, kernel_size=1)
        self.out_proj = WNConv1d(codebook_dim, input_dim, kernel_size=1)
        self.codebook = nn.Embedding(codebook_size, codebook_dim)
        
        # EMA usage tracking
        self.register_buffer("ema_usage", torch.zeros(codebook_size))
        self.register_buffer("usage_initialized", torch.tensor(False))

    def forward(self, z):
        z_e = self.in_proj(z)
        z_q, indices = self.decode_latents(z_e)

        commitment_loss = F.mse_loss(z_e, z_q.detach(), reduction="none").mean([1, 2])
        codebook_loss = F.mse_loss(z_q, z_e.detach(), reduction="none").mean([1, 2])

        z_q = z_e + (z_q - z_e).detach()
        z_q = self.out_proj(z_q)
        
        # Update usage
        if self.training:
            self._update_usage(indices)

        return z_q, commitment_loss, codebook_loss, indices, z_e

    def _update_usage(self, indices: torch.Tensor):
        """Update EMA usage statistics."""
        with torch.no_grad():
            flat_indices = indices.flatten()
            batch_usage = torch.zeros(self.codebook_size, device=indices.device)
            unique_codes, counts = flat_indices.unique(return_counts=True)
            batch_usage[unique_codes] = counts.float()
            batch_usage = batch_usage / flat_indices.numel()
            
            if not self.usage_initialized:
                self.ema_usage.copy_(batch_usage)
                self.usage_initialized.fill_(True)
            else:
                self.ema_usage.mul_(self.ema_decay).add_(batch_usage, alpha=1 - self.ema_decay)

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id):
        return self.embed_code(embed_id).transpose(1, 2)

    def decode_latents(self, latents):
        encodings = rearrange(latents, "b d t -> (b t) d")
        codebook = self.codebook.weight

        encodings = F.normalize(encodings, dim=-1)
        codebook = F.normalize(codebook, dim=-1)

        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = rearrange((-dist).max(1)[1], "(b t) -> b t", b=latents.size(0))
        z_q = self.decode_code(indices)
        return z_q, indices
    
    @property
    def utilization(self) -> float:
        if not self.usage_initialized:
            return 0.0
        uniform_usage = 1.0 / self.codebook_size
        threshold = uniform_usage / 10.0
        return (self.ema_usage > threshold).sum().item() / self.codebook_size
    
    @property
    def perplexity(self) -> float:
        if not self.usage_initialized:
            return 0.0
        usage = self.ema_usage + 1e-10
        usage = usage / usage.sum()
        entropy = -(usage * torch.log(usage)).sum()
        return torch.exp(entropy).item()
    
    def reset_usage(self):
        self.ema_usage.zero_()
        self.usage_initialized.fill_(False)


class ResidualVectorQuantize(nn.Module):

    def __init__(
        self,
        input_dim: int = 512,
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: Union[int, list] = 8,
        quantizer_dropout: float = 0.0,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        if isinstance(codebook_dim, int):
            codebook_dim = [codebook_dim] * n_codebooks

        self.n_codebooks = n_codebooks
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size
        self.quantizer_dropout = quantizer_dropout

        self.quantizers = nn.ModuleList([
            VectorQuantize(input_dim, codebook_size, codebook_dim[i], ema_decay)
            for i in range(n_codebooks)
        ])

    def forward(self, z, n_quantizers: int = None):
        z_q = 0
        residual = z
        commitment_loss = 0
        codebook_loss = 0

        codebook_indices = []
        latents = []

        if n_quantizers is None:
            n_quantizers = self.n_codebooks
        if self.training:
            n_quantizers = torch.ones((z.shape[0],)) * self.n_codebooks + 1
            dropout = torch.randint(1, self.n_codebooks + 1, (z.shape[0],))
            n_dropout = int(z.shape[0] * self.quantizer_dropout)
            n_quantizers[:n_dropout] = dropout[:n_dropout]
            n_quantizers = n_quantizers.to(z.device)

        for i, quantizer in enumerate(self.quantizers):
            if self.training is False and i >= n_quantizers:
                break

            z_q_i, commitment_loss_i, codebook_loss_i, indices_i, z_e_i = quantizer(residual)

            mask = (
                torch.full((z.shape[0],), fill_value=i, device=z.device) < n_quantizers
            )
            z_q = z_q + z_q_i * mask[:, None, None]
            residual = residual - z_q_i

            commitment_loss += (commitment_loss_i * mask).mean()
            codebook_loss += (codebook_loss_i * mask).mean()

            codebook_indices.append(indices_i)
            latents.append(z_e_i)

        codes = torch.stack(codebook_indices, dim=1)
        latents = torch.cat(latents, dim=1)

        return z_q, codes, latents, commitment_loss, codebook_loss

    def from_codes(self, codes: torch.Tensor):
        z_q = 0.0
        z_p = []
        for i in range(codes.shape[1]):
            z_p_i = self.quantizers[i].decode_code(codes[:, i, :])
            z_p.append(z_p_i)
            z_q = z_q + self.quantizers[i].out_proj(z_p_i)
        return z_q, torch.cat(z_p, dim=1), codes
    
    @property
    def utilization(self) -> List[float]:
        return [q.utilization for q in self.quantizers]
    
    @property
    def perplexity(self) -> List[float]:
        return [q.perplexity for q in self.quantizers]
    
    def reset_usage(self):
        for q in self.quantizers:
            q.reset_usage()
