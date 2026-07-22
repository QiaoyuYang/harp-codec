"""
Harp: Hierarchical Audio Representation with Prioritized bands

Revised implementation with ablation study support.

Ablation flags:
- use_contribution: bool (False = supervise cumulative only, no Δ)
- use_stop_gradient: bool (False = band loss gradients flow to all groups)
- band_order: str ("ascending", "descending", "random")

Note on gradient flow:
- Reconstruction, adversarial, and feature matching losses ALWAYS backpropagate
  through ALL stages (this is correct per the paper)
- Stop-gradient (use_stop_gradient) only affects BAND losses, isolating each
  group's band loss to only update that group's parameters

Existing config options that support ablations:
- learnable_bands: bool (False = fixed bands)
- baseline: float (0.0 = hard bands, 0.5 = weak, 1.0 = no priority)
- stage_groups: list (modify for 2/9 group ablations)
- lambdas.band/total: float (0.0 = no band supervision)

Supports variable codebook configurations:
- Default: 9 codebooks, 4 bands with 3-2-2-2 distribution (~7.7 kbps full rate)
- Configurable via stage_groups parameter
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict

from .dac import DAC


@dataclass
class HarpConfig:
    """Configuration for Harp model"""
    # DAC config
    encoder_dim: int = 64
    encoder_rates: List[int] = field(default_factory=lambda: [2, 4, 8, 8])
    decoder_dim: int = 1536
    decoder_rates: List[int] = field(default_factory=lambda: [8, 8, 4, 2])
    n_codebooks: int = 9
    codebook_size: int = 1024
    codebook_dim: int = 8
    quantizer_dropout: float = 0.0
    sample_rate: int = 44100
    
    # Band prioritization
    n_mels: int = 80
    n_fft: int = 1024
    hop_length: int = 256
    # Default: 9 codebooks with 3-2-2-2 distribution (bass-heavy)
    stage_groups: Tuple[Tuple[int, int], ...] = ((0, 3), (3, 5), (5, 7), (7, 9))
    learnable_bands: bool = True
    baseline: float = 0.3
    
    # Group-aware dropout (only drops higher bands)
    group_dropout_prob: float = 0.5
    min_groups: int = 1  # Always keep at least this many groups (lowest frequencies)
    
    # === ABLATION FLAGS ===
    use_contribution: bool = True      # False = "cumulative only (no Δ)" ablation
    use_stop_gradient: bool = True     # False = "no stop-gradient" ablation
    band_order: str = "ascending"      # "descending" = reversed, "random" = random assignment
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "HarpConfig":
        if "dac" in config_dict:
            dac_config = config_dict["dac"]
            band_config = config_dict.get("band_priority", {})
            return cls(
                encoder_dim=dac_config.get("encoder_dim", 64),
                encoder_rates=dac_config.get("encoder_rates", [2, 4, 8, 8]),
                decoder_dim=dac_config.get("decoder_dim", 1536),
                decoder_rates=dac_config.get("decoder_rates", [8, 8, 4, 2]),
                n_codebooks=dac_config.get("n_codebooks", 9),
                codebook_size=dac_config.get("codebook_size", 1024),
                codebook_dim=dac_config.get("codebook_dim", 8),
                quantizer_dropout=dac_config.get("quantizer_dropout", 0.0),
                sample_rate=dac_config.get("sample_rate", 44100),
                n_mels=band_config.get("n_mels", 80),
                n_fft=band_config.get("n_fft", 1024),
                hop_length=band_config.get("hop_length", 256),
                stage_groups=tuple(tuple(g) for g in band_config.get(
                    "stage_groups", [[0, 3], [3, 5], [5, 7], [7, 9]]
                )),
                learnable_bands=band_config.get("learnable_bands", True),
                baseline=band_config.get("baseline", 0.3),
                group_dropout_prob=band_config.get("group_dropout_prob", 0.5),
                min_groups=band_config.get("min_groups", 1),
                # Ablation flags
                use_contribution=band_config.get("use_contribution", True),
                use_stop_gradient=band_config.get("use_stop_gradient", True),
                band_order=band_config.get("band_order", "ascending"),
            )
        return cls(**config_dict)


class GroupAwareDropout(nn.Module):
    """
    Dropout that only removes higher frequency bands, always keeping lower bands.
    This ensures the model learns strong low-frequency representations.
    """
    
    def __init__(
        self,
        n_groups: int = 4,
        dropout_prob: float = 0.5,
        min_groups: int = 1,
    ):
        super().__init__()
        self.n_groups = n_groups
        self.dropout_prob = dropout_prob
        self.min_groups = min_groups
        
        # Weights biased toward using more groups
        # e.g., for 4 groups with min_groups=1: weights = [1, 2, 3, 4]
        weights = torch.arange(1, n_groups - min_groups + 2, dtype=torch.float)
        self.register_buffer('sampling_weights', weights)
    
    def sample_n_groups(self) -> int:
        """Sample number of groups to use, biased toward more groups."""
        if not self.training or torch.rand(1).item() > self.dropout_prob:
            return self.n_groups
        
        # Sample with bias toward more groups
        idx = torch.multinomial(self.sampling_weights, 1).item()
        return self.min_groups + idx
    
    def forward(self, n_groups: Optional[int] = None) -> int:
        """Returns number of groups to use."""
        if n_groups is not None:
            return n_groups
        return self.sample_n_groups()


class BandPrioritizer(nn.Module):
    """
    Computes frequency-weighted losses that prioritize different mel bands
    for different stage groups, while allowing cross-band contributions.
    
    Implements Eq. 4-6 from the paper:
    - Soft Gaussian weighting centered on target band (Eq. 4)
    - Normalized weights for stable loss scaling (Eq. 5)
    - Subband contribution supervision (Eq. 6)
    
    Supports ablations:
    - band_order: "ascending" (default), "descending" (reversed), "random"
    """
    
    def __init__(
        self,
        n_mels: int = 80,
        n_groups: int = 4,
        learnable: bool = True,
        baseline: float = 0.3,
        band_order: str = "ascending",
    ):
        super().__init__()
        self.n_mels = n_mels
        self.n_groups = n_groups
        self.learnable = learnable
        self.band_order = band_order
        
        # Initialize centers evenly across mel bins (paper: "divide the mel spectrum evenly")
        centers = torch.tensor([
            n_mels * (i + 0.5) / n_groups for i in range(n_groups)
        ])
        
        # Apply band ordering for ablation studies
        if band_order == "descending":
            # Reversed: treble in early stages, bass in later stages
            centers = centers.flip(0)
        elif band_order == "random":
            # Random assignment of bands to groups
            perm = torch.randperm(n_groups)
            centers = centers[perm]
            # Store permutation for reproducibility analysis
            self.register_buffer('band_permutation', perm)
        # "ascending" (default): bass in early stages, treble in later stages
        
        # Initialize widths as σ_k = M/(3K) per paper
        widths = torch.full((n_groups,), n_mels / (3 * n_groups))
        
        if learnable:
            self.centers = nn.Parameter(centers)
            self.widths = nn.Parameter(widths)
            self.baseline = nn.Parameter(torch.tensor(baseline))
        else:
            self.register_buffer('centers', centers)
            self.register_buffer('widths', widths)
            self.register_buffer('baseline', torch.tensor(baseline))
    
    def get_weights(self, group_idx: int) -> torch.Tensor:
        """
        Compute normalized band weights for a group (Eq. 4-5).
        
        Returns:
            Normalized weights summing to 1, shape (n_mels,)
        """
        bins = torch.arange(self.n_mels, device=self.centers.device, dtype=torch.float)
        
        center = self.centers[group_idx]
        # Use softplus to ensure positive width, add small constant for stability
        width = F.softplus(self.widths[group_idx]) + 1.0
        # Sigmoid ensures baseline is in (0, 1)
        baseline = torch.sigmoid(self.baseline) if self.learnable else self.baseline
        
        # Eq. 4: Unnormalized weights with Gaussian priority
        priority = torch.exp(-0.5 * ((bins - center) / width) ** 2)
        weights_unnorm = baseline + (1 - baseline) * priority
        
        # Eq. 5: Normalize to ensure consistent loss magnitude
        weights = weights_unnorm / weights_unnorm.sum()
        
        return weights
    
    def forward(
        self,
        mel_contribution: torch.Tensor,
        mel_target_contribution: torch.Tensor,
        group_idx: int,
    ) -> torch.Tensor:
        """
        Compute band loss for a group using subband contribution supervision (Eq. 6).
        
        Args:
            mel_contribution: ΔM_k = M(x̂_≤k) - M(sg[x̂_≤k-1]), shape (B, n_mels, T)
            mel_target_contribution: ΔM_k* = M(x) - M(sg[x̂_≤k-1]), shape (B, n_mels, T)
            group_idx: Index of the current group
            
        Returns:
            Scalar loss for this group
        """
        # Get normalized weights, shape (n_mels,) -> (1, n_mels, 1) for broadcasting
        weights = self.get_weights(group_idx).view(1, -1, 1)
        
        # Eq. 6: Weighted L1 loss between contribution and target contribution
        error = (mel_contribution - mel_target_contribution).abs()
        weighted_error = weights * error
        
        return weighted_error.mean()
    
    @torch.no_grad()
    def get_band_info(self) -> Dict[str, any]:
        """Get current band parameters for logging/analysis."""
        info = {
            'centers': self.centers.detach().cpu().tolist(),
            'widths': F.softplus(self.widths).detach().cpu().tolist(),
            'baseline': (torch.sigmoid(self.baseline) if self.learnable 
                        else self.baseline).item(),
            'band_order': self.band_order,
        }
        if hasattr(self, 'band_permutation'):
            info['band_permutation'] = self.band_permutation.cpu().tolist()
        return info


class Harp(nn.Module):
    """
    Harp: Hierarchical Audio Representation with Prioritized bands
    
    Default configuration: 9 codebooks, 4 frequency bands (3-2-2-2 distribution):
        - Group 0 (codebooks 0-2): Bass (~0-1 kHz) - 3 codebooks
        - Group 1 (codebooks 3-4): Low-mid (~1-4 kHz) - 2 codebooks
        - Group 2 (codebooks 5-6): High-mid (~4-10 kHz) - 2 codebooks
        - Group 3 (codebooks 7-8): Treble (~10-22 kHz) - 2 codebooks
    
    Bitrate tiers (~86 Hz frame rate, 10 bits/codebook):
        - 1 group (3 codebooks): ~2.6 kbps (bass only)
        - 2 groups (5 codebooks): ~4.3 kbps (bass + low-mid)
        - 3 groups (7 codebooks): ~6.0 kbps (bass + low-mid + high-mid)
        - 4 groups (9 codebooks): ~7.7 kbps (full rate)
    
    Key design principles from paper:
        - Cumulative decoding: Each group sees all lower-frequency content
        - Stop-gradient isolation: Each group's loss only updates that group
        - Soft band weighting: Gaussian weights with baseline for cross-band coherence
        - Bass-heavy distribution: More codebooks for perceptually important bass
    
    Ablation support:
        - use_contribution: False = supervise cumulative only (no Δ)
        - use_stop_gradient: False = band loss gradients flow to all groups (not just current)
        - band_order: "descending" = reversed order, "random" = random assignment
        - learnable_bands: False = fixed band centers/widths
        - baseline: 0.0 = hard bands, 1.0 = no priority
    
    Note: use_stop_gradient only affects band losses. Reconstruction, adversarial,
    and feature matching losses always backpropagate through ALL stages.
    """
    
    def __init__(self, config: HarpConfig = None, **kwargs):
        super().__init__()
        
        if config is None:
            config = HarpConfig(**kwargs)
        self.config = config
        
        # Core DAC model
        self.dac = DAC(
            encoder_dim=config.encoder_dim,
            encoder_rates=config.encoder_rates,
            latent_dim=None,
            decoder_dim=config.decoder_dim,
            decoder_rates=config.decoder_rates,
            n_codebooks=config.n_codebooks,
            codebook_size=config.codebook_size,
            codebook_dim=config.codebook_dim,
            quantizer_dropout=config.quantizer_dropout,
            sample_rate=config.sample_rate,
        )
        
        # Band prioritizer for training
        n_groups = len(config.stage_groups)
        self.band_prioritizer = BandPrioritizer(
            n_mels=config.n_mels,
            n_groups=n_groups,
            learnable=config.learnable_bands,
            baseline=config.baseline,
            band_order=config.band_order,
        )
        
        # Group-aware dropout
        self.group_dropout = GroupAwareDropout(
            n_groups=n_groups,
            dropout_prob=config.group_dropout_prob,
            min_groups=config.min_groups,
        )
        
        # Mel spectrogram transform
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
        )
        
        self.sample_rate = config.sample_rate
        self.hop_length = self.dac.hop_length
        self.n_codebooks = config.n_codebooks
        
        # Log ablation configuration
        self._log_ablation_config()
    
    def _log_ablation_config(self):
        """Log ablation-relevant configuration for debugging."""
        ablation_info = {
            'use_contribution': self.config.use_contribution,
            'use_stop_gradient': self.config.use_stop_gradient,
            'band_order': self.config.band_order,
            'learnable_bands': self.config.learnable_bands,
            'baseline': self.config.baseline,
            'n_groups': len(self.config.stage_groups),
            'stage_groups': self.config.stage_groups,
        }
        print(f"[HARP] Ablation config: {ablation_info}")
    
    def compute_mel(self, audio: torch.Tensor) -> torch.Tensor:
        """Compute log-mel spectrogram from audio."""
        if self.mel_spec.mel_scale.fb.device != audio.device:
            self.mel_spec = self.mel_spec.to(audio.device)
        mel = self.mel_spec(audio.squeeze(1))
        return torch.log(mel + 1e-5)
    
    def preprocess(self, audio: torch.Tensor, sample_rate: Optional[int] = None) -> torch.Tensor:
        return self.dac.preprocess(audio, sample_rate)
    
    def encode(self, audio: torch.Tensor, n_quantizers: Optional[int] = None):
        return self.dac.encode(audio, n_quantizers)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dac.decode(z)
    
    def forward(
        self,
        audio: torch.Tensor,
        sample_rate: Optional[int] = None,
        n_quantizers: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        return self.dac(audio, sample_rate, n_quantizers)
    
    def get_n_groups(self, n_groups: Optional[int] = None) -> int:
        """Get number of groups to use, applying dropout during training."""
        return self.group_dropout(n_groups)
    
    def groups_to_quantizers(self, n_groups: int) -> int:
        """Convert number of groups to number of quantizers."""
        if n_groups >= len(self.config.stage_groups):
            return self.n_codebooks
        return self.config.stage_groups[n_groups - 1][1] if n_groups > 0 else 0
    
    def forward_with_bands(
        self,
        audio: torch.Tensor,
        sample_rate: Optional[int] = None,
        n_groups: Optional[int] = None,
        apply_dropout: bool = True,
    ) -> Dict[str, any]:
        """
        Forward pass with progressive decoding for band-prioritized training.
        
        Implements Algorithm 1 from the paper:
        - Each group's cumulative reconstruction is decoded and stored
        - All stages receive gradients from final reconstruction/adversarial losses
        - Stop-gradient for band loss isolation is handled in compute_band_losses()
        
        Args:
            audio: Input audio tensor
            sample_rate: Optional sample rate for resampling
            n_groups: Number of groups to use (None = use dropout sampling)
            apply_dropout: Whether to apply group dropout during training
            
        Returns:
            Dictionary containing:
                - cumulative_recons: List of cumulative reconstructions per group
                - codes: Stacked codebook indices
                - audio: Final reconstruction
                - commitment_loss, codebook_loss: VQ losses
                - n_groups, n_quantizers: Active counts
        """
        # Determine number of groups
        if apply_dropout and self.training:
            active_n_groups = self.get_n_groups(n_groups)
        else:
            active_n_groups = n_groups if n_groups is not None else len(self.config.stage_groups)
        
        # Get active stage groups (always from lowest frequency up)
        active_stage_groups = self.config.stage_groups[:active_n_groups]
        n_q = self.groups_to_quantizers(active_n_groups)
        
        # Preprocess and encode
        audio = self.preprocess(audio, sample_rate)
        z = self.dac.encoder(audio)
        
        quantizer = self.dac.quantizer
        decoder = self.dac.decoder
        
        outputs = {
            'preprocessed_audio': audio,
            'cumulative_recons': [],
            'codes': [],
            'commitment_loss': 0.0,
            'codebook_loss': 0.0,
            'n_groups': active_n_groups,
            'n_quantizers': n_q,
        }
        
        residual = z
        quantized_sum = torch.zeros_like(z)
        stage_idx = 0
        
        for group_idx, (start, end) in enumerate(active_stage_groups):
            # Quantize stages in this group
            for i in range(start, min(end, n_q)):
                # VectorQuantize returns: z_q, commitment_loss, codebook_loss, indices, z_e
                z_q_i, commit_loss_i, cb_loss_i, indices_i, z_e_i = quantizer.quantizers[i](residual)
                
                # Detach from residual computation (standard RVQ practice)
                residual = residual - z_q_i.detach()
                # Accumulate quantized representations (gradients flow through)
                quantized_sum = quantized_sum + z_q_i
                
                outputs['codes'].append(indices_i)
                outputs['commitment_loss'] = outputs['commitment_loss'] + commit_loss_i.mean()
                outputs['codebook_loss'] = outputs['codebook_loss'] + cb_loss_i.mean()
                stage_idx += 1
            
            # Decode cumulative sum for this group
            cumulative_recon = decoder(quantized_sum)
            outputs['cumulative_recons'].append(cumulative_recon)
            
            # NOTE: We do NOT detach quantized_sum here!
            # All stages should receive gradients from L_rec, L_adv, L_feat
            # Stop-gradient is only applied in compute_band_losses() for band loss isolation
            
            if stage_idx >= n_q:
                break
        
        # Normalize losses
        if stage_idx > 0:
            outputs['commitment_loss'] = outputs['commitment_loss'] / stage_idx
            outputs['codebook_loss'] = outputs['codebook_loss'] / stage_idx
        
        outputs['z'] = quantized_sum
        outputs['codes'] = torch.stack(outputs['codes'], dim=1) if outputs['codes'] else None
        outputs['audio'] = outputs['cumulative_recons'][-1] if outputs['cumulative_recons'] else None
        
        return outputs
    
    def compute_band_losses(
        self,
        outputs: Dict[str, any],
        audio: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute band-prioritized losses using subband contribution supervision (Eq. 6).
        
        For each group k:
            ΔM_k = M(x̂_≤k) - M(sg[x̂_≤k-1])     # what group k contributed
            ΔM_k* = M(x) - M(sg[x̂_≤k-1])        # what should have been added
            L_band^(k) = Σ w_k(m) |ΔM_k - ΔM_k*|
        
        The stop-gradient on x̂_≤k-1 ensures that L_band^(k) only backpropagates
        through group k's quantizers, not through earlier groups. This is the key
        mechanism for gradient isolation per group.
        
        ABLATION: When use_stop_gradient=False, band loss gradients flow through
        all previous groups as well.
        
        NOTE: This stop-gradient only affects band losses. The reconstruction,
        adversarial, and feature matching losses (computed on outputs['audio'])
        backpropagate through ALL stages.
        
        Args:
            outputs: Dictionary from forward_with_bands
            audio: Original input audio (preprocessed)
            
        Returns:
            Dictionary of band losses per group and total
        """
        mel_target = self.compute_mel(audio)
        
        losses = {}
        prev_mel = None  # M(sg[x̂_≤k-1]), starts as None (zeros conceptually)
        
        for group_idx, cumulative_recon in enumerate(outputs['cumulative_recons']):
            mel_recon = self.compute_mel(cumulative_recon)
            
            # ABLATION: use_contribution controls whether we supervise the
            # contribution (default) or the cumulative reconstruction directly
            if self.config.use_contribution:
                # Default behavior: Subband contribution supervision (Eq. 6)
                if prev_mel is None:
                    # First group: contribution is the full reconstruction
                    mel_contribution = mel_recon
                    mel_target_contribution = mel_target
                else:
                    # Subsequent groups: contribution is the difference
                    mel_contribution = mel_recon - prev_mel
                    mel_target_contribution = mel_target - prev_mel
            else:
                # ABLATION: "cumulative only (no Δ)" - supervise cumulative directly
                mel_contribution = mel_recon
                mel_target_contribution = mel_target
            
            # Compute weighted band loss
            priority_loss = self.band_prioritizer(
                mel_contribution, mel_target_contribution, group_idx
            )
            losses[f'band/group_{group_idx}'] = priority_loss
            
            # Stop-gradient for band loss isolation (paper Section 4.4)
            # When use_stop_gradient=True (default): Band loss L_band^(k) only
            # backpropagates through group k's quantizers
            # When use_stop_gradient=False (ablation): Band loss gradients flow 
            # through all previous groups too
            if self.config.use_stop_gradient:
                prev_mel = mel_recon.detach()
            else:
                prev_mel = mel_recon
        
        losses['band/total'] = sum(v for k, v in losses.items() if k.startswith('band/group'))
        losses['band/n_groups'] = outputs['n_groups']
        
        return losses
    
    @torch.no_grad()
    def analyze_bands(self, audio: torch.Tensor) -> Dict[str, any]:
        """
        Analyze frequency specialization of each stage group.
        
        Computes spectral centroid and spread for each group's contribution,
        as described in Section 5.5 of the paper.
        """
        outputs = self.forward_with_bands(audio, apply_dropout=False)
        
        analysis = {
            'band_info': self.band_prioritizer.get_band_info(),
            'group_stats': [],
            'energy_distributions': [],
            'ablation_config': {
                'use_contribution': self.config.use_contribution,
                'use_stop_gradient': self.config.use_stop_gradient,
                'band_order': self.config.band_order,
            }
        }
        
        prev_recon = None
        for group_idx, cumulative_recon in enumerate(outputs['cumulative_recons']):
            # Compute contribution in audio domain
            if prev_recon is None:
                contribution = cumulative_recon
            else:
                contribution = cumulative_recon - prev_recon
            prev_recon = cumulative_recon
            
            # Analyze spectral content of contribution
            mel_contrib = self.compute_mel(contribution.abs() + 1e-8)
            energy = mel_contrib.exp().mean(dim=(0, -1))  # Average over batch and time
            
            # Compute centroid and spread (normalized to [0, 1])
            bins = torch.arange(self.config.n_mels, device=energy.device, dtype=torch.float)
            total_energy = energy.sum() + 1e-8
            centroid = (energy * bins).sum() / total_energy
            spread = ((bins - centroid).pow(2) * energy).sum() / total_energy
            spread = spread.sqrt()
            
            analysis[f'centroid/group_{group_idx}'] = centroid.item() / self.config.n_mels
            analysis[f'spread/group_{group_idx}'] = spread.item() / self.config.n_mels
            analysis['group_stats'].append({
                'centroid': centroid.item(),
                'spread': spread.item(),
                'centroid_normalized': centroid.item() / self.config.n_mels,
                'spread_normalized': spread.item() / self.config.n_mels,
            })
            analysis['energy_distributions'].append(energy.cpu().numpy())
        
        return analysis
    
    def get_bitrate(self, n_groups: Optional[int] = None) -> float:
        """Calculate bitrate in kbps for given number of groups."""
        if n_groups is None:
            n_groups = len(self.config.stage_groups)
        n_q = self.groups_to_quantizers(n_groups)
        
        import math
        frame_rate = self.sample_rate / self.hop_length
        bits_per_frame = n_q * math.log2(self.config.codebook_size)
        return frame_rate * bits_per_frame / 1000
    
    def get_bitrate_tiers(self) -> Dict[int, float]:
        """Get available bitrate tiers based on group configurations."""
        return {
            n_groups: self.get_bitrate(n_groups)
            for n_groups in range(1, len(self.config.stage_groups) + 1)
        }
    
    def get_ablation_summary(self) -> Dict[str, any]:
        """Get summary of ablation-relevant settings for logging."""
        return {
            'use_contribution': self.config.use_contribution,
            'use_stop_gradient': self.config.use_stop_gradient,
            'band_order': self.config.band_order,
            'learnable_bands': self.config.learnable_bands,
            'baseline': self.config.baseline,
            'n_groups': len(self.config.stage_groups),
            'stage_groups': list(self.config.stage_groups),
        }
    
    def compress(self, audio_signal, n_groups: Optional[int] = None, **kwargs):
        """Compress audio with optional bitrate control via n_groups."""
        if n_groups is not None:
            n_q = self.groups_to_quantizers(n_groups)
            kwargs['n_quantizers'] = n_q
        return self.dac.compress(audio_signal, **kwargs)
    
    def decompress(self, compressed, **kwargs):
        return self.dac.decompress(compressed, **kwargs)