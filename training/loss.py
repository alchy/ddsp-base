"""
training/loss.py — Loss functions for DDSP training

mrstft_loss:    Multi-Resolution STFT loss (spectral convergence + log-magnitude L1)
_attack_weight: Attack-emphasis weighting derived from loudness derivative
"""

import torch
import torch.nn.functional as F


def _attack_weight(lo_db: torch.Tensor, n_samples: int,
                   alpha: float = 4.0, sigma: float = 2.0) -> torch.Tensor:
    """Attack-emphasis weight upsampled to sample level.

    lo_db   : (B, F) loudness in dB per frame
    returns : (B, 1, n_samples) weight tensor, floor=1.0, peak=1+alpha
    """
    B, n_frames = lo_db.shape
    radius  = int(4 * sigma + 0.5)
    k       = torch.arange(-radius, radius + 1, dtype=lo_db.dtype, device=lo_db.device)
    kernel  = torch.exp(-0.5 * (k / sigma) ** 2)
    kernel  = kernel / kernel.sum()
    lo_smooth = F.conv1d(lo_db.unsqueeze(1), kernel.view(1, 1, -1),
                          padding=radius).squeeze(1)
    dl      = torch.diff(lo_smooth, prepend=lo_smooth[:, :1], dim=1)
    dl_pos  = dl.clamp(min=0.0)
    dl_norm = dl_pos / (dl_pos.amax(dim=1, keepdim=True) + 1e-6)
    attack_w = 1.0 + alpha * dl_norm
    return F.interpolate(attack_w.unsqueeze(1), size=n_samples,
                          mode='linear', align_corners=False)


def mrstft_loss(pred: torch.Tensor, target: torch.Tensor,
                fft_sizes: tuple = (256, 1024, 4096, 16384)) -> torch.Tensor:
    """Multi-Resolution STFT loss.

    pred, target : (B, C, T)
    Combines spectral convergence + log-magnitude L1 at multiple resolutions.
    """
    B, C, T = pred.shape
    p_flat  = pred.reshape(B * C, T)
    t_flat  = target.reshape(B * C, T)
    total, n_valid = pred.sum() * 0.0, 0
    for n_fft in fft_sizes:
        if n_fft >= T:
            continue
        hop = n_fft // 4
        win = torch.hann_window(n_fft, device=pred.device)
        Sp  = torch.stft(p_flat, n_fft, hop, n_fft, win, return_complex=True)
        St  = torch.stft(t_flat, n_fft, hop, n_fft, win, return_complex=True)
        mp  = (Sp.real.pow(2) + Sp.imag.pow(2) + 1e-8).sqrt()
        mt  = (St.real.pow(2) + St.imag.pow(2) + 1e-8).sqrt()
        sc  = ((mp - mt).pow(2).sum() + 1e-8).sqrt() / (mt.pow(2).sum() + 1e-8).sqrt()
        lm  = F.l1_loss(torch.log(mp + 1e-7), torch.log(mt + 1e-7))
        total += sc + lm
        n_valid += 1
    return total / max(n_valid, 1)
