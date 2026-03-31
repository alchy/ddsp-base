"""
synth/noise.py — Filtered noise synthesizer (harmonic-relative STFT shaping)

The spectral envelope is stored in harmonic-relative space:
    noise bin k  →  absolute freq  k · f0 · N_HARM / N

This ensures the learned spectral shape scales with pitch — a peak between
the 2nd and 3rd harmonic stays there regardless of the note played.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import SR, F0_MIN, N_HARM, NOISE_FFT


class NoiseSynth(nn.Module):
    """
    Filtered noise via STFT magnitude shaping.
    Output: (B, 1, n_samples)
    """

    def __init__(self, n_fft: int = NOISE_FFT):
        super().__init__()
        self.n_fft = n_fft
        self.hop   = n_fft // 4

    def forward(self, noise_mags: torch.Tensor, f0_hz: torch.Tensor,
                n_samples: int) -> torch.Tensor:
        """
        noise_mags : (B, T_frames, N)  spectral shape in harmonic-relative space
        f0_hz      : (B, T)            fundamental frequency in Hz
        n_samples  : int
        """
        B, T_frames, N = noise_mags.shape
        device = noise_mags.device
        n_fft, hop = self.n_fft, self.hop
        n_bins = n_fft // 2 + 1

        window = torch.hann_window(n_fft, device=device)
        noise  = torch.randn(B, n_samples, device=device)
        stft   = torch.stft(noise, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                             window=window, return_complex=True)
        T_stft = stft.shape[-1]

        # Upsample noise_mags to STFT frame rate
        mag_up = F.interpolate(noise_mags.permute(0, 2, 1), size=T_stft,
                                mode='linear', align_corners=False)          # (B, N, T_stft)

        # Warp harmonic-relative bins → absolute STFT bins
        f0_mean    = f0_hz.clamp(min=F0_MIN).mean(dim=1)                    # (B,)
        stft_freqs = torch.arange(n_bins, dtype=torch.float32, device=device) \
                     * (SR / n_fft)                                          # (n_bins,)
        src_pos    = stft_freqs.unsqueeze(0) / f0_mean.unsqueeze(1) \
                     * (N / N_HARM)                                          # (B, n_bins)
        src_pos    = src_pos.clamp(0.0, N - 1.0)

        # Linear interpolation
        idx_lo  = src_pos.long()
        idx_hi  = (idx_lo + 1).clamp(max=N - 1)
        frac    = (src_pos - idx_lo.float()) \
                  .unsqueeze(-1).expand(-1, -1, T_stft)
        idx_lo  = idx_lo.unsqueeze(-1).expand(-1, -1, T_stft)
        idx_hi  = idx_hi.unsqueeze(-1).expand(-1, -1, T_stft)
        val_lo  = torch.gather(mag_up, 1, idx_lo)
        val_hi  = torch.gather(mag_up, 1, idx_hi)
        mag_abs = val_lo + frac * (val_hi - val_lo)                          # (B, n_bins, T_stft)

        out = torch.istft(stft * mag_abs, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                           window=window, length=n_samples)
        return out.unsqueeze(1)
