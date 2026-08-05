"""
Wu et al. 2024 (RSSRnet) I/O adapter for MMECG.

The MMECG dataset provides 50-bin FMCW radar @ 200 Hz over 8s (1600 pt) windows.
Wu's paper takes a single 1-D displacement signal @ 100 Hz, STFT'd into freq-time.

This adapter:
  1. top-1 range-bin selection by 0.8-3.5 Hz cardiac-band energy (deferred to dataset
     via topk_bins=1) — caller is responsible for setting up the dataset that way.
  2. Resample 200 → 100 Hz (linear interpolation in torch; torch.stft handles the rest).
  3. STFT (n_fft=256, win_length=40 = 0.4 s, hop_length=10 = 0.1 s).
  4. Keep first 64 freq bins (matches paper) → stack real+imag → (B, 2, 64, T_stft).

Inverse path:
  5. (B, 2, 64, T_stft) → complex tensor → ISTFT (length=800) → (B, 1, 800) @ 100 Hz.
  6. Resample 100 → 200 Hz → (B, 1, 1600).

All metrics are computed in the original 200 Hz / 1600 pt domain via
src.utils.metrics, so this adapter must round-trip the time grid faithfully.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

# ── STFT / sampling constants ────────────────────────────────────────────────
STFT_N_FFT = 256
STFT_WIN = 40          # 0.4 s at 100 Hz
STFT_HOP = 10          # 0.1 s at 100 Hz
KEEP_FREQ_BINS = 64    # first 64 bins out of (n_fft//2 + 1 = 129)
SR_NATIVE = 200        # MMECG native rate
SR_TARGET = 100        # Wu paper rate
WIN_LEN_NATIVE = 1600  # 8 s @ 200 Hz
WIN_LEN_TARGET = 800   # 8 s @ 100 Hz


def _resample(x: torch.Tensor, sr_in: int, sr_out: int, length_out: int) -> torch.Tensor:
    """Linear resample using torch.nn.functional.interpolate.

    x: (..., L) any leading dims.
    Output last-dim = length_out.
    """
    if sr_in == sr_out:
        return x
    orig_shape = x.shape
    flat = x.reshape(-1, 1, orig_shape[-1])
    out = F.interpolate(flat, size=length_out, mode="linear", align_corners=False)
    return out.reshape(*orig_shape[:-1], length_out)


def radar_to_stft_input(radar_1ch_200hz: torch.Tensor) -> torch.Tensor:
    """Convert (B, 1, 1600) radar at 200 Hz → (B, 2, 64, T_stft) STFT input.

    radar_1ch_200hz must already be the *single-bin* RCG (caller selects bin upstream
    via MMECGWindowedH5Dataset(topk_bins=1)).
    """
    assert radar_1ch_200hz.dim() == 3 and radar_1ch_200hz.shape[1] == 1, \
        f"expected (B,1,1600), got {tuple(radar_1ch_200hz.shape)}"
    # 1. Resample 200 → 100 Hz
    radar_100hz = _resample(radar_1ch_200hz, SR_NATIVE, SR_TARGET, WIN_LEN_TARGET)  # (B, 1, 800)

    # 2. STFT (per-window Hann)
    window = torch.hann_window(STFT_WIN, device=radar_100hz.device, dtype=radar_100hz.dtype)
    radar_flat = radar_100hz.squeeze(1)                                              # (B, 800)
    spec = torch.stft(
        radar_flat,
        n_fft=STFT_N_FFT,
        hop_length=STFT_HOP,
        win_length=STFT_WIN,
        window=window,
        return_complex=True,
        center=True,
    )                                                                                 # (B, 129, T_stft)
    spec = spec[:, :KEEP_FREQ_BINS, :]                                                # (B, 64, T_stft)
    inp = torch.stack([spec.real, spec.imag], dim=1)                                  # (B, 2, 64, T_stft)
    return inp


def stft_output_to_ecg(stft_real_imag: torch.Tensor) -> torch.Tensor:
    """Convert (B, 2, 64, T_stft) STFT output → (B, 1, 1600) ECG at 200 Hz.

    Steps:
      1. zero-pad freq bins back to (n_fft//2 + 1) = 129
      2. ISTFT (length=800) → (B, 800)
      3. resample 100 → 200 Hz → (B, 1, 1600)
    """
    assert stft_real_imag.dim() == 4 and stft_real_imag.shape[1] == 2, \
        f"expected (B,2,64,T), got {tuple(stft_real_imag.shape)}"
    B, _, F_kept, T = stft_real_imag.shape
    assert F_kept == KEEP_FREQ_BINS, f"expected freq={KEEP_FREQ_BINS}, got {F_kept}"

    full_freq = STFT_N_FFT // 2 + 1   # 129
    full = torch.zeros(B, full_freq, T, dtype=torch.complex64, device=stft_real_imag.device)
    full[:, :KEEP_FREQ_BINS, :] = stft_real_imag[:, 0] + 1j * stft_real_imag[:, 1]

    window = torch.hann_window(STFT_WIN, device=stft_real_imag.device, dtype=torch.float32)
    ecg_100hz = torch.istft(
        full, n_fft=STFT_N_FFT, hop_length=STFT_HOP, win_length=STFT_WIN,
        window=window, length=WIN_LEN_TARGET, center=True,
    )                                                                                 # (B, 800)
    ecg_100hz = ecg_100hz.unsqueeze(1)                                                # (B, 1, 800)
    ecg_200hz = _resample(ecg_100hz, SR_TARGET, SR_NATIVE, WIN_LEN_NATIVE)            # (B, 1, 1600)
    return ecg_200hz


def ecg_gt_to_stft(ecg_gt_200hz: torch.Tensor) -> torch.Tensor:
    """Convert GT ECG (B, 1, 1600) @ 200 Hz → STFT (B, 2, 64, T_stft) for spectral loss.

    Note: training-time L1 loss is computed in time domain at 100 Hz (matching paper);
    spectral loss is computed on STFT magnitudes. This util produces the STFT target
    if needed; we mostly compute losses in time domain after ISTFT on the prediction.
    """
    ecg_100hz = _resample(ecg_gt_200hz, SR_NATIVE, SR_TARGET, WIN_LEN_TARGET)         # (B, 1, 800)
    window = torch.hann_window(STFT_WIN, device=ecg_100hz.device, dtype=ecg_100hz.dtype)
    spec = torch.stft(
        ecg_100hz.squeeze(1),
        n_fft=STFT_N_FFT, hop_length=STFT_HOP, win_length=STFT_WIN,
        window=window, return_complex=True, center=True,
    )
    spec = spec[:, :KEEP_FREQ_BINS, :]
    return torch.stack([spec.real, spec.imag], dim=1)


def ecg_gt_to_100hz(ecg_gt_200hz: torch.Tensor) -> torch.Tensor:
    """GT ECG (B, 1, 1600) @ 200 Hz → (B, 1, 800) @ 100 Hz, for time-domain L1 loss."""
    return _resample(ecg_gt_200hz, SR_NATIVE, SR_TARGET, WIN_LEN_TARGET)


def compute_t_stft(win_len_target: int = WIN_LEN_TARGET) -> int:
    """Compute the T_stft frame count for a given target window length (sanity)."""
    # torch.stft with center=True: T = win_len // hop + 1
    return win_len_target // STFT_HOP + 1
