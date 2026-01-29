# role_refinement.py

import numpy as np
import scipy.signal as signal


def to_mono(x):
    if x.ndim > 1:
        return np.mean(x, axis=1)
    return x


def normalize(x):
    return x / (np.max(np.abs(x)) + 1e-9)


def smooth_mask(mask, kernel_size=5):
    kernel = np.ones(kernel_size) / kernel_size
    for row in range(mask.shape[0]):
        mask[row, :] = np.convolve(mask[row, :], kernel, mode="same")
    return mask


def build_harmonic_mask(ref_stft, freqs,
                        fmin=80,
                        fmax=4000,
                        bandwidth_hz=100,
                        max_harmonics=4):

    mag = np.abs(ref_stft)
    mask = np.zeros_like(ref_stft, dtype=float)
    freq_res = freqs[1] - freqs[0]

    for t in range(mag.shape[1]):
        frame = mag[:, t]
        valid = np.where((freqs > fmin) & (freqs < fmax))[0]

        if len(valid) == 0:
            continue

        peak_idx = valid[np.argmax(frame[valid])]
        peak_freq = freqs[peak_idx]

        for h in range(1, max_harmonics + 1):
            target = peak_freq * h
            idx = np.argmin(np.abs(freqs - target))
            bw = int(bandwidth_hz / freq_res)
            mask[max(0, idx - bw):min(len(freqs), idx + bw), t] = 1.0

    return mask


def role_guided_refine(original_audio,
                       guide_audio,
                       sr,
                       nperseg=2048):

    original_audio = to_mono(original_audio)
    guide_audio = to_mono(guide_audio)

    min_len = min(len(original_audio), len(guide_audio))
    original_audio = original_audio[:min_len]
    guide_audio = guide_audio[:min_len]

    noverlap = nperseg // 2

    freqs, _, Z_orig = signal.stft(original_audio, sr,
                                   nperseg=nperseg,
                                   noverlap=noverlap)

    _, _, Z_guide = signal.stft(guide_audio, sr,
                                nperseg=nperseg,
                                noverlap=noverlap)

    mask = build_harmonic_mask(Z_guide, freqs)
    mask = smooth_mask(mask, kernel_size=5)

    mask = 0.9 * mask + 0.1

    refined_spec = Z_orig * mask
    _, refined_audio = signal.istft(refined_spec, sr,
                                    nperseg=nperseg,
                                    noverlap=noverlap)

    return normalize(refined_audio)
