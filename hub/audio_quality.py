"""Numeric input diagnostics only; does not store audio or infer SNR."""
import math

import numpy as np


def pcm_stats(pcm: bytes, sample_rate: int) -> dict:
    samples = np.frombuffer(pcm[:len(pcm) - len(pcm) % 2], dtype='<i2').astype(np.float32) / 32768
    rms = float(np.sqrt(np.mean(samples * samples))) if samples.size else 0.0
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    return {
        'seconds': round(samples.size / sample_rate, 3),
        'rms_dbfs': round(20 * math.log10(max(rms, 1e-8)), 1),
        'peak_dbfs': round(20 * math.log10(max(peak, 1e-8)), 1),
        'clipped_percent': round(float(np.mean(np.abs(samples) >= .999)) * 100, 3) if samples.size else 0.0,
    }
