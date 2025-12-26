from __future__ import annotations

import math
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np


@dataclass
class Peak:
    frequency: float
    amplitude: float
    bin_index: int
    delta: float
    label: str | None = None


@dataclass
class AnalysisResult:
    path: Path
    sample_rate: int
    raw: np.ndarray
    segment: np.ndarray
    segment_start: float
    windowed: np.ndarray
    window: np.ndarray
    freqs: np.ndarray
    amplitude: np.ndarray
    peaks: List[Peak]
    fundamental_hint: float | None
    n_fft: int


def _next_pow_two(n: int) -> int:
    return 1 if n <= 1 else 2 ** math.ceil(math.log2(n))


class Analyzer:
    """Signal analysis pipeline described in the paper."""

    def __init__(self, zero_padding_factor: int = 2, max_peaks: int = 6):
        self.zero_padding_factor = max(1, int(zero_padding_factor))
        self.max_peaks = max(1, int(max_peaks))

    def analyze_file(
        self,
        path: Path,
        start: float = 0.0,
        duration: float | None = None,
        fundamental_hint: float | None = None,
    ) -> AnalysisResult:
        sample_rate, raw = self._read_wave(path)
        processed = self._preprocess(raw)
        segment = self._select_segment(processed, sample_rate, start=start, duration=duration)
        if segment.size == 0:
            raise ValueError(f"Selected segment is empty for {path}")

        windowed, window = self._apply_window(segment)
        freqs, amplitude, n_fft = self._compute_fft(windowed, sample_rate, window)
        peaks = self._pick_peaks(amplitude, sample_rate, n_fft)
        self._label_fundamental(peaks, fundamental_hint)

        return AnalysisResult(
            path=Path(path),
            sample_rate=sample_rate,
            raw=processed,
            segment=segment,
            segment_start=float(start),
            windowed=windowed,
            window=window,
            freqs=freqs,
            amplitude=amplitude,
            peaks=peaks,
            fundamental_hint=fundamental_hint,
            n_fft=n_fft,
        )

    def _read_wave(self, path: Path) -> Tuple[int, np.ndarray]:
        path = Path(path)
        with wave.open(path.as_posix(), "rb") as wf:
            sample_rate = wf.getframerate()
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            frames = wf.readframes(wf.getnframes())

        data = self._pcm_to_array(frames, sampwidth)
        if channels > 1:
            data = data.reshape(-1, channels).mean(axis=1)
        return sample_rate, data.astype(np.float64)

    def _pcm_to_array(self, frames: bytes, sampwidth: int) -> np.ndarray:
        if sampwidth == 1:
            data = np.frombuffer(frames, dtype=np.uint8).astype(np.int16) - 128
            scale = 128.0
        elif sampwidth == 2:
            data = np.frombuffer(frames, dtype="<i2")
            scale = float(1 << 15)
        elif sampwidth == 3:
            as_uint = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3)
            data = (
                as_uint[:, 0].astype(np.int32)
                | (as_uint[:, 1].astype(np.int32) << 8)
                | (as_uint[:, 2].astype(np.int32) << 16)
            )
            sign_mask = 1 << 23
            data = (data ^ sign_mask) - sign_mask
            scale = float(1 << 23)
        elif sampwidth == 4:
            data = np.frombuffer(frames, dtype="<i4")
            scale = float(1 << 31)
        else:
            raise ValueError(f"Unsupported sample width: {sampwidth} bytes")

        return data.astype(np.float64) / scale

    def _preprocess(self, signal: np.ndarray) -> np.ndarray:
        centered = signal - np.mean(signal)
        rms = np.sqrt(np.mean(centered ** 2))
        if rms > 0:
            centered = centered / rms
        return centered

    def _select_segment(
        self, signal: np.ndarray, sample_rate: int, start: float = 0.0, duration: float | None = None
    ) -> np.ndarray:
        start_idx = max(0, int(start * sample_rate))
        if duration is not None:
            end_idx = start_idx + int(duration * sample_rate)
            return signal[start_idx:end_idx]
        return signal[start_idx:]

    def _apply_window(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        window = np.hanning(len(signal))
        return signal * window, window

    def _compute_fft(self, signal: np.ndarray, sample_rate: int, window: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        n_samples = len(signal)
        n_fft = _next_pow_two(n_samples) * self.zero_padding_factor

        spectrum = np.fft.rfft(signal, n=n_fft)
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)

        window_gain = float(window.mean())
        amplitude = np.abs(spectrum)
        amplitude = amplitude / max(1e-12, n_samples * window_gain)
        if amplitude.size > 1:
            amplitude[1:-1] *= 2.0
        return freqs, amplitude, n_fft

    def _pick_peaks(self, amplitude: np.ndarray, sample_rate: int, n_fft: int) -> List[Peak]:
        if amplitude.size == 0:
            return []

        candidates = amplitude.copy()
        candidates[0] = 0.0
        top_indices = np.argpartition(candidates, -self.max_peaks)[-self.max_peaks :]
        top_indices = top_indices[np.argsort(candidates[top_indices])[::-1]]

        peaks: List[Peak] = []
        for idx in top_indices:
            frequency, refined_amp, delta = self._parabolic_interp(amplitude, idx, sample_rate, n_fft)
            peaks.append(Peak(frequency=frequency, amplitude=refined_amp, bin_index=int(idx), delta=float(delta)))
        return peaks

    def _parabolic_interp(
        self, amplitude: np.ndarray, index: int, sample_rate: int, n_fft: int
    ) -> Tuple[float, float, float]:
        if index <= 0 or index >= len(amplitude) - 1:
            freq = index * sample_rate / n_fft
            return freq, float(amplitude[index]), 0.0

        m1, m0, p1 = amplitude[index - 1], amplitude[index], amplitude[index + 1]
        denom = (m1 - 2 * m0 + p1)
        delta = 0.0 if denom == 0 else 0.5 * (m1 - p1) / denom
        delta = float(np.clip(delta, -1.0, 1.0))
        refined_amp = m0 - 0.25 * (m1 - p1) * delta
        freq = (index + delta) * sample_rate / n_fft
        return float(freq), float(refined_amp), delta

    def _label_fundamental(self, peaks: List[Peak], fundamental_hint: float | None) -> None:
        if not peaks:
            return
        if fundamental_hint is None:
            peaks[0].label = "fundamental?"
            return

        target = min(peaks, key=lambda p: abs(p.frequency - fundamental_hint))
        target.label = "fundamental"