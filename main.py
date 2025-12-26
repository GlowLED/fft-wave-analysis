from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.analyzer import AnalysisResult, Analyzer, Peak


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Guitar string FFT analyzer")
    parser.add_argument("--input-dir", type=Path, default=Path("samples"), help="Directory containing wav files")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Where to write plots")
    parser.add_argument("--start", type=float, default=0.0, help="Segment start time in seconds")
    parser.add_argument("--duration", type=float, default=None, help="Optional segment duration in seconds")
    parser.add_argument(
        "--zero-padding", type=int, default=2, help="Zero padding factor (1=no padding, 2=double length, ...)"
    )
    parser.add_argument("--max-peaks", type=int, default=6, help="How many peaks to annotate")
    parser.add_argument("--base-frequency", type=float, default=196.0, help="Open G string frequency in Hz")
    parser.add_argument("--max-plot-frequency", type=float, default=4000.0, help="Max frequency shown on plots")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(input_dir.glob("*.wav"))
    if not wav_files:
        print(f"No wav files found in {input_dir}. Place files there before running.")
        return

    analyzer = Analyzer(zero_padding_factor=args.zero_padding, max_peaks=args.max_peaks)

    for wav_path in wav_files:
        fundamental_hint = infer_fundamental_from_name(wav_path, args.base_frequency)
        result = analyzer.analyze_file(
            wav_path,
            start=args.start,
            duration=args.duration,
            fundamental_hint=fundamental_hint,
        )
        harmonic_lines = build_harmonic_lines(fundamental_hint, result.sample_rate)
        plot_path = plot_result(result, output_dir, harmonic_lines, args.max_plot_frequency)
        describe_result(result, wav_path, fundamental_hint, plot_path)


def infer_fundamental_from_name(path: Path, base_frequency: float) -> float | None:
    try:
        fret = int(path.stem)
    except ValueError:
        return None
    return base_frequency * (2 ** (fret / 12))


def build_harmonic_lines(fundamental: float | None, sample_rate: int) -> List[float]:
    if fundamental is None or fundamental <= 0:
        return []
    nyquist = sample_rate / 2
    lines: List[float] = []
    harmonic = fundamental
    order = 1
    while harmonic < nyquist:
        lines.append(harmonic)
        order += 1
        harmonic = fundamental * order
    return lines


def plot_result(
    result: AnalysisResult, output_dir: Path, harmonic_lines: Iterable[float], max_freq: float
) -> Path:
    sr = result.sample_rate
    time_raw = np.arange(result.raw.size) / sr
    time_seg = result.segment_start + (np.arange(result.segment.size) / sr)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), constrained_layout=True)

    axes[0].plot(time_raw, result.raw, color="0.6", linewidth=0.8, label="preprocessed")
    axes[0].plot(time_seg, result.segment, color="C0", linewidth=0.9, label="selected segment")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Amplitude (RMS=1)")
    axes[0].legend(loc="upper right")

    freqs = result.freqs
    amplitudes = result.amplitude
    axes[1].plot(freqs, amplitudes, color="C0", linewidth=0.9)
    _annotate_peaks(axes[1], result.peaks)
    _draw_harmonics(axes[1], harmonic_lines)
    axes[1].set_xlim(0, max_freq)
    axes[1].set_xlabel("Frequency [Hz]")
    axes[1].set_ylabel("Amplitude (single-sided)")
    axes[1].set_title("Linear spectrum")

    amplitudes_db = 20 * np.log10(np.clip(amplitudes, 1e-12, None))
    axes[2].plot(freqs, amplitudes_db, color="C1", linewidth=0.9)
    _annotate_peaks(axes[2], result.peaks)
    _draw_harmonics(axes[2], harmonic_lines)
    axes[2].set_xlim(0, max_freq)
    axes[2].set_xlabel("Frequency [Hz]")
    axes[2].set_ylabel("Magnitude [dB]")
    axes[2].set_title("Log magnitude spectrum")

    output_path = output_dir / f"{result.path.stem}.png"
    fig.suptitle(result.path.name)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def _annotate_peaks(axis, peaks: Iterable[Peak]) -> None:
    for peak in peaks:
        axis.scatter(peak.frequency, peak.amplitude, color="red", s=16)
        label = peak.label or "peak"
        axis.annotate(
            f"{label}\n{peak.frequency:.1f} Hz",
            xy=(peak.frequency, peak.amplitude),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            color="red",
        )


def _draw_harmonics(axis, harmonic_lines: Iterable[float]) -> None:
    for f in harmonic_lines:
        axis.axvline(f, color="0.6", linestyle="--", linewidth=0.8)
        axis.text(f, axis.get_ylim()[1] * 0.9, f"{f:.0f} Hz", rotation=90, fontsize=7, color="0.4")


def describe_result(result: AnalysisResult, path: Path, fundamental_hint: float | None, plot_path: Path) -> None:
    top_peak = max(result.peaks, key=lambda p: p.amplitude, default=None)
    print(f"\nFile: {path.name}")
    if fundamental_hint:
        print(f"  Fundamental hint: {fundamental_hint:.2f} Hz")
    if top_peak:
        print(f"  Dominant peak: {top_peak.frequency:.2f} Hz, amplitude={top_peak.amplitude:.3f}")
    print(f"  Saved plot to: {plot_path}")


if __name__ == "__main__":
    main()
