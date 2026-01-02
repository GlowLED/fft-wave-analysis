"""
生成测试用的吉他弦音频文件

根据论文中的描述，生成模拟吉他弦振动的音频文件。
包含基频和多个谐波，并模拟拨弦后的衰减特性。
"""

from __future__ import annotations

import argparse
import wave
from pathlib import Path

import numpy as np


def generate_guitar_string_audio(
    fundamental_freq: float,
    duration: float = 2.0,
    sample_rate: int = 44100,
    num_harmonics: int = 6,
    attack_time: float = 0.01,
    decay_time: float = 0.1,
    sustain_level: float = 0.3,
    release_time: float = 1.5,
    noise_level: float = 0.02,
) -> np.ndarray:
    """
    生成模拟吉他弦振动的音频信号
    
    参数:
        fundamental_freq: 基频 (Hz)
        duration: 音频时长 (秒)
        sample_rate: 采样率 (Hz)
        num_harmonics: 谐波数量
        attack_time: 起音时间 (秒)
        decay_time: 衰减时间 (秒)
        sustain_level: 持续电平 (0-1)
        release_time: 释放时间 (秒)
        noise_level: 噪声水平 (用于模拟拨弦瞬态)
    
    返回:
        归一化的音频信号数组
    """
    n_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    
    # 生成谐波：基频的整数倍
    signal = np.zeros(n_samples, dtype=np.float64)
    
    # 谐波相对幅度（模拟真实吉他的频谱特性）
    # 高次谐波逐渐衰减，但某些谐波可能较强
    harmonic_amplitudes = [
        1.0,      # 基频
        0.6,      # 2次谐波
        0.4,      # 3次谐波
        0.25,     # 4次谐波
        0.15,     # 5次谐波
        0.1,      # 6次谐波
    ]
    
    # 添加一些相位随机性，使声音更自然
    phases = np.random.uniform(0, 2 * np.pi, num_harmonics)
    
    for n in range(1, num_harmonics + 1):
        freq = fundamental_freq * n
        if freq >= sample_rate / 2:  # 避免超过Nyquist频率
            break
        
        amplitude = harmonic_amplitudes[n - 1] if n <= len(harmonic_amplitudes) else 1.0 / n
        phase = phases[n - 1]
        
        # 生成精确的谐波频率（整数倍）
        signal += amplitude * np.sin(2 * np.pi * freq * t + phase)
    
    # 生成ADSR包络（Attack-Decay-Sustain-Release）
    envelope = np.ones(n_samples)
    
    attack_samples = int(attack_time * sample_rate)
    decay_samples = int(decay_time * sample_rate)
    release_samples = int(release_time * sample_rate)
    sustain_samples = n_samples - attack_samples - decay_samples - release_samples
    
    if attack_samples > 0:
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    
    if decay_samples > 0:
        decay_start = attack_samples
        decay_end = decay_start + decay_samples
        envelope[decay_start:decay_end] = np.linspace(1, sustain_level, decay_samples)
    
    if sustain_samples > 0:
        sustain_start = attack_samples + decay_samples
        sustain_end = sustain_start + sustain_samples
        # 持续段也有轻微衰减
        sustain_decay = np.exp(-t[sustain_start:sustain_end] / (duration * 2))
        envelope[sustain_start:sustain_end] = sustain_level * sustain_decay
    
    if release_samples > 0:
        release_start = n_samples - release_samples
        envelope[release_start:] = np.linspace(
            envelope[release_start - 1] if release_start > 0 else sustain_level,
            0,
            release_samples
        )
    
    # 应用包络
    signal *= envelope
    
    # 添加拨弦瞬态噪声（模拟拨弦瞬间的高频成分）
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, n_samples)
        # 噪声只在开始部分
        noise_envelope = np.exp(-t / 0.05)
        signal += noise * noise_envelope
    
    # 归一化到合理范围，避免削波
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal = signal / max_val * 0.8  # 留一些余量
    
    return signal.astype(np.float32)


def save_wav_file(signal: np.ndarray, output_path: Path, sample_rate: int = 44100) -> None:
    """保存音频信号为WAV文件"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 转换为16位PCM格式
    signal_int16 = (signal * 32767).astype(np.int16)
    
    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(1)  # 单声道
        wf.setsampwidth(2)  # 16位 = 2字节
        wf.setframerate(sample_rate)
        wf.writeframes(signal_int16.tobytes())
    
    print(f"已保存: {output_path}")


def fret_to_frequency(base_freq: float, fret: int) -> float:
    """
    根据品数计算频率
    
    参数:
        base_freq: 空弦基频 (Hz)
        fret: 品数 (0为空弦)
    
    返回:
        对应频率 (Hz)
    """
    return base_freq * (2 ** (fret / 12))


def main() -> None:
    parser = argparse.ArgumentParser(description="生成测试用的吉他弦音频文件")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("samples"),
        help="输出目录（默认: samples）",
    )
    parser.add_argument(
        "--base-frequency",
        type=float,
        default=196.0,
        help="G弦（3弦）空弦基频 (Hz，默认: 196.0)",
    )
    parser.add_argument(
        "--frets",
        type=int,
        nargs="+",
        default=[0, 3, 5, 7, 9, 12],
        help="要生成的品数列表（默认: 0 3 5 7 9 12）",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=2.0,
        help="每个音频文件的时长（秒，默认: 2.0）",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=44100,
        help="采样率 (Hz，默认: 44100)",
    )
    parser.add_argument(
        "--harmonics",
        type=int,
        default=6,
        help="谐波数量（默认: 6）",
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"生成测试音频文件到: {output_dir}")
    print(f"基频: {args.base_frequency} Hz (G弦空弦)")
    print(f"采样率: {args.sample_rate} Hz")
    print(f"时长: {args.duration} 秒")
    print()
    
    for fret in args.frets:
        freq = fret_to_frequency(args.base_frequency, fret)
        print(f"生成 {fret}品 (频率: {freq:.2f} Hz)...")
        
        audio = generate_guitar_string_audio(
            fundamental_freq=freq,
            duration=args.duration,
            sample_rate=args.sample_rate,
            num_harmonics=args.harmonics,
        )
        
        output_path = output_dir / f"{fret}.wav"
        save_wav_file(audio, output_path, args.sample_rate)
    
    print()
    print(f"完成！已生成 {len(args.frets)} 个音频文件。")


if __name__ == "__main__":
    main()

