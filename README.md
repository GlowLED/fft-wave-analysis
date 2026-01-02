# 傅立叶变换音频分析器

按照论文 `paper.typ` 中的流程，实现了离线音频频谱分析：

- 从 `samples/` 目录读取 `*.wav`（44100 Hz 推荐），去直流并做 RMS 归一化。
- 可选截取指定时间段、加 Hann 窗，按指定零填充倍数计算单边幅值谱。
- 使用抛物线插值细化峰值频率，对文件名中的品位数字推算理论基频并标注谐波。
- 输出线性与对数幅值谱图，保存到 `outputs/`。

## 使用方法

```bash
python main.py \
  --input-dir samples \
  --output-dir outputs \
  --start 0.0 \
  --duration 1.5 \
  --zero-padding 2 \
  --max-peaks 6 \
  --base-frequency 196
```

文件名若为数字（如 `0.wav`、`3.wav`），程序会假设 G 弦基频为 `base-frequency`，按半音公式推算理论频率用于标注。若没有样本文件，运行时会提示。

## 生成测试音频

使用 `generate_test_audio.py` 可以生成模拟吉他弦振动的测试音频文件：

```bash
python generate_test_audio.py \
  --output-dir samples \
  --base-frequency 196.0 \
  --frets 0 3 5 7 9 12 \
  --duration 2.0 \
  --harmonics 6
```

该程序会生成包含多个谐波的合成音频，模拟吉他弦的振动特性：
- 包含基频和多个整数倍谐波
- 使用ADSR包络模拟拨弦后的衰减
- 添加轻微的相位随机性和拨弦瞬态噪声
- 输出为16位PCM WAV格式，采样率44100 Hz

参数说明：
- `--output-dir`: 输出目录（默认: `samples`）
- `--base-frequency`: G弦空弦基频，单位Hz（默认: 196.0）
- `--frets`: 要生成的品数列表（默认: 0 3 5 7 9 12）
- `--duration`: 每个音频文件的时长，单位秒（默认: 2.0）
- `--sample-rate`: 采样率，单位Hz（默认: 44100）
- `--harmonics`: 谐波数量（默认: 6）