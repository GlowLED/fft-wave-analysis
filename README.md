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