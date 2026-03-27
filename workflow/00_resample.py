"""
workflow/00_resample.py

WAV ファイルを 16kHz モノラルに変換する。
pyshiro の特徴量抽出は 16kHz モノラルを前提としているため、
学習・アライメントの前にこのスクリプトで変換しておく。

使い方:
  python workflow/00_resample.py \
      --in_dir  example/raw_wav \
      --out_dir example/wav_16k

オプション:
  --target_sr  変換後のサンプリングレート（デフォルト: 16000）
  --pattern    対象ファイルのパターン（デフォルト: *.wav）
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd


def resample_to_mono(wav: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """多チャンネル / 任意サンプリングレートの音声を target_sr のモノラルに変換する。"""
    # モノラル化
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    wav = wav.astype(np.float32)

    if sr == target_sr:
        return wav

    # 最小公約数で up/down を約分して精度よくリサンプル
    g = gcd(target_sr, sr)
    up, down = target_sr // g, sr // g
    resampled = resample_poly(wav, up, down).astype(np.float32)
    return resampled


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--in_dir",    type=Path, required=True, help="入力 WAV ディレクトリ")
    parser.add_argument("--out_dir",   type=Path, required=True, help="出力ディレクトリ")
    parser.add_argument("--target_sr", type=int,  default=16000, help="目標サンプリングレート")
    parser.add_argument("--pattern",   type=str,  default="*.wav")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(args.in_dir.glob(args.pattern))
    if not wav_files:
        print(f"WAV ファイルが見つかりません: {args.in_dir}/{args.pattern}")
        return

    print(f"{len(wav_files)} ファイルを変換中 ({args.target_sr} Hz モノラル) ...")
    for path in wav_files:
        wav, sr = sf.read(path, dtype="float32", always_2d=False)
        out = resample_to_mono(wav, sr, args.target_sr)
        out_path = args.out_dir / path.name
        sf.write(out_path, out, args.target_sr, subtype="PCM_16")
        ch = "stereo→mono" if wav.ndim > 1 else "mono"
        print(f"  {path.name}  {sr} Hz {ch} → {args.target_sr} Hz  [{len(out)/args.target_sr:.1f}s]")

    print(f"\n完了。出力先: {args.out_dir}")


if __name__ == "__main__":
    main()
