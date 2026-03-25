# pyshiro

[SHIRO](https://github.com/Sleepwalking/SHIRO)（Hidden Semi-Markov Model ベースの音素強制アライメントツール）の Python 再実装です。日本語歌声を主なターゲットとしています。

英語版: [README.en.md](README.en.md)

## 特徴

- **強制アライメント**: HSMM Viterbi デコーディングによる音素と音声の位置合わせ
- **2パスアライメント**: HMM でブートストラップ → HSMM で精細化（オリジナル SHIRO と同方式）
- **モデル訓練**: ラベル付きコーパスから `.hsmm` モデルを学習
- **ラベル出力**: HTK `.lab`（ENUNU/UTAU 互換）および Praat TextGrid
- **かな→音素変換**: ひらがな歌詞 → 音素列変換（ENUNU 互換）
- **軽量**: コア部分の依存は `numpy` / `scipy` / `soundfile` / `numba` / `msgpack` のみ

## インストール

```bash
git clone --recurse-submodules https://github.com/your-username/pyshiro
cd pyshiro
pip install -e .
```

> `--recurse-submodules` を付けると `models/` 以下の訓練済みモデルも同時に取得できます。

## クイックスタート

```python
import pyshiro

# 訓練済みモデルを読み込む
model    = pyshiro.load_hsmm("models/intunist-jp6_generic.hsmm")
phonemap = pyshiro.load_phonemap("models/intunist-jp6_phonemap.json")

# 特徴量を抽出
streams  = pyshiro.extract_mfcc_from_file("audio.wav")

# 歌詞を音素列に変換
table    = pyshiro.load_table()   # 同梱の kana2phonemes テーブルを使用
phonemes = pyshiro.convert_lyric_file("lyrics.txt", table)

# アライメント
T         = streams[0].shape[0]
state_seq = pyshiro.build_state_sequence(phonemes, phonemap, T)
segments  = pyshiro.forced_align_2pass(model, streams, state_seq)

# .lab ファイルに書き出す
from pyshiro.labels import segments_to_phoneme_intervals, write_lab
intervals = segments_to_phoneme_intervals(phonemes, segments)
write_lab(intervals, "output.lab")
```

## 歌詞ファイルのフォーマット

歌詞 `.txt` はひらがなで1フレーズ1行で記述します。改行が `pau`（ポーズ）になります。

```
きっと
とべば
そらまでとどく
```

吐息（`br`）や無音（`pau`）は直接記述することもできます：

```
pau
きっと
br
とべば
```

## CLI

```bash
# アライメント
pyshiro-align audio.wav lyrics.txt \
  --model models/intunist-jp6_generic.hsmm \
  --phonemap models/intunist-jp6_phonemap.json \
  --out output.lab

# 訓練
pyshiro-train \
  --wav_dir  corpus/wav \
  --lab_dir  corpus/lab \
  --phonemap models/intunist-jp6_phonemap.json \
  --out      my_model.hsmm \
  --iters    5
```

## 訓練済みモデル

[intunist/SHIRO-Models-Japanese](https://github.com/intunist/SHIRO-Models-Japanese) を git submodule として同梱しています。男女を含む日本語 17.8 時間のデータセットで訓練されたモデルです。

## 動作環境

- Python 3.10 以上
- numpy, scipy, soundfile, numba, msgpack

## ライセンス

GPLv3 — オリジナルの [SHIRO](https://github.com/Sleepwalking/SHIRO) および [liblrhsmm](https://github.com/Sleepwalking/liblrhsmm) に準拠します。
