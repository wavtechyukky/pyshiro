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

無音（`pau`）は直接記述することもできます：

```
pau
きっと
pau
とべば
```

> **注意**: 吐息（`br`）は変換テーブル上は定義されていますが、同梱の訓練済みモデルが `br` 音素に未対応のため現時点では使用できません。`br` 対応モデルの学習は TODO として予定しています。

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

## TODO

オリジナル SHIRO との機能差を埋めるべく、以下の実装を予定しています。

- [ ] **スキップ可能音素**（`pskip`）: phonemap に確率を指定することで、pau/br などを実際に存在しない場合にスキップできる
- [ ] **音素内トポロジー選択**（type-b / type-c / skip-boundary）: 状態間のスキップ遷移で短い音素を柔軟に扱う
- [ ] **HMM ブートストラップ訓練**（`shiro-rest -g` 相当）: HSMM 訓練の前段として HMM で事前学習し、収束を改善する
- [ ] **DAEM 訓練**（`shiro-rest -D` 相当）: 温度係数付き EM によりフラットスタートからの精度を向上させる
- [ ] **`shiro-untie` 相当**: モノフォンモデルをトライフォンに展開するツール
- [ ] **`br`（吐息）音素対応**: 訓練データに br ラベルを含めた専用モデルの学習と同梱
- [ ] **GMM 混合数の増加**: 現在 nmix=1 固定。複数混合成分による表現力向上
- [ ] **Audacity ラベル入出力**: `.lab` ↔ Audacity テキストファイル変換

## 謝辞

- **[Sleepwalking (Kanru Hua)](https://github.com/Sleepwalking)** — オリジナル [SHIRO](https://github.com/Sleepwalking/SHIRO) および [liblrhsmm](https://github.com/Sleepwalking/liblrhsmm) の設計・実装。本プロジェクトのアルゴリズムと `.hsmm` フォーマットはこれらに基づいています。
- **[intunist](https://github.com/intunist)** — 日本語 17.8 時間コーパスで訓練した [SHIRO-Models-Japanese](https://github.com/intunist/SHIRO-Models-Japanese) の公開。同梱の訓練済みモデルはこのリポジトリのものです。

## ライセンス

GPLv3 — オリジナルの [SHIRO](https://github.com/Sleepwalking/SHIRO) および [liblrhsmm](https://github.com/Sleepwalking/liblrhsmm) に準拠します。
