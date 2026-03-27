# pyshiro

[SHIRO](https://github.com/Sleepwalking/SHIRO)（Hidden Semi-Markov Model ベースの音素強制アライメントツール）のPython実装です。日本語の歌声を主なターゲットとしています。

英語版: [README.en.md](README.en.md)

## 特徴

オリジナル [SHIRO](https://github.com/Sleepwalking/SHIRO) の機能をほぼ完全にPythonへ移植し、さらに独自機能を追加しています。特に日本語での学習・アノテーションをスムーズに行えるよう、専用のインターフェースを整備しています。

**アライメント**
- **HSMM 強制アライメント**: 音素境界を自動推定
- **HMM→HSMM の2段階アライメント**: まず持続長を考慮しない HMM で粗いアライメントを行い、次に HSMM で精細化（オリジナル SHIRO と同方式）
- **スキップ可能音素の指定**（`pskip`）: pau / br などを phonemap で省略可能に指定できる
- **topology の設定**: 音素ごとの状態遷移パターンを phonemap で指定できる

**訓練**
- **コーパスからの学習**: ラベル付き音声コーパスから `.hsmm` モデルを学習
- **HMM プレトレーニング**（`--hmm_iters`）: 持続長モデルを無効にした HMM モードで先にモデルを初期化してから HSMM 学習に移行（`shiro-rest -g` 相当）
- **DAEM**（`--daem`）: アニーリングで学習を安定化（`shiro-rest -D` 相当）
- **GMM 成分数の増加**（`--nmix`）: 段階的な GMM 分割で表現力を向上
- **トライフォン化**（`pyshiro.untie`）: モノフォンモデルをコンテキスト依存モデルへ変換（`shiro-untie` 相当）

**入出力**
- **ラベル出力**: `.lab`（ENUNU 互換）、Praat TextGrid、Audacity ラベル
- **ラベル読み込み**: `.lab`、Audacity ラベル
- **かな→音素変換**: ひらがな歌詞 → 音素列変換（ENUNU の変換テーブルをベースに同梱）

## インストール

```bash
git clone --recurse-submodules https://github.com/wavtechyukky/pyshiro
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
行中に `br`（吐息）などの音素を直接埋め込むこともできます。

```
brねがいわいちどbrはなしたら
brくずれてく
brしゅんかんむかいかぜbrだれもしらないbrあしあとうぉ
br
```

`pau` も直接記述できます：

```
pau
きっと
pau
とべば
```

音素をスペース区切りで直書きすることもできます（ASCII 英字のみのファイルは自動的に音素列として解釈されます）：

```
pau k i cl t o pau t o b e b a pau
```

## CLI

```bash
# アライメント
pyshiro-align audio.wav lyrics.txt \
  --model    models/intunist-jp6_generic.hsmm \
  --phonemap models/intunist-jp6_phonemap.json \
  --out      output.lab

# 出力形式を指定（lab / textgrid / audacity）
pyshiro-align audio.wav lyrics.txt \
  --model    models/intunist-jp6_generic.hsmm \
  --phonemap models/intunist-jp6_phonemap.json \
  --format   textgrid \
  --out      output.TextGrid

# 訓練（基本）
pyshiro-train \
  --wav_dir  corpus/wav \
  --lab_dir  corpus/lab \
  --phonemap models/intunist-jp6_phonemap.json \
  --out      my_model.hsmm \
  --iters    20 \
  --jobs     8   # 並列ワーカー数（デフォルト: CPU コア数）

# 訓練（HMM プレトレーニング + DAEM + GMM 成分数増加）
pyshiro-train \
  --wav_dir   corpus/wav \
  --lab_dir   corpus/lab \
  --phonemap  models/intunist-jp6_phonemap.json \
  --out       my_model.hsmm \
  --iters     20 \
  --hmm_iters 3 \
  --daem \
  --nmix      4

# 途中から再開
# イテレーション完了ごとに my_model.iter1.hsmm, my_model.iter2.hsmm ... が自動保存される
pyshiro-train \
  --wav_dir    corpus/wav \
  --lab_dir    corpus/lab \
  --phonemap   models/intunist-jp6_phonemap.json \
  --out        my_model.hsmm \
  --iters      20 \
  --init_model my_model.iter10.hsmm \
  --start_iter 10

# cap_relax_iter: 訓練初期は探索範囲を制限し、後半で解除する
# ロングトーンや長い pau を含むコーパスで収束が不安定な場合に有効
pyshiro-train \
  --wav_dir        corpus/wav \
  --lab_dir        corpus/lab \
  --phonemap       models/intunist-jp6_phonemap.json \
  --out            my_model.hsmm \
  --iters          20 \
  --hmm_iters      2 \
  --cap_relax_iter 10

# トライフォン化
python -m pyshiro.untie \
  --phonemap     models/intunist-jp6_phonemap.json \
  --model        my_model.hsmm \
  --lab_dir      corpus/lab \
  --out_phonemap my_tri_phonemap.json \
  --out_model    my_tri_model.hsmm
```

## ラベル出力

```python
from pyshiro.labels import (
    write_lab, write_textgrid, write_audacity, read_audacity
)

# 書き出し
write_lab(intervals, "output.lab")           # .lab (ENUNU互換)
write_textgrid(intervals, "output.TextGrid") # Praat TextGrid
write_audacity(intervals, "output.txt")      # Audacity ラベル

# 読み込み（手修正済みラベルを訓練データに戻す場合など）
intervals = read_audacity("corrected.txt")
```

## スキップ可能音素の設定（pskip）

phonemap の音素エントリに `"pskip"` を指定すると、その音素が実際に発音されていない場合にアライメントで省略されます。`br`（吐息）を多めに入力しても正しく処理できます。

```json
{
  "phone_map": {
    "pau": { "pskip": 0.5, "states": [...] },
    "br":  { "pskip": 0.5, "states": [...] }
  }
}
```

## topologyの設定

```json
{
  "phone_map": {
    "cl": { "topology": "type-b", "states": [...] }
  }
}
```

| topology | 動作 |
|---|---|
| `type-a`（デフォルト） | 0→1→2 の左右方向のみ |
| `type-b` | 各状態から最終状態へのスキップを追加 |
| `type-c` | 各状態から2つ先へのスキップを追加 |
| `skip-boundary` | 音素境界の先頭・末尾状態をスキップ可能にする |

## かな→音素変換テーブルについて

同梱の `pyshiro/data/kana2phonemes.table` は[ENUNU](https://github.com/oatsu-gh/ENUNU) の `kana2phonemes_etk_001.table` をベースとした音素変換テーブルです。

## 訓練済みモデル

[intunist/SHIRO-Models-Japanese](https://github.com/intunist/SHIRO-Models-Japanese) を git submodule として同梱しています。男女を含む日本語17.8時間の歌唱音声のデータセットで訓練されたモデルです。

## 動作環境

- Python 3.10 以上
- numpy, scipy, soundfile, numba, msgpack

## TODO

- [ ] **`br`（吐息）音素対応モデルの訓練・同梱**: 訓練データに br ラベルを含めた専用モデル

## 謝辞

- **[Sleepwalking (Kanru Hua)](https://github.com/Sleepwalking)** — オリジナル [SHIRO](https://github.com/Sleepwalking/SHIRO) および [liblrhsmm](https://github.com/Sleepwalking/liblrhsmm) の設計・実装。本プロジェクトのアルゴリズムと `.hsmm` フォーマットはこれらに基づいています。
- **[intunist](https://github.com/intunist)** — [SHIRO-Models-Japanese](https://github.com/intunist/SHIRO-Models-Japanese) 同梱の訓練済みモデルはこのリポジトリのものです。
- **[oatsu-gh](https://github.com/oatsu-gh)** — [ENUNU](https://github.com/oatsu-gh/ENUNU) の開発。同梱のかな→音素変換テーブルは ENUNU の `kana2phonemes_etk_001.table` をベースとしています。

## ライセンス

GPLv3 — オリジナルの [SHIRO](https://github.com/Sleepwalking/SHIRO) および [liblrhsmm](https://github.com/Sleepwalking/liblrhsmm) に準拠します。
