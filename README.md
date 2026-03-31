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
- **ラベル出力**: `.lab`（HTK 100ns 整数形式・ENUNU / NNSVS / vLabeler 互換）、Praat TextGrid、Audacity ラベル
- **ラベル読み込み**: `.lab`（HTK 100ns・秒単位を自動判定）、Audacity ラベル
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

# 特徴量を抽出（16kHz モノラル WAV）
streams  = pyshiro.extract_mfcc_from_file("example/wav_16k/akai_kutsu.wav")

# 歌詞を音素列に変換
table    = pyshiro.load_table()   # 同梱の kana2phonemes テーブルを使用
phonemes = pyshiro.convert_lyric_file("example/lyrics/akai_kutsu.txt", table)

# アライメント
T         = streams[0].shape[0]
state_seq = pyshiro.build_state_sequence(phonemes, phonemap, T)
segments  = pyshiro.forced_align_2pass(model, streams, state_seq)

# .lab ファイルに書き出す
from pyshiro.labels import segments_to_phoneme_intervals, write_lab
intervals = segments_to_phoneme_intervals(phonemes, segments)
write_lab(intervals, "example/labels/akai_kutsu.lab")
```

## アノテーションガイド

実際にアノテーションに活用したい方に向けて、訓練済みモデルを使って WAV コーパスの `.lab` を完成させるまでのワークフローを **[workflow/annotation_guide.ipynb](workflow/annotation_guide.ipynb)** で解説しています。音声の変換・分割・自動アライメント・手動修正・結合まで、一連の手順をノートブック上でガイドします。

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
  --iters    10 \
  --jobs     8   # 並列ワーカー数（デフォルト: CPU コア数）

# 訓練（HMM プレトレーニング + DAEM + GMM 成分数増加）
pyshiro-train \
  --wav_dir   corpus/wav \
  --lab_dir   corpus/lab \
  --phonemap  models/intunist-jp6_phonemap.json \
  --out       my_model.hsmm \
  --iters     10 \
  --hmm_iters 2 \
  --daem \
  --nmix      4

# 途中から再開
# イテレーション完了ごとに my_model.iter001.hsmm, my_model.iter002.hsmm ... が自動保存される
pyshiro-train \
  --wav_dir    corpus/wav \
  --lab_dir    corpus/lab \
  --phonemap   models/intunist-jp6_phonemap.json \
  --out        my_model.hsmm \
  --iters      10 \
  --init_model my_model.iter005.hsmm \
  --start_iter 5

# cap_relax_iter: 訓練初期は探索範囲を制限し、後半で解除する
# ロングトーンや長い pau を含むコーパスで収束が不安定な場合に有効
# test_wav_dir / test_lab_dir を指定するとイテレーションごとにテスト対数尤度を記録する
# ↓ おすすめ設定
pyshiro-train \
  --wav_dir        corpus/wav \
  --lab_dir        corpus/lab \
  --test_wav_dir   corpus/test/wav \
  --test_lab_dir   corpus/test/lab \
  --phonemap       models/intunist-jp6_phonemap.json \
  --out            my_model.hsmm \
  --iters          10 \
  --hmm_iters      2 \
  --cap_relax_iter 5

# トライフォン化
python -m pyshiro.untie \
  --phonemap     models/intunist-jp6_phonemap.json \
  --model        my_model.hsmm \
  --lab_dir      corpus/lab \
  --out_phonemap my_tri_phonemap.json \
  --out_model    my_tri_model.hsmm

# アライメント結果の可視化（波形・メルスペクトログラム・GT・推定ラベルを PNG 出力）
python tests/plot_alignment.py \
  --wav_dir  corpus/wav \
  --lab_dir  corpus/lab \
  --model    my_model.hsmm \
  --phonemap my_phonemap.json \
  --out_dir  plots
```

## 訓練のヒント

実際の使用から得られた知見をまとめます。

**入力音声は短く細切れにする**
1ファイルあたり20秒以下を目安にすると安定して動作しやすい。長い pau（無音区間）が含まれると探索が破綻しやすいため、曲全体をそのまま渡すのは避け、フレーズ単位に分割してから使用することを推奨する。

**`--nmix` は 1 を推奨**
GMM の混合数を増やすと表現力は上がりますが、学習データが少ない場合はパラメータ過多になりやすい。歌声コーパス程度の規模では `--nmix 1`（デフォルト）が安定して良い結果を出す。

**`--cap_relax_iter` で序盤の探索範囲を制限する**
HMM プレトレーニング直後は音響モデルが粗く、ロングトーンや長い pau に誤って大量フレームを割り当ててしまいやすい。`--cap_relax_iter 5` を指定すると序盤の探索範囲を制限しながら安定的に収束させ、後半で制限を外して精細化できる。ほとんどのコーパスで有効なため、上記のおすすめ設定に含めている。

**過学習が早い：イテレーション数より教師データの質が重要**
train の対数尤度が改善し続けていても、テストデータの対数尤度は数イテレーションで頭打ちになることが多い。学習を長く回すより、**ラベリングの一貫性が高いコーパスを用意すること**の方がアライメント精度への寄与が大きい。異なるラベラーのデータを混在させる場合は、音素境界の基準が揃っているかを確認することを推奨する。

## ラベル出力

```python
from pyshiro.labels import (
    write_lab, write_textgrid, write_audacity,
    read_lab, read_textgrid, read_audacity,
)

# 書き出し
write_lab(intervals, "output.lab")           # .lab (HTK 100ns整数形式・ENUNU / NNSVS / vLabeler 互換)
write_textgrid(intervals, "output.TextGrid") # Praat TextGrid
write_audacity(intervals, "output.txt")      # Audacity ラベル

# 読み込み（手修正済みラベルを訓練データに戻す場合など）
intervals = read_lab("corrected.lab")        # HTK 100ns / 秒単位を自動判定
intervals = read_textgrid("corrected.TextGrid")
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

### カスタムモデルの学習データ謝辞

別途配布するカスタム訓練済みモデルは、以下の歌声データベースを使用して訓練されています。各データベースの制作者・権利者に深く感謝申し上げます。

- **御丹宮くるみ歌声データベース** — 御丹宮くるみ（[https://onikuru.info](https://onikuru.info)）
- **おふとんP歌声データベース** — DB制作：おふとんP（[https://sites.google.com/view/oftn-utagoedb](https://sites.google.com/view/oftn-utagoedb)）
- **波音リツ** — カノン（[https://www.canon-voice.com](https://www.canon-voice.com)）
- **東北きりたん歌唱データベース** — ©SSS（[https://zunko.jp/kiridev/login.php](https://zunko.jp/kiridev/login.php)）
- **No.7 歌唱データベース** — ©No.7製作委員会（[https://voiceseven.com/7dev/login.php](https://voiceseven.com/7dev/login.php)）
- **夏目悠李歌声データベース** — 歌声DB制作：アマノケイ、音声提供者：霧野蒼太（[https://ksdcm1ng.wixsite.com/njksofficial](https://ksdcm1ng.wixsite.com/njksofficial)）

## 動作環境

- Python 3.10 以上
- numpy, scipy, soundfile, numba, msgpack

## 更新履歴

- **2025-03** — アノテーションワークフロー（`workflow/`）を追加。音声の変換・分割・自動アライメント・結合までをノートブックでガイド。
- **2025-03** — 独自訓練済みモデル（`checkpoint/pyshiro-jp-v1.hsmm`）を追加。複数歌声データベース計 19.4時間で追加訓練。`br`（吐息）音素に対応。

## 謝辞

- **[Sleepwalking (Kanru Hua)](https://github.com/Sleepwalking)** — オリジナル [SHIRO](https://github.com/Sleepwalking/SHIRO) および [liblrhsmm](https://github.com/Sleepwalking/liblrhsmm) の設計・実装。本プロジェクトのアルゴリズムと `.hsmm` フォーマットはこれらに基づいています。
- **[intunist](https://github.com/intunist)** — [SHIRO-Models-Japanese](https://github.com/intunist/SHIRO-Models-Japanese) 同梱の訓練済みモデルはこのリポジトリのものです。
- **[oatsu-gh](https://github.com/oatsu-gh)** — [ENUNU](https://github.com/oatsu-gh/ENUNU) の開発。同梱のかな→音素変換テーブルは ENUNU の `kana2phonemes_etk_001.table` をベースとしています。

## ライセンス

GPLv3 — オリジナルの [SHIRO](https://github.com/Sleepwalking/SHIRO) および [liblrhsmm](https://github.com/Sleepwalking/liblrhsmm) に準拠します。
