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
- **ラベル出力**: `.lab`（HTK 100ns 整数形式・ENUNU / NNSVS / vLabeler 互換）、`.lab`（秒単位スペース区切り）、Praat TextGrid、Audacity ラベル
- **ラベル読み込み**: `.lab`（HTK 100ns・秒単位を自動判定）、TextGrid、Audacity ラベル
- **書き出し関数のフレキシブルな入力**: `write_lab` / `write_textgrid` / `write_audacity` / `write_lab_sec` はフレームインデックス（int）と秒（float）のどちらでも受け付ける
- **かな→音素変換**: ひらがな歌詞 → 音素列変換（ENUNU の変換テーブルをベースに同梱）

**ラベル変換**
- **外部ラベルの再ラベリング**: 他ツールで作成したラベルファイルを pyshiro モデルのアライメント基準に変換する（`realign_external_labels`）
- 長い母音・pau をアンカーとして音声を区分し、各区間を HSMM で再アライメント
- 修正する音素境界タイプを `--fix_transitions` で指定可能（例: 母音→子音・無音→子音のみ修正）

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
# ↓ おすすめ設定（同梱の checkpoint/pyshiro-jp-v2.hsmm と同じ設定）
# --nmix 4 のとき、GMM の分割は iters を3等分した位置で2回実行される。
# 分割後の反復回数を確保するため iters は多めに取る（16 なら iter 5 と 10 で分割）。
pyshiro-train \
  --wav_dir        corpus/wav \
  --lab_dir        corpus/lab \
  --test_wav_dir   corpus/test/wav \
  --test_lab_dir   corpus/test/lab \
  --phonemap       models/intunist-jp6_phonemap.json \
  --out            my_model.hsmm \
  --iters          16 \
  --hmm_iters      2 \
  --daem \
  --nmix           4 \
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

**`--nmix` は 4 を目安にする**
以前このドキュメントでは `--nmix 1` を推奨していましたが、これは `nmix > 1` の対数尤度計算に不具合があったためで、誤った結論でした（更新履歴を参照）。修正後に測り直すと、学習に使っていない歌手の音素境界では `--nmix 1` から `--nmix 4` で平均誤差が約 30% 改善します。`--nmix 8` ではそれ以上の改善は見られず、未学習の歌手で大きく外す事例がむしろ増えたため、歌声コーパス程度の規模では 4 付近が上限の目安です。

**`--cap_relax_iter` で序盤の探索範囲を制限する**
HMM プレトレーニング直後は音響モデルが粗く、ロングトーンや長い pau に誤って大量フレームを割り当ててしまいやすい。`--cap_relax_iter 5` を指定すると序盤の探索範囲を制限しながら安定的に収束させ、後半で制限を外して精細化できる。ほとんどのコーパスで有効なため、上記のおすすめ設定に含めている。

**過学習が早い：イテレーション数より教師データの質が重要**
train の対数尤度が改善し続けていても、テストデータの対数尤度は数イテレーションで頭打ちになることが多い。学習を長く回すより、**ラベリングの一貫性が高いコーパスを用意すること**の方がアライメント精度への寄与が大きい。異なるラベラーのデータを混在させる場合は、音素境界の基準が揃っているかを確認することを推奨する。

## ラベル出力

書き出し関数はフレームインデックス（`int`）と秒（`float`）のどちらでも受け付けます。

```python
from pyshiro.labels import (
    write_lab, write_lab_sec, write_textgrid, write_audacity,
    read_lab, read_textgrid, read_audacity,
)

# 書き出し（フレームインデックスでも秒でも可）
write_lab(intervals, "output.lab")           # HTK 100ns 整数（ENUNU / NNSVS / vLabeler 互換）
write_lab_sec(intervals, "output.lab")       # 秒単位スペース区切り（同形式・値が秒）
write_textgrid(intervals, "output.TextGrid") # Praat TextGrid
write_audacity(intervals, "output.txt")      # Audacity ラベル（タブ区切り・秒）

# 読み込み（手修正済みラベルを訓練データに戻す場合など）
# read_lab は HTK 100ns 整数・秒単位のどちらも自動判定して秒で返す
intervals = read_lab("corrected.lab")
intervals = read_textgrid("corrected.TextGrid")
intervals = read_audacity("corrected.txt")
```

## ラベル変換（外部ラベルの再ラベリング）

東北きりたん歌声DB などの外部ラベルを pyshiro モデルのアライメント基準に変換します。  
詳細は **[workflow/convert_labels.ipynb](workflow/convert_labels.ipynb)** を参照してください。

```bash
python workflow/04_convert_labels.py \
    audio.wav  external.lab \
    --model    models/intunist-jp6_generic.hsmm \
    --phonemap models/intunist-jp6_phonemap.json \
    --out      converted.lab \
    --format   lab_sec
```

主なオプション:

| オプション | 説明 |
|---|---|
| `--format` | `lab`（HTK 100ns）/ `lab_sec`（秒）/ `textgrid` / `audacity` |
| `--fix_transitions FROM-TO,...` | 修正する音素境界タイプをカンマ区切りで指定（省略時は全境界を修正）。`FROM`/`TO` は `silence` / `vowel` / `consonant`。例: `--fix_transitions vowel-consonant,silence-consonant,vowel-silence` |
| `--anchor_vowel_min` | アンカーにする母音の最小長（秒, デフォルト 0.5） |
| `--anchor_pau_min` | アンカーにする pau の最小長（秒, デフォルト 0.5） |
| `--phoneme_map` | 音素名変換テーブル JSON（外部→pyshiro） |
| `--unknown_phoneme_map` | 未知音素のアライメント時代替（例: `vy=v,fy=f`）。出力ラベルは元の音素名を維持 |

## ラベルの外れ値チェック

同一種類の子音のうち duration が外れ値のものを検出し、波形と音素ラベルを 1 枚の画像にまとめてプロットします。**短すぎる子音はラベリングに失敗している可能性が高い**、という経験則に基づくチェック用ツールです。

```bash
python workflow/05_check_consonant_outliers.py audio.wav labels.lab \
    --side short \
    --out outlier_plots.png
```

`--side short`（デフォルト）と `--side long` を選べますが、**実用上は短い側（short）を見ることがほとんど**です。長い側は閉鎖区間込みの正常な伸長が多く、明確な失敗の信号にはなりにくいためです。

主なオプション:

| オプション | デフォルト | 説明 |
|---|---|---|
| `--side` | `short` | 検出する外れ値の側（`short` / `long`）|
| `--floor_ms` | `40` | [short] この長さ（ms）未満は無条件でフラグ |
| `--max_dur_ms` | `50` | [short] この長さ未満かつ統計的外れ値を検出 |
| `--iqr_k` | `1.5` | 通常子音の IQR 倍率 |
| `--plosive_iqr_k` | `3.0` | 破裂音・破擦音の IQR 倍率（閉鎖込みで二峰性のため大きめ）|

出力例: [example/convert_labels/outlier_plots/](example/convert_labels/outlier_plots/)（東北きりたん DB の変換結果に対する short 側プロット）

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

### 同梱のカスタムモデル

`checkpoint/` に2つ同梱しています。用途によって使い分けてください。

| モデル | nmix | 特徴 |
|---|---|---|
| `pyshiro-jp-v1.hsmm` | 1 | 従来モデル。**`br`（息継ぎ）の区間推定はこちらの方が安定します。** |
| `pyshiro-jp-v2.hsmm` | 4 | 新モデル。全体の境界精度は v1 より高い。ただし `br` は苦手（下記）。 |

**v2 の既知の弱点 — `br`（息継ぎ）**
v2 は`br` の区間を実際より短く推定する傾向があります。手元の測定（ある歌唱データベースの `br` 1,015 区間）では、元ラベルとの平均差が v1 の 8.4 フレームに対し v2 は 16.4 フレーム、200 ms 以上ずれた区間は v1 の 13 件に対し v2 は 133 件でした（1 フレーム = 5 ms）。`br` を多く含む素材を扱う場合は v1 の併用を検討してください。この差は継続時間の下限（`durfloor`）の調整では埋まらないことを確認しています。

それ以外の用途では v2 が上回ります。学習に使っていない歌手の人手検証済み音素境界で、平均誤差は v1 の 3.27 フレームに対し v2 は 2.05 フレームでした。

### `pyshiro-jp-v1.hsmm` の学習データ謝辞

`checkpoint/pyshiro-jp-v1.hsmm` は、以下の歌声データベースを使用して訓練されています。各データベースの制作者・権利者に深く感謝申し上げます。

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

- **2026-07** — **`--nmix > 1` のときに対数尤度が誤って計算される不具合を修正。**
  混合ガウス分布の log-sum-exp を累算する変数を `-inf`（= log 0）ではなく `0`（= log 1）で
  初期化していたため、`log(Σ w·N)` ではなく `log(1 + Σ w·N)` を計算していました。
  対数尤度が `+0.0` より下がらないため「このフレームはこの状態ではない」という強い否定ができず、
  識別が必要な音素境界でアライメントが破綻します。悪化は誤差の裾に強く偏るため気づきにくく、
  手元の測定では中央値が 2.7 倍になる一方で最大誤差は 8.8 倍になりました。
  `--nmix 1` は別の経路（`if gmm.nmix == 1`）を通るため影響を受けません。
  **既存モデルの出力は変わりません**（同一データで修正前後の結果が完全に一致することを確認済み）。
  回帰テストを [tests/test_align_gmm.py](tests/test_align_gmm.py) に追加しました。
- **2026-07** — 訓練済みモデル `checkpoint/pyshiro-jp-v2.hsmm` を追加（`--nmix 4` で訓練）。
  学習に使っていない歌手の人手検証済み音素境界で、平均誤差が v1 比 3.27 → 2.05 フレーム
  （1 フレーム = 5 ms）に改善しました。ただし `br`（息継ぎ）は v1 の方が安定します（[同梱のカスタムモデル](#同梱のカスタムモデル)を参照）。
  `checkpoint/pyshiro-jp-v1.hsmm` も引き続き同梱します。
- **2026-06** — ラベル変換・品質チェック機能を追加。
  - **ラベル変換**（`realign_external_labels` / `workflow/04_convert_labels.py`）: 他ツールで作成した外部ラベルを pyshiro モデルの基準に再ラベリング。長い母音・pau をアンカーに音声を区分し、各区間を HSMM で再アライメントする。修正する音素境界タイプを `--fix_transitions` で選択可能（例: `vowel-consonant,silence-consonant,vowel-silence` で母音→子音・無音→子音・母音→無音のみ修正）。詳しい手順は [workflow/convert_labels.ipynb](workflow/convert_labels.ipynb)。
    ```bash
    python workflow/04_convert_labels.py audio.wav external.lab \
        --model M.hsmm --phonemap P.json --out out.lab --format lab_sec \
        --fix_transitions vowel-consonant,silence-consonant,vowel-silence
    ```
  - **外れ値チェック**（`workflow/05_check_consonant_outliers.py`）: 同種子音内で duration が外れ値の箇所を検出し、波形＋音素ラベルを 1 枚にプロット。短すぎる子音はラベリング失敗の可能性が高いという経験則に基づく。実用上は `--side short`（デフォルト）を見ることがほとんど。
    ```bash
    python workflow/05_check_consonant_outliers.py audio.wav labels.lab \
        --side short --out outliers.png
    ```
  - 入出力の拡充: `write_lab` / `write_textgrid` / `write_audacity` がフレームインデックス（int）・秒（float）の両入力に対応。秒単位出力 `write_lab_sec` を追加。
- **2025-03** — アノテーションワークフロー（`workflow/`）を追加。音声の変換・分割・自動アライメント・結合までをノートブックでガイド。
- **2025-03** — 独自訓練済みモデル（`checkpoint/pyshiro-jp-v1.hsmm`）を追加。複数歌声データベース計 19.4時間で追加訓練。`br`（吐息）音素に対応。

## 謝辞

- **[Sleepwalking (Kanru Hua)](https://github.com/Sleepwalking)** — オリジナル [SHIRO](https://github.com/Sleepwalking/SHIRO) および [liblrhsmm](https://github.com/Sleepwalking/liblrhsmm) の設計・実装。本プロジェクトのアルゴリズムと `.hsmm` フォーマットはこれらに基づいています。
- **[intunist](https://github.com/intunist)** — [SHIRO-Models-Japanese](https://github.com/intunist/SHIRO-Models-Japanese) 同梱の訓練済みモデルはこのリポジトリのものです。
- **[oatsu-gh](https://github.com/oatsu-gh)** — [ENUNU](https://github.com/oatsu-gh/ENUNU) の開発。同梱のかな→音素変換テーブルは ENUNU の `kana2phonemes_etk_001.table` をベースとしています。

## ライセンス

GPLv3 — オリジナルの [SHIRO](https://github.com/Sleepwalking/SHIRO) および [liblrhsmm](https://github.com/Sleepwalking/liblrhsmm) に準拠します。
