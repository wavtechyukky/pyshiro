# pyshiro

Python reimplementation of [SHIRO](https://github.com/Sleepwalking/SHIRO) — a phoneme-to-speech forced alignment toolkit based on Hidden Semi-Markov Models (HSMM), designed for Japanese singing voice.

Japanese: [README.md](README.md)

## Features

This project is a near-complete Python port of the original [SHIRO](https://github.com/Sleepwalking/SHIRO), with additional features on top. It is specifically designed to make training and annotation for Japanese singing voice as smooth as possible.

**Alignment**
- **HSMM forced alignment**: Automatic phoneme boundary estimation
- **2-pass alignment**: HMM pre-training → HSMM refinement (same approach as original SHIRO)
- **Skippable phonemes** (`pskip`): Phonemes such as `pau` / `br` can be skipped if absent, specified per-phoneme in the phonemap
- **Topology setting**: Choose state transition pattern per phoneme in the phonemap

**Training**
- **Corpus training**: Train `.hsmm` models from a labeled corpus
- **HMM pre-training** (`--hmm_iters`): Initialize GMM parameters in duration-free HMM mode before enabling the HSMM duration model (equivalent to `shiro-rest -g`)
- **DAEM** (`--daem`): Annealing-based training to stabilize convergence (equivalent to `shiro-rest -D`)
- **GMM mixture splitting** (`--nmix`): Incrementally double the number of Gaussian components for greater expressiveness
- **Triphone expansion** (`pyshiro.untie`): Expand a monophone model into context-dependent triphones (equivalent to `shiro-untie`)

**I/O**
- **Label output**: `.lab` (HTK 100ns integer format, compatible with ENUNU / NNSVS / vLabeler), `.lab` (space-separated seconds), Praat TextGrid, Audacity labels
- **Label input**: `.lab` (HTK 100ns and seconds format auto-detected), TextGrid, Audacity labels
- **Flexible writer input**: `write_lab` / `write_textgrid` / `write_audacity` / `write_lab_sec` accept either frame indices (int) or seconds (float)
- **Kana-to-phoneme**: Japanese hiragana → phoneme sequence (bundled table based on ENUNU's conversion table)

**Label conversion**
- **Re-labeling external labels**: Convert labels produced by other tools to this model's alignment conventions (`realign_external_labels`)
- Segments audio using long vowels / pauses as anchors, then re-aligns each segment with the HSMM
- Choose which phoneme-boundary types to fix with `--fix_transitions` (e.g. only vowel→consonant and silence→consonant)

**Lightweight**: Core depends only on `numpy`, `scipy`, `soundfile`, `numba`, `msgpack`

## Installation

```bash
git clone --recurse-submodules https://github.com/wavtechyukky/pyshiro
cd pyshiro
pip install -e .
```

> `--recurse-submodules` clones the bundled pre-trained models under `models/`.

## Quick Start

```python
import pyshiro

# Load pre-trained model
model    = pyshiro.load_hsmm("models/intunist-jp6_generic.hsmm")
phonemap = pyshiro.load_phonemap("models/intunist-jp6_phonemap.json")

# Extract features (16kHz mono WAV)
streams  = pyshiro.extract_mfcc_from_file("example/wav_16k/akai_kutsu.wav")

# Convert lyrics to phonemes
table    = pyshiro.load_table()   # bundled kana2phonemes table
phonemes = pyshiro.convert_lyric_file("example/lyrics/akai_kutsu.txt", table)

# Align
T         = streams[0].shape[0]
state_seq = pyshiro.build_state_sequence(phonemes, phonemap, T)
segments  = pyshiro.forced_align_2pass(model, streams, state_seq)

# Save as .lab
from pyshiro.labels import segments_to_phoneme_intervals, write_lab
intervals = segments_to_phoneme_intervals(phonemes, segments)
write_lab(intervals, "example/labels/akai_kutsu.lab")
```

## Annotation Guide

For those who want to use pyshiro for practical annotation, **[workflow/annotation_guide.ipynb](workflow/annotation_guide.ipynb)** walks you through the full workflow using a pre-trained model: resampling, segmentation, automatic alignment, manual correction, and merging back into a full-song `.lab`.

## Lyrics Format

The lyrics `.txt` file should be written in hiragana, one phrase per line. Each line break becomes a `pau` (pause) in the phoneme sequence. Phoneme tokens such as `br` (breath) can be embedded directly inline.

```
brねがいわいちどbrはなしたら
brくずれてく
brしゅんかんむかいかぜbrだれもしらないbrあしあとうぉ
br
```

`pau` can also be written directly:

```
pau
きっと
pau
とべば
```

Space-separated phoneme sequences are also supported (a file containing only ASCII letters is automatically interpreted as a phoneme sequence):

```
pau k i cl t o pau t o b e b a pau
```

## CLI

```bash
# Alignment
pyshiro-align audio.wav lyrics.txt \
  --model    models/intunist-jp6_generic.hsmm \
  --phonemap models/intunist-jp6_phonemap.json \
  --out      output.lab

# Output format (lab / textgrid / audacity)
pyshiro-align audio.wav lyrics.txt \
  --model    models/intunist-jp6_generic.hsmm \
  --phonemap models/intunist-jp6_phonemap.json \
  --format   textgrid \
  --out      output.TextGrid

# Training (basic)
pyshiro-train \
  --wav_dir  corpus/wav \
  --lab_dir  corpus/lab \
  --phonemap models/intunist-jp6_phonemap.json \
  --out      my_model.hsmm \
  --iters    10 \
  --jobs     8   # parallel workers (default: CPU core count)

# Training (HMM pre-training + DAEM + GMM splitting)
pyshiro-train \
  --wav_dir   corpus/wav \
  --lab_dir   corpus/lab \
  --phonemap  models/intunist-jp6_phonemap.json \
  --out       my_model.hsmm \
  --iters     10 \
  --hmm_iters 2 \
  --daem \
  --nmix      4

# Resuming from a checkpoint
# Checkpoints are saved automatically as my_model.iter001.hsmm, my_model.iter002.hsmm, ...
pyshiro-train \
  --wav_dir    corpus/wav \
  --lab_dir    corpus/lab \
  --phonemap   models/intunist-jp6_phonemap.json \
  --out        my_model.hsmm \
  --iters      10 \
  --init_model my_model.iter005.hsmm \
  --start_iter 5

# cap_relax_iter: constrain the search range in early training, release it later
# Useful for corpora with long sustained notes or long pauses
# Specifying test_wav_dir / test_lab_dir logs test log-likelihood at each iteration
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

# Triphone expansion
python -m pyshiro.untie \
  --phonemap     models/intunist-jp6_phonemap.json \
  --model        my_model.hsmm \
  --lab_dir      corpus/lab \
  --out_phonemap my_tri_phonemap.json \
  --out_model    my_tri_model.hsmm

# Visualize alignment (outputs PNG with waveform, mel spectrogram, GT and estimated labels)
python tests/plot_alignment.py \
  --wav_dir  corpus/wav \
  --lab_dir  corpus/lab \
  --model    my_model.hsmm \
  --phonemap my_phonemap.json \
  --out_dir  plots
```

## Training Tips

Practical notes from real-world usage.

**Keep input audio short**
Files up to around 20 seconds tend to work reliably. Long silent sections (`pau`) can cause the search to fail, so avoid passing full song recordings directly — split audio into phrase-level segments before use.

**Use `--nmix 1` (default)**
Increasing the GMM mixture count adds expressiveness, but tends to overfit when the training corpus is small. For typical singing voice corpora, the default of `--nmix 1` gives stable results.

**Use `--cap_relax_iter` to constrain early training**
Right after HMM pre-training, the acoustic model is still rough and tends to misalign long sustained notes or long pauses. Setting `--cap_relax_iter 5` restricts the search range in early iterations for stable convergence, then releases the constraint in later iterations for refinement. This is recommended for most corpora and is included in the recommended settings above.

**Overfitting is fast — data quality matters more than iteration count**
Even as training log-likelihood continues to improve, test log-likelihood typically plateaus within a few iterations. Running more iterations yields diminishing returns; providing a **corpus with consistent, high-quality labels** has a much greater impact on alignment accuracy. When mixing data from multiple annotators, ensure that phoneme boundary conventions are aligned.

## Label Formats

Writer functions accept either frame indices (`int`) or seconds (`float`).

```python
from pyshiro.labels import (
    write_lab, write_lab_sec, write_textgrid, write_audacity,
    read_lab, read_textgrid, read_audacity,
)

# Write (frame indices or seconds both accepted)
write_lab(intervals, "output.lab")           # HTK 100ns integer (ENUNU / NNSVS / vLabeler compatible)
write_lab_sec(intervals, "output.lab")       # space-separated seconds (same layout, values in seconds)
write_textgrid(intervals, "output.TextGrid") # Praat TextGrid
write_audacity(intervals, "output.txt")      # Audacity labels (tab-separated, seconds)

# Read back hand-corrected labels
# read_lab auto-detects HTK 100ns integer vs seconds, and returns seconds
intervals = read_lab("corrected.lab")
intervals = read_textgrid("corrected.TextGrid")
intervals = read_audacity("corrected.txt")
```

## Label Conversion (re-labeling external labels)

Convert external labels (e.g. from the Tohoku Kiritan singing DB) to this model's
alignment conventions. See **[workflow/convert_labels.ipynb](workflow/convert_labels.ipynb)** for details.

```bash
python workflow/04_convert_labels.py \
    audio.wav  external.lab \
    --model    models/intunist-jp6_generic.hsmm \
    --phonemap models/intunist-jp6_phonemap.json \
    --out      converted.lab \
    --format   lab_sec
```

Main options:

| Option | Description |
|---|---|
| `--format` | `lab` (HTK 100ns) / `lab_sec` (seconds) / `textgrid` / `audacity` |
| `--fix_transitions FROM-TO,...` | Comma-separated boundary types to fix (default: fix all). `FROM`/`TO` is `silence` / `vowel` / `consonant`. e.g. `--fix_transitions vowel-consonant,silence-consonant,vowel-silence` |
| `--anchor_vowel_min` | Min vowel length to use as anchor (sec, default 0.5) |
| `--anchor_pau_min` | Min pause length to use as anchor (sec, default 0.5) |
| `--phoneme_map` | Phoneme-name mapping JSON (external → pyshiro) |
| `--unknown_phoneme_map` | Alignment-time substitution for unknown phonemes (e.g. `vy=v,fy=f`). Output keeps original names |

## Label Outlier Check

Detect consonants whose duration is an outlier within the same phoneme type, and plot
their waveforms and phoneme labels on a single image. Based on the heuristic that
**too-short consonants are likely mislabeled**.

```bash
python workflow/05_check_consonant_outliers.py audio.wav labels.lab \
    --side short \
    --out outlier_plots.png
```

`--side short` (default) and `--side long` are available, but **in practice you almost
always look at the short side** — the long side mostly reflects normal lengthening
(consonant closures included) and is a weak signal of mislabeling.

| Option | Default | Description |
|---|---|---|
| `--side` | `short` | Outlier side to detect (`short` / `long`) |
| `--floor_ms` | `40` | [short] Below this length (ms) is flagged unconditionally |
| `--max_dur_ms` | `50` | [short] Detect below this length AND a statistical outlier |
| `--iqr_k` | `1.5` | IQR multiplier for normal consonants |
| `--plosive_iqr_k` | `3.0` | IQR multiplier for plosives/affricates (larger, since closure makes durations bimodal) |

Example output: [example/convert_labels/outlier_plots/](example/convert_labels/outlier_plots/) (short-side plots for the converted Tohoku Kiritan DB).

## Skippable Phonemes (pskip)

Adding `"pskip"` to a phoneme entry in the phonemap allows that phoneme to be skipped during alignment if it is not actually present. This lets you freely annotate `br` (breath) phonemes without worrying about false positives.

```json
{
  "phone_map": {
    "pau": { "pskip": 0.5, "states": [...] },
    "br":  { "pskip": 0.5, "states": [...] }
  }
}
```

## Topology Setting

```json
{
  "phone_map": {
    "cl": { "topology": "type-b", "states": [...] }
  }
}
```

| topology | behavior |
|---|---|
| `type-a` (default) | left-to-right only: 0→1→2 |
| `type-b` | adds skip from each state to the final state |
| `type-c` | adds skip from each state two steps ahead |
| `skip-boundary` | allows skipping the first and last states of a phoneme |

## Kana-to-Phoneme Table

The bundled `pyshiro/data/kana2phonemes.table` is based on the `kana2phonemes_etk_001.table` from [ENUNU](https://github.com/oatsu-gh/ENUNU).

## Pre-trained Models

Pre-trained Japanese models are included as a git submodule from [intunist/SHIRO-Models-Japanese](https://github.com/intunist/SHIRO-Models-Japanese), trained on 17.8 hours of male and female Japanese singing voice.

### Training Data Credits (Custom Model)

The separately distributed custom trained model was built using the following singing voice databases. We are deeply grateful to the creators and rights holders of each database.

- **Onikurumi Voice Database** — Onikurumi ([https://onikuru.info](https://onikuru.info))
- **Ofuton P Voice Database** — DB production: Ofuton P ([https://sites.google.com/view/oftn-utagoedb](https://sites.google.com/view/oftn-utagoedb))
- **Namine Ritsu** — Canon ([https://www.canon-voice.com](https://www.canon-voice.com))
- **Tohoku Kiritan Singing Database** — ©SSS ([https://zunko.jp/kiridev/login.php](https://zunko.jp/kiridev/login.php))
- **No.7 Singing Database** — ©No.7製作委員会 ([https://voiceseven.com/7dev/login.php](https://voiceseven.com/7dev/login.php))
- **Natsume Yuri Voice Database** — DB production: Amanokei, voice provider: Kirino Sota ([https://ksdcm1ng.wixsite.com/njksofficial](https://ksdcm1ng.wixsite.com/njksofficial))

## Requirements

- Python 3.10+
- numpy, scipy, soundfile, numba, msgpack

## Changelog

- **2026-06** — Added label conversion and quality-check tools.
  - **Label conversion** (`realign_external_labels` / `workflow/04_convert_labels.py`): re-label external labels to this model's conventions. Segments audio using long vowels / pauses as anchors and re-aligns each segment with the HSMM. Choose which boundary types to fix via `--fix_transitions` (e.g. `vowel-consonant,silence-consonant,vowel-silence`). See [workflow/convert_labels.ipynb](workflow/convert_labels.ipynb).
    ```bash
    python workflow/04_convert_labels.py audio.wav external.lab \
        --model M.hsmm --phonemap P.json --out out.lab --format lab_sec \
        --fix_transitions vowel-consonant,silence-consonant,vowel-silence
    ```
  - **Outlier check** (`workflow/05_check_consonant_outliers.py`): detect consonants with outlier durations and plot waveforms + labels on one image. Based on the heuristic that too-short consonants are likely mislabeled; in practice `--side short` (default) is the one to look at.
    ```bash
    python workflow/05_check_consonant_outliers.py audio.wav labels.lab \
        --side short --out outliers.png
    ```
  - I/O: `write_lab` / `write_textgrid` / `write_audacity` now accept both frame indices (int) and seconds (float). Added seconds output `write_lab_sec`.
- **2025-03** — Added annotation workflow (`workflow/`). A notebook guides users through resampling, segmentation, automatic alignment, and merging.
- **2025-03** — Added custom pre-trained model (`checkpoint/pyshiro-jp-v1.hsmm`). Trained from scratch on 19.4h across multiple voice databases. Supports `br` (breath) phoneme.

## Acknowledgements

- **[Sleepwalking (Kanru Hua)](https://github.com/Sleepwalking)** — Original [SHIRO](https://github.com/Sleepwalking/SHIRO) and [liblrhsmm](https://github.com/Sleepwalking/liblrhsmm). The HSMM algorithms and `.hsmm` format used in this project are based on his work.
- **[intunist](https://github.com/intunist)** — [SHIRO-Models-Japanese](https://github.com/intunist/SHIRO-Models-Japanese), the bundled pre-trained model.
- **[oatsu-gh](https://github.com/oatsu-gh)** — [ENUNU](https://github.com/oatsu-gh/ENUNU). The bundled kana-to-phoneme table is based on ENUNU's `kana2phonemes_etk_001.table`.

## License

GPLv3 — same as the original [SHIRO](https://github.com/Sleepwalking/SHIRO) and [liblrhsmm](https://github.com/Sleepwalking/liblrhsmm).
