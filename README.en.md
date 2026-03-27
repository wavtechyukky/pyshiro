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
- **Label output**: `.lab` (ENUNU compatible), Praat TextGrid, Audacity labels
- **Label input**: `.lab`, Audacity labels
- **Kana-to-phoneme**: Japanese hiragana → phoneme sequence (bundled table based on ENUNU's conversion table)

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

# Extract features
streams  = pyshiro.extract_mfcc_from_file("audio.wav")

# Convert lyrics to phonemes
table    = pyshiro.load_table()   # bundled kana2phonemes table
phonemes = pyshiro.convert_lyric_file("lyrics.txt", table)

# Align
T         = streams[0].shape[0]
state_seq = pyshiro.build_state_sequence(phonemes, phonemap, T)
segments  = pyshiro.forced_align_2pass(model, streams, state_seq)

# Save as .lab
from pyshiro.labels import segments_to_phoneme_intervals, write_lab
intervals = segments_to_phoneme_intervals(phonemes, segments)
write_lab(intervals, "output.lab")
```

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
  --iters    20 \
  --jobs     8   # parallel workers (default: CPU core count)

# Training (HMM pre-training + DAEM + GMM splitting)
pyshiro-train \
  --wav_dir   corpus/wav \
  --lab_dir   corpus/lab \
  --phonemap  models/intunist-jp6_phonemap.json \
  --out       my_model.hsmm \
  --iters     20 \
  --hmm_iters 3 \
  --daem \
  --nmix      4

# Resuming from a checkpoint
# Checkpoints are saved automatically as my_model.iter1.hsmm, my_model.iter2.hsmm, ...
pyshiro-train \
  --wav_dir    corpus/wav \
  --lab_dir    corpus/lab \
  --phonemap   models/intunist-jp6_phonemap.json \
  --out        my_model.hsmm \
  --iters      20 \
  --init_model my_model.iter10.hsmm \
  --start_iter 10

# cap_relax_iter: constrain the search range in early training, release it later
# Useful for corpora with long sustained notes or long pauses
pyshiro-train \
  --wav_dir        corpus/wav \
  --lab_dir        corpus/lab \
  --phonemap       models/intunist-jp6_phonemap.json \
  --out            my_model.hsmm \
  --iters          20 \
  --hmm_iters      2 \
  --cap_relax_iter 10

# Triphone expansion
python -m pyshiro.untie \
  --phonemap     models/intunist-jp6_phonemap.json \
  --model        my_model.hsmm \
  --lab_dir      corpus/lab \
  --out_phonemap my_tri_phonemap.json \
  --out_model    my_tri_model.hsmm
```

## Label Formats

```python
from pyshiro.labels import (
    write_lab, write_textgrid, write_audacity, read_audacity
)

# Write
write_lab(intervals, "output.lab")           # .lab (ENUNU compatible)
write_textgrid(intervals, "output.TextGrid") # Praat TextGrid
write_audacity(intervals, "output.txt")      # Audacity labels

# Read back hand-corrected labels
intervals = read_audacity("corrected.txt")
```

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

## Requirements

- Python 3.10+
- numpy, scipy, soundfile, numba, msgpack

## TODO

- [ ] **`br` (breath) model**: Bundle a model trained with `br` phoneme labels

## Acknowledgements

- **[Sleepwalking (Kanru Hua)](https://github.com/Sleepwalking)** — Original [SHIRO](https://github.com/Sleepwalking/SHIRO) and [liblrhsmm](https://github.com/Sleepwalking/liblrhsmm). The HSMM algorithms and `.hsmm` format used in this project are based on his work.
- **[intunist](https://github.com/intunist)** — [SHIRO-Models-Japanese](https://github.com/intunist/SHIRO-Models-Japanese), the bundled pre-trained model.
- **[oatsu-gh](https://github.com/oatsu-gh)** — [ENUNU](https://github.com/oatsu-gh/ENUNU). The bundled kana-to-phoneme table is based on ENUNU's `kana2phonemes_etk_001.table`.

## License

GPLv3 — same as the original [SHIRO](https://github.com/Sleepwalking/SHIRO) and [liblrhsmm](https://github.com/Sleepwalking/liblrhsmm).
