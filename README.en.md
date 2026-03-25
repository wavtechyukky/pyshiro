# pyshiro

Python reimplementation of [SHIRO](https://github.com/Sleepwalking/SHIRO) — a phoneme-to-speech forced alignment toolkit based on Hidden Semi-Markov Models (HSMM), designed for Japanese singing voice.

## Features

- **Forced alignment**: Align phoneme transcriptions with audio using HSMM Viterbi decoding
- **2-pass alignment**: HMM bootstrap → HSMM refinement (same approach as original SHIRO)
- **Model training**: Train `.hsmm` models from a labeled corpus
- **Label output**: HTK `.lab` (ENUNU/UTAU compatible) and Praat TextGrid
- **Kana-to-phoneme conversion**: Japanese hiragana → phoneme sequence
- **Lightweight**: Core depends only on `numpy`, `scipy`, `soundfile`, `numba`, `msgpack`

## Installation

```bash
git clone --recurse-submodules https://github.com/your-username/pyshiro
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

The lyrics `.txt` file should be written in hiragana, one phrase per line. Each line break becomes a `pau` (pause) in the phoneme sequence.

```
きっと
とべば
そらまでとどく
```

Breath (`br`) and pause (`pau`) can also be written directly:

```
pau
きっと
br
とべば
```

## CLI

```bash
# Alignment
pyshiro-align audio.wav lyrics.txt \
  --model models/intunist-jp6_generic.hsmm \
  --phonemap models/intunist-jp6_phonemap.json \
  --out output.lab

# Training
pyshiro-train \
  --wav_dir  corpus/wav \
  --lab_dir  corpus/lab \
  --phonemap models/intunist-jp6_phonemap.json \
  --out      my_model.hsmm \
  --iters    5
```

## Pre-trained Models

Pre-trained Japanese models are included as a git submodule from [intunist/SHIRO-Models-Japanese](https://github.com/intunist/SHIRO-Models-Japanese), trained on 17.8 hours of male and female Japanese speech.

## Requirements

- Python 3.10+
- numpy, scipy, soundfile, numba, msgpack

## License

GPLv3 — same as the original [SHIRO](https://github.com/Sleepwalking/SHIRO) and [liblrhsmm](https://github.com/Sleepwalking/liblrhsmm).
