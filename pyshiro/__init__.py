"""
pyshiro — Python reimplementation of SHIRO phoneme-to-speech alignment toolkit.

Provides:
  - HSMM model loading/saving (.hsmm format)
  - MFCC feature extraction (SHIRO/ciglet compatible)
  - Viterbi forced alignment (2-pass: HMM bootstrap + HSMM refinement)
  - Label output (HTK .lab and Praat TextGrid)
  - Kana-to-phoneme conversion (Japanese)
  - HSMM model training from corpus
"""

from pyshiro.model    import load_hsmm, save_hsmm, HSMMModel
from pyshiro.features import extract_mfcc, extract_mfcc_from_file
from pyshiro.align    import (load_phonemap, build_state_sequence,
                               forced_align, forced_align_2pass,
                               refine_pau_boundaries,
                               realign_external_labels,
                               SILENCE_PHONES)
from pyshiro.labels   import write_lab, write_lab_sec, write_textgrid
from pyshiro.phonemes import load_table, convert_lyric_file, text_to_phonemes

__version__ = "0.1.0"
__all__ = [
    "load_hsmm", "save_hsmm", "HSMMModel",
    "extract_mfcc", "extract_mfcc_from_file",
    "load_phonemap", "build_state_sequence",
    "forced_align", "forced_align_2pass",
    "refine_pau_boundaries", "realign_external_labels",
    "SILENCE_PHONES",
    "write_lab", "write_lab_sec", "write_textgrid",
    "load_table", "convert_lyric_file", "text_to_phonemes",
]
