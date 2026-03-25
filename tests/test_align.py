"""
基本的な動作確認テスト。
モデルファイル (.hsmm, phonemap.json) へのパスは環境変数または引数で渡す。
"""

import os
import pytest
import numpy as np
from pathlib import Path

import pyshiro


HSMM_PATH     = os.environ.get("PYSHIRO_MODEL",    "")
PHONEMAP_PATH = os.environ.get("PYSHIRO_PHONEMAP", "")


def test_mfcc_shape():
    audio = np.zeros(16000, dtype=np.float32)
    streams = pyshiro.extract_mfcc(audio)
    assert len(streams) == 3
    for s in streams:
        assert s.shape[1] == 12


def test_phonemes_conversion():
    table = pyshiro.load_table()
    phonemes = pyshiro.text_to_phonemes("きっと\nとべば", table)
    assert phonemes[0] == "pau"
    assert phonemes[-1] == "pau"
    assert "k" in phonemes
    assert "pau" in phonemes


@pytest.mark.skipif(not HSMM_PATH or not PHONEMAP_PATH,
                    reason="PYSHIRO_MODEL / PYSHIRO_PHONEMAP not set")
def test_forced_align():
    model    = pyshiro.load_hsmm(Path(HSMM_PATH))
    phonemap = pyshiro.load_phonemap(Path(PHONEMAP_PATH))
    table    = pyshiro.load_table()

    audio    = np.zeros(16000, dtype=np.float32)
    streams  = pyshiro.extract_mfcc(audio)
    T        = streams[0].shape[0]

    phonemes  = pyshiro.text_to_phonemes("きっと", table)
    state_seq = pyshiro.build_state_sequence(phonemes, phonemap, T)
    segments  = pyshiro.forced_align(model, streams, state_seq,
                                     use_duration=False)
    assert len(segments) == len(state_seq)
