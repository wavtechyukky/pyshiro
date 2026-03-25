"""
model.py

liblrhsmm の .hsmm ファイル（MessagePack バイナリ）を Python で読み書きする。

serial.c (liblrhsmm) の lrh_read_model を参考に実装。
構造:
  model    = array(2) [ [stream...], [duration...] ]
  stream   = array(2) [ weight:float, [gmm...] ]
  gmm      = array(4) [ nmix:int, ndim:int, [weight...], [mean,var,vf ...] ]
  duration = array(6) [ mean, var, floor, ceil, fixed_mean, vfloor ]
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import msgpack


# ---------------------------------------------------------------------------
# データクラス
# ---------------------------------------------------------------------------

@dataclass
class GMM:
    nmix: int
    ndim: int
    weights: np.ndarray   # (nmix,)
    means:   np.ndarray   # (nmix, ndim)
    vars:    np.ndarray   # (nmix, ndim)  対角共分散
    varfloors: np.ndarray # (nmix, ndim)


@dataclass
class Stream:
    weight: float
    gmms: List[GMM]       # ngmm 個（= 状態数）


@dataclass
class Duration:
    mean:       float
    var:        float
    floor:      int
    ceil:       int
    fixed_mean: int
    vfloor:     float


@dataclass
class HSMMModel:
    streams:   List[Stream]
    durations: List[Duration]

    @property
    def nstream(self) -> int:
        return len(self.streams)

    @property
    def nduration(self) -> int:
        return len(self.durations)

    @property
    def nstate(self) -> int:
        """ストリーム0 の GMM 数（= 全状態数）"""
        return len(self.streams[0].gmms)

    @property
    def ndim(self) -> int:
        """ストリーム0 の次元数"""
        return self.streams[0].gmms[0].ndim


# ---------------------------------------------------------------------------
# パーサー
# ---------------------------------------------------------------------------

def _parse_gmm(obj) -> GMM:
    assert len(obj) == 4
    nmix  = obj[0]
    ndim  = obj[1]
    w_raw = obj[2]
    m_raw = obj[3]

    weights   = np.array(w_raw, dtype=np.float32)
    means     = np.zeros((nmix, ndim), dtype=np.float32)
    vars_     = np.zeros((nmix, ndim), dtype=np.float32)
    varfloors = np.zeros((nmix, ndim), dtype=np.float32)

    idx = 0
    for i in range(nmix):
        for j in range(ndim):
            means[i, j]     = m_raw[idx];     idx += 1
            vars_[i, j]     = m_raw[idx];     idx += 1
            varfloors[i, j] = m_raw[idx];     idx += 1

    return GMM(nmix=nmix, ndim=ndim,
               weights=weights, means=means,
               vars=vars_, varfloors=varfloors)


def _parse_stream(obj) -> Stream:
    assert len(obj) == 2
    return Stream(weight=float(obj[0]),
                  gmms=[_parse_gmm(g) for g in obj[1]])


def _parse_duration(obj) -> Duration:
    assert len(obj) == 6
    return Duration(
        mean=float(obj[0]), var=float(obj[1]),
        floor=int(obj[2]),  ceil=int(obj[3]),
        fixed_mean=int(obj[4]), vfloor=float(obj[5]),
    )


def load_hsmm(path: Path) -> HSMMModel:
    """.hsmm ファイルを読み込んで HSMMModel を返す。"""
    data = Path(path).read_bytes()
    obj  = msgpack.unpackb(data, raw=False)
    assert len(obj) == 2
    return HSMMModel(
        streams=[_parse_stream(s) for s in obj[0]],
        durations=[_parse_duration(d) for d in obj[1]],
    )


# ---------------------------------------------------------------------------
# シリアライザ
# ---------------------------------------------------------------------------

def save_hsmm(model: HSMMModel, path: Path) -> None:
    """.hsmm (MessagePack) に書き出す。"""

    def _gmm_to_obj(gmm: GMM):
        m_raw = []
        for i in range(gmm.nmix):
            for j in range(gmm.ndim):
                m_raw.append(float(gmm.means[i, j]))
                m_raw.append(float(gmm.vars[i, j]))
                m_raw.append(float(gmm.varfloors[i, j]))
        return [gmm.nmix, gmm.ndim,
                [float(w) for w in gmm.weights],
                m_raw]

    obj = [
        [[s.weight, [_gmm_to_obj(g) for g in s.gmms]] for s in model.streams],
        [[d.mean, d.var, d.floor, d.ceil, d.fixed_mean, d.vfloor]
         for d in model.durations],
    ]
    Path(path).write_bytes(msgpack.packb(obj))
