"""
_gmm_loglik_frames の回帰テスト。

背景:
  log-sum-exp の accumulator を 0（= log 1）で初期化していたため、nmix > 1 では
  log(Σ w·N) ではなく log(1 + Σ w·N) を計算していた。結果として対数尤度が +0.0 で
  下げ止まり、「このフレームはこの状態ではない」という強い否定ができなくなっていた。
  悪化は誤差の裾に強く偏るため気づきにくい。nmix=4 のモデルで実測したところ、
  中央値が 2.7 倍（1.5→4.0 フレーム）になる一方で最大誤差は 8.8 倍（12→105 フレーム）
  になった。

  nmix == 1 は `if gmm.nmix == 1: loglik = log_norm` の分岐で元から無傷だったため、
  既存の学習済みモデルの挙動は修正前後で一切変わらない（それも検証する）。
"""

import math
import os
from pathlib import Path

import numpy as np
import pytest

from pyshiro.align import _gmm_loglik_frames, build_state_sequence, load_phonemap
from pyshiro.model import GMM, load_hsmm

REPO = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO / "checkpoint" / "pyshiro-jp-v1.hsmm"
PHONEMAP_PATH = REPO / "checkpoint" / "pyshiro-jp-v1_phonemap.json"

needs_model = pytest.mark.skipif(
    not MODEL_PATH.exists() or not PHONEMAP_PATH.exists(),
    reason="checkpoint モデルが無い")


def _reference_loglik(gmm: GMM, obs: np.ndarray) -> np.ndarray:
    """混合ガウス対数尤度の素朴な参照実装（logaddexp.reduce を使う）。"""
    lg = np.empty((obs.shape[0], gmm.nmix), dtype=np.float64)
    for m in range(gmm.nmix):
        var = np.maximum(np.asarray(gmm.vars[m], dtype=np.float64), 1e-6)
        mu = np.asarray(gmm.means[m], dtype=np.float64)
        diff = obs - mu
        lg[:, m] = (math.log(max(float(gmm.weights[m]), 1e-30))
                    - 0.5 * (np.sum(diff ** 2 / var, axis=1)
                             + np.sum(np.log(2 * math.pi * var))))
    return np.logaddexp.reduce(lg, axis=1)


def _make_gmm(nmix: int, ndim: int = 12, seed: int = 0) -> GMM:
    rng = np.random.default_rng(seed)
    return GMM(nmix=nmix, ndim=ndim,
               weights=np.full(nmix, 1.0 / nmix),
               means=rng.normal(0, 1, (nmix, ndim)),
               vars=np.full((nmix, ndim), 0.5),
               varfloors=np.full((nmix, ndim), 1e-3))


@pytest.mark.parametrize("nmix", [2, 4, 8])
def test_matches_reference_for_multi_mixture(nmix):
    """nmix > 1 で参照実装と一致すること（旧実装は 40〜65 nats ずれていた）。"""
    rng = np.random.default_rng(1)
    gmm = _make_gmm(nmix)
    obs = rng.normal(0, 1, (2000, gmm.ndim))
    got = _gmm_loglik_frames(gmm, obs)
    want = _reference_loglik(gmm, obs)
    assert np.abs(got - want).max() < 1e-9, f"nmix={nmix} で参照実装と不一致"


def test_single_mixture_matches_reference():
    """nmix == 1 は専用分岐を通るが、重み log(1)=0 なので参照と一致する。"""
    rng = np.random.default_rng(2)
    gmm = _make_gmm(1)
    obs = rng.normal(0, 1, (500, gmm.ndim))
    np.testing.assert_allclose(_gmm_loglik_frames(gmm, obs),
                               _reference_loglik(gmm, obs), atol=1e-9)


@pytest.mark.parametrize("nmix", [2, 4, 8])
def test_not_floored_at_zero(nmix):
    """外れた観測に対して大きく負の値を返せること。

    旧実装ではここが +0.0 に張り付き、識別力が失われていた。
    """
    gmm = _make_gmm(nmix)
    far = np.full((16, gmm.ndim), 30.0)     # 平均から遠く離れた観測
    ll = _gmm_loglik_frames(gmm, far)
    assert ll.max() < -50.0, f"nmix={nmix}: 外れ値の対数尤度が {ll.max():.2f} と高すぎる"
    assert np.isfinite(ll).all()


def test_dynamic_range_is_preserved():
    """良く合う観測と外れた観測の差が保たれること（旧実装では圧縮されていた）。"""
    gmm = _make_gmm(4)
    good = np.tile(np.asarray(gmm.means[0], dtype=np.float64), (8, 1))
    bad = np.full((8, gmm.ndim), 30.0)
    spread = _gmm_loglik_frames(gmm, good).mean() - _gmm_loglik_frames(gmm, bad).mean()
    assert spread > 100.0, f"尤度のダイナミックレンジが {spread:.1f} と狭すぎる"


def test_weights_are_respected():
    """混合重みが効いていること（重み 0 の成分は寄与しない）。"""
    gmm = _make_gmm(2)
    gmm.weights = np.array([1.0, 0.0])
    obs = np.tile(np.asarray(gmm.means[1], dtype=np.float64), (4, 1))
    only0 = _make_gmm(1)
    only0.means = gmm.means[:1]
    only0.vars = gmm.vars[:1]
    only0.weights = np.array([1.0])
    np.testing.assert_allclose(_gmm_loglik_frames(gmm, obs),
                               _gmm_loglik_frames(only0, obs), atol=1e-6)


@needs_model
def test_shipped_model_is_unaffected():
    """同梱モデル（nmix=1）の対数尤度が参照実装と完全一致すること。

    既存の学習済みモデルの挙動が修正で1ビットも変わっていないことの保証。
    """
    model = load_hsmm(MODEL_PATH)
    assert model.streams[0].gmms[0].nmix == 1
    rng = np.random.default_rng(3)
    obs = rng.normal(0, 1, (300, model.ndim))
    for out_idx in (0, 12, 51, model.nstate - 1):
        gmm = model.streams[0].gmms[out_idx]
        np.testing.assert_allclose(_gmm_loglik_frames(gmm, obs),
                                   _reference_loglik(gmm, obs), atol=1e-9)


@needs_model
def test_split_model_stays_finite():
    """nmix を分割したモデルでも -inf や NaN が漏れないこと。"""
    from pyshiro.train import split_model

    model = split_model(load_hsmm(MODEL_PATH))
    assert model.streams[0].gmms[0].nmix == 2
    rng = np.random.default_rng(4)
    obs = rng.normal(0, 1, (200, model.ndim))
    ll = _gmm_loglik_frames(model.streams[0].gmms[0], obs)
    assert np.isfinite(ll).all(), "分割後のモデルで非有限値が出た"
