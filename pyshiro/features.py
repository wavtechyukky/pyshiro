"""
features.py

SHIRO 互換 MFCC 特徴抽出（shiro-xxcc / ciglet の完全再現）

アルゴリズム:
  1. フレーム化: center = i * 80、Blackman 窓、境界はゼロパディング
  2. FFT (512点) → 振幅スペクトル
  3. 三角メルフィルタバンク (36チャンネル, 50-8000 Hz, min_width=400 Hz)
     ・面積正規化なし（ciglet スタイル）
  4. log2 + フロア -15
  5. 直交 DCT-II → C0 スキップ → C1〜C12 (12次元)
  6. デルタ/加速度: [-0.5,0,0.5] / [0.25,0,-0.5,0,0.25]、境界ゼロパディング
"""

from pathlib import Path
from typing import List, Union

import numpy as np
import soundfile as sf
from scipy.fft import rfft, dct as scipy_dct


# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------

SR         = 16000
FRAME_SIZE = 512
HOP_SIZE   = 80
N_MELS     = 36
N_MFCC     = 12
FMIN       = 50.0
LOG_FLOOR  = -15.0
_EPS       = 1e-30

HOP_TIME = HOP_SIZE / SR   # 0.005 s/frame


# ---------------------------------------------------------------------------
# ciglet 互換メルスケール変換
# ---------------------------------------------------------------------------

def _freq2mel(f: np.ndarray) -> np.ndarray:
    return 1125.0 * np.log2(1.0 + f / 700.0)


def _mel2freq(m: np.ndarray) -> np.ndarray:
    return 700.0 * (2.0 ** (m / 1125.0) - 1.0)


# ---------------------------------------------------------------------------
# ciglet 互換三角メルフィルタバンク（面積正規化なし）
# ---------------------------------------------------------------------------

def _build_melfilterbank(nfft: int, sr: int, n_mels: int,
                         fmin: float, fmax: float,
                         min_width: float = 400.0) -> np.ndarray:
    """
    ciglet の cig_create_melfreq_filterbank を再現する。
    面積正規化なし・min_width 制約あり。

    Returns
    -------
    fb : (n_mels, nfft//2+1)
    """
    nf   = nfft // 2 + 1
    fnyq = sr / 2.0

    # n_mels+1 点のメル均等分割 → Hz 変換
    mmin  = _freq2mel(np.array([fmin]))[0]
    mmax  = _freq2mel(np.array([fmax]))[0]
    freqs = _mel2freq(np.arange(n_mels + 1) / n_mels * (mmax - mmin) + mmin)

    fb = np.zeros((n_mels, nf), dtype=np.float64)

    for j in range(n_mels):
        f_0 = 0.0 if j == 0 else freqs[j - 1]
        f_1 = freqs[j]
        f_2 = freqs[j + 1]

        # min_width 制約（低周波チャンネルを最低 min_width Hz に広げる）
        if f_0 > f_1 - min_width:
            f_0 = max(0.0, f_1 - min_width)
        if f_2 < f_1 + min_width:
            f_2 = f_1 + min_width

        # FFT ビンインデックスへ変換（scale=1.0）
        lower_idx = int(np.floor(f_0 * nf / fnyq)) if j > 0 else 0
        upper_idx = min(nf, int(np.ceil(f_2 * nf / fnyq)))
        centr_idx = min(upper_idx - 1, round(f_1 * nf / fnyq))
        lower_idx = min(lower_idx, centr_idx - 1)

        # 三角フィルタ（上昇・下降スロープ）
        if centr_idx > lower_idx:
            for k in range(lower_idx, centr_idx):
                fb[j, k] = (k - lower_idx + 1) / (centr_idx - lower_idx)
        if upper_idx > centr_idx:
            for k in range(centr_idx, upper_idx):
                fb[j, k] = 1.0 - (k - centr_idx) / (upper_idx - centr_idx)

    return fb


# ---------------------------------------------------------------------------
# 起動時にフィルタバンクと窓関数を生成
# ---------------------------------------------------------------------------

_WINDOW = np.blackman(FRAME_SIZE)
_MEL_FB = _build_melfilterbank(FRAME_SIZE, SR, N_MELS, FMIN, SR / 2.0)


# ---------------------------------------------------------------------------
# デルタ計算（SHIRO 互換: 境界ゼロパディング）
# ---------------------------------------------------------------------------

def _apply_delta(feat: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    境界をゼロパディングして有限差分デルタを計算する（SHIRO/ciglet 方式）。
    """
    T, D = feat.shape
    half  = len(kernel) // 2
    # ゼロパディング（前後 half フレーム分）
    padded = np.zeros((T + 2 * half, D), dtype=np.float64)
    padded[half: half + T] = feat

    delta = np.zeros_like(feat)
    for k, w in enumerate(kernel):
        if w != 0.0:
            delta += w * padded[k: k + T]
    return delta


# ---------------------------------------------------------------------------
# メイン: MFCC 抽出
# ---------------------------------------------------------------------------

def extract_mfcc(audio: np.ndarray, sr: int = SR) -> List[np.ndarray]:
    """
    音声配列から SHIRO 互換 MFCC を抽出する。

    フレーム化方式: center = i * HOP_SIZE（ciglet 方式、境界ゼロパディング）

    Parameters
    ----------
    audio : (N,) float32 モノラル波形
    sr    : サンプルレート

    Returns
    -------
    [mfcc, delta1, delta2]  各 shape (T, 12)
    """
    if sr != SR:
        from scipy.signal import resample_poly
        audio = resample_poly(audio, SR, sr).astype(np.float32)

    audio = np.asarray(audio, dtype=np.float64)
    n_frames = len(audio) // HOP_SIZE
    half     = FRAME_SIZE // 2

    # ----- フレーム化（center ベース, ゼロパディング） -----
    # shape (n_frames, FRAME_SIZE)
    frames = np.zeros((n_frames, FRAME_SIZE), dtype=np.float64)
    for i in range(n_frames):
        center = i * HOP_SIZE
        a_start = center - half
        a_end   = center + half          # exclusive

        # 音声配列の有効範囲
        src_lo = max(0, a_start)
        src_hi = min(len(audio), a_end)
        # フレーム内の書き込み位置
        dst_lo = src_lo - a_start
        dst_hi = dst_lo + (src_hi - src_lo)
        frames[i, dst_lo:dst_hi] = audio[src_lo:src_hi]

    # ----- Blackman 窓 -----
    frames *= _WINDOW  # (n_frames, FRAME_SIZE)

    # ----- FFT → 振幅スペクトル -----
    mag = np.abs(rfft(frames, n=FRAME_SIZE, axis=1))  # (T, 257)

    # ----- メルフィルタバンク -----
    mel = mag @ _MEL_FB.T  # (T, 36)

    # ----- log2 + フロア -----
    log_mel = np.maximum(np.log2(mel + _EPS), LOG_FLOOR)  # (T, 36)

    # ----- 直交 DCT-II → C1〜C12（C0 スキップ） -----
    cepstrum = scipy_dct(log_mel, type=2, norm='ortho', axis=1)  # (T, 36)
    mfcc     = cepstrum[:, 1: N_MFCC + 1]                        # (T, 12)

    # ----- デルタ / 加速度（ゼロパディング） -----
    _D1 = np.array([-0.5,  0.0,  0.5])
    _D2 = np.array([ 0.25, 0.0, -0.5, 0.0, 0.25])
    delta1 = _apply_delta(mfcc, _D1)
    delta2 = _apply_delta(mfcc, _D2)

    return [mfcc.astype(np.float32),
            delta1.astype(np.float32),
            delta2.astype(np.float32)]


def extract_mfcc_from_file(wav_path: Union[str, Path]) -> List[np.ndarray]:
    """WAV ファイルから MFCC を抽出する。"""
    audio, sr = sf.read(str(wav_path), always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return extract_mfcc(audio.astype(np.float32), sr)
