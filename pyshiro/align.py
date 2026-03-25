"""
align.py

SHIRO 互換 HSMM 強制アライメント（Viterbi デコーディング）

phonemap.json で定義された音素→状態マッピングと
.hsmm モデルの GMM + 継続時間分布を使い、
音素列の各音素がどのフレーム区間に対応するかを推定する。
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import numba

from pyshiro.model import HSMMModel, GMM

# フレームレート (hop=80, sr=16000)
HOP_TIME = 80 / 16000   # 0.005 s/frame
FPS      = 1.0 / HOP_TIME  # 200 frames/sec

NEG_INF = -1e30


# ---------------------------------------------------------------------------
# phonemap 読み込み
# ---------------------------------------------------------------------------

def load_phonemap(path: Path) -> dict:
    """phonemap.json を読み込み phone_map dict を返す。"""
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    return data['phone_map']


# ---------------------------------------------------------------------------
# 状態列の構築
# ---------------------------------------------------------------------------

@dataclass
class StateInfo:
    phoneme:   str
    state_idx: int   # 音素内での状態番号 (0/1/2)
    dur_idx:   int   # モデルの継続時間状態インデックス
    out_idx:   int   # GMM インデックス（全ストリーム共通）
    floor:     int   # 最小継続フレーム数
    ceil:      int   # 最大継続フレーム数（0=無制限）


def build_state_sequence(phonemes: List[str], phonemap: dict,
                         total_frames: int) -> List[StateInfo]:
    """
    音素列 → StateInfo のリストに展開する。
    各音素は 3 状態を持つ。
    """
    states: List[StateInfo] = []

    for ph in phonemes:
        if ph not in phonemap:
            raise ValueError(f"音素 '{ph}' が phonemap に存在しません")
        entry  = phonemap[ph]
        nst    = len(entry['states'])
        floors = entry.get('durfloor', [0.0] * nst)
        ceils  = entry.get('durceil',  [0.0] * nst)

        for k, st in enumerate(entry['states']):
            fl_sec = floors[k] if k < len(floors) else 0.0
            ce_sec = ceils[k]  if k < len(ceils)  else 0.0

            floor_f = max(1, round(fl_sec * FPS)) if fl_sec > 0 else 1
            ceil_f  = round(ce_sec * FPS)          if ce_sec > 0 else total_frames

            states.append(StateInfo(
                phoneme   = ph,
                state_idx = k,
                dur_idx   = st['dur'],
                out_idx   = st['out'][0],  # 全ストリーム共通
                floor     = floor_f,
                ceil      = ceil_f,
            ))

    return states


# ---------------------------------------------------------------------------
# フレームレベル対数尤度の計算
# ---------------------------------------------------------------------------

def _gmm_loglik_frames(gmm: GMM, obs: np.ndarray) -> np.ndarray:
    """
    単一 GMM の全フレームに対する対数尤度を返す。

    Parameters
    ----------
    gmm : GMM  (nmix=1 を想定)
    obs : (T, ndim)

    Returns
    -------
    loglik : (T,)
    """
    T = obs.shape[0]
    loglik = np.zeros(T, dtype=np.float64)

    for m in range(gmm.nmix):
        mu  = gmm.means[m]        # (ndim,)
        var = gmm.vars[m]         # (ndim,)
        var = np.maximum(var, 1e-6)  # ゼロ分散を防ぐ

        diff = obs - mu           # (T, ndim)
        # log N(x | mu, diag(var)) = -0.5 * [sum_j (x_j-mu_j)^2/var_j + log(2pi*var_j)]
        log_norm = -0.5 * (np.sum(diff ** 2 / var, axis=1)
                           + np.sum(np.log(2 * math.pi * var)))
        if gmm.nmix == 1:
            loglik = log_norm
        else:
            log_w = math.log(max(gmm.weights[m], 1e-30))
            loglik = np.logaddexp(loglik, log_w + log_norm)

    return loglik


def compute_frame_loglik(model: HSMMModel, out_idx: int,
                         streams: List[np.ndarray]) -> np.ndarray:
    """
    あるモデル状態（GMM インデックス = out_idx）における
    全フレームの対数尤度（全ストリーム加重和）を返す。

    Parameters
    ----------
    model   : HSMMModel
    out_idx : int  GMM インデックス
    streams : list of (T, 12)  各ストリームの特徴量

    Returns
    -------
    loglik : (T,)
    """
    T      = streams[0].shape[0]
    loglik = np.zeros(T, dtype=np.float64)

    for l, stream in enumerate(model.streams):
        gmm = stream.gmms[out_idx]
        ll  = _gmm_loglik_frames(gmm, streams[l].astype(np.float64))
        loglik += stream.weight * ll

    return loglik


# ---------------------------------------------------------------------------
# 継続時間対数尤度
# ---------------------------------------------------------------------------

def _log_dur_prob(d: int, dur_mean: float, dur_var: float) -> float:
    """
    ガウス継続時間分布の対数確率。
    var <= 0 の場合は一様分布（0 を返す）。
    """
    if dur_var <= 0:
        return 0.0
    return -0.5 * ((d - dur_mean) ** 2 / dur_var
                   + math.log(2 * math.pi * dur_var))


# ---------------------------------------------------------------------------
# Viterbi DP カーネル（Numba JIT）
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def _viterbi_dp(N, T,
                cum_ll_mat,  # (n_unique_out, T+1) float64
                out_row,     # (N,) int32  — cum_ll_mat の行インデックス
                floors,      # (N,) int32
                ceils,       # (N,) int32
                dur_means,   # (N,) float64
                dur_vars,    # (N,) float64
                t_lo_arr,    # (N,) int32
                t_hi_arr,    # (N,) int32
                use_duration):
    """HSMM Viterbi DP の内側ループ（純粋 numpy 配列だけ受け取る）。"""
    NEG_INF = -1e30
    TWO_PI  = 2.0 * math.pi

    score = np.full((N, T + 1), NEG_INF)
    back  = np.zeros((N, T + 1), dtype=np.int32)

    # --- 初期化（状態 0、フレーム 0 から始まる） ---
    fl0 = floors[0]
    ce0 = ceils[0]
    for t in range(t_lo_arr[0], t_hi_arr[0] + 1):
        d = t
        if d < fl0 or d > ce0:
            continue
        s = cum_ll_mat[out_row[0], t] - cum_ll_mat[out_row[0], 0]
        if use_duration:
            dv = dur_vars[0]
            if dv > 0.0:
                dm = dur_means[0]
                s += -0.5 * ((d - dm) ** 2 / dv + math.log(TWO_PI * dv))
        score[0, t] = s
        back[0, t]  = d

    # --- 再帰 ---
    for n in range(1, N):
        fl  = floors[n]
        ce  = ceils[n]
        row = out_row[n]
        dm  = dur_means[n]
        dv  = dur_vars[n]
        use_dur_n = use_duration and dv > 0.0
        log_denom = math.log(TWO_PI * dv) if use_dur_n else 0.0

        for t in range(t_lo_arr[n], t_hi_arr[n] + 1):
            best_score = NEG_INF
            best_d     = 0
            d_max = ce if ce < t else t
            for d in range(fl, d_max + 1):
                prev_t = t - d
                ps = score[n - 1, prev_t]
                if ps <= NEG_INF:
                    continue
                ll = cum_ll_mat[row, t] - cum_ll_mat[row, prev_t]
                if use_dur_n:
                    ll += -0.5 * ((d - dm) ** 2 / dv + log_denom)
                lp = ps + ll
                if lp > best_score:
                    best_score = lp
                    best_d     = d
            score[n, t] = best_score
            back[n, t]  = best_d

    return score, back


# ---------------------------------------------------------------------------
# Viterbi 強制アライメント
# ---------------------------------------------------------------------------

def forced_align(model: HSMMModel, streams: List[np.ndarray],
                 state_seq: List[StateInfo],
                 use_duration: bool = True,
                 initial_segments: List[Tuple[int, int]] = None,
                 window: int = 60) -> List[Tuple[int, int]]:
    """
    HSMM Viterbi 強制アライメント。

    Parameters
    ----------
    model            : HSMMModel
    streams          : list of (T, 12)  MFCC / delta / delta-delta
    state_seq        : StateInfo のリスト（音素 × 3状態）
    use_duration     : False のとき継続時間分布を無視（HMM モード）
    initial_segments : 第2パス用の初期境界 [(start, end), ...]
    window           : initial_segments からの探索範囲（フレーム数）

    Returns
    -------
    segments : list of (start_frame, end_frame)  各状態の区間（end は exclusive）
    """
    T = streams[0].shape[0]
    N = len(state_seq)

    # --- ユニークな out_idx についてのみフレーム尤度を計算 ---
    unique_out  = sorted({s.out_idx for s in state_seq})
    out_to_row  = {idx: r for r, idx in enumerate(unique_out)}
    n_unique    = len(unique_out)

    # cum_ll_mat[row, :] = 累積対数尤度（行 = unique_out の順）
    cum_ll_mat = np.zeros((n_unique, T + 1), dtype=np.float64)
    for idx in unique_out:
        ll = compute_frame_loglik(model, idx, streams)
        cum_ll_mat[out_to_row[idx], 1:] = np.cumsum(ll)

    # --- 状態ごとの配列を組み立て ---
    floors    = np.array([s.floor for s in state_seq], dtype=np.int32)
    ceils     = np.array([min(s.ceil, T) for s in state_seq], dtype=np.int32)
    out_row   = np.array([out_to_row[s.out_idx] for s in state_seq], dtype=np.int32)
    dur_means = np.array([model.durations[s.dur_idx].mean for s in state_seq], dtype=np.float64)
    dur_vars  = np.array([model.durations[s.dur_idx].var  for s in state_seq], dtype=np.float64)

    # --- t の探索範囲 ---
    t_lo_arr = np.empty(N, dtype=np.int32)
    t_hi_arr = np.empty(N, dtype=np.int32)
    for n, sn in enumerate(state_seq):
        if initial_segments is None:
            t_lo_arr[n] = sn.floor
            t_hi_arr[n] = T
        else:
            e0 = initial_segments[n][1]
            lo = max(sn.floor, e0 - window)
            hi = e0 + window if n < N - 1 else T
            t_lo_arr[n] = lo
            t_hi_arr[n] = min(T, hi)

    # --- Numba JIT カーネルで DP 実行 ---
    score, back = _viterbi_dp(N, T, cum_ll_mat, out_row,
                               floors, ceils, dur_means, dur_vars,
                               t_lo_arr, t_hi_arr, use_duration)

    # --- 強制アライメント: 最後の状態は必ずフレーム T で終わる ---
    if score[N - 1, T] <= -1e30:
        raise RuntimeError("アライメント失敗: 有効なパスが見つかりませんでした。"
                           " 音素列と音声の長さを確認してください。")

    # --- バックトラック ---
    segments = []
    t = T
    for n in range(N - 1, -1, -1):
        d     = int(back[n, t])
        start = t - d
        segments.append((start, t))
        t = start

    segments.reverse()
    return segments


def forced_align_2pass(model: HSMMModel, streams: List[np.ndarray],
                       state_seq: List[StateInfo],
                       window: int = 60) -> List[Tuple[int, int]]:
    """
    2パス強制アライメント（SHIRO 方式）。

    1パス目: HMM モード（継続時間分布なし）で粗いアライメント
    2パス目: HSMM モードで 1パス目の境界付近を精細化

    Parameters
    ----------
    window : 2パス目で 1パス目境界から探索するフレーム幅（デフォルト 60 = 0.3s）
    """
    initial  = forced_align(model, streams, state_seq, use_duration=False)
    refined  = forced_align(model, streams, state_seq, use_duration=True,
                            initial_segments=initial, window=window)
    return refined


# ---------------------------------------------------------------------------
# スタンドアロン実行（動作確認用）
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    from pyshiro.features import extract_mfcc_from_file, HOP_TIME as HT
    from pyshiro.model    import load_hsmm
    from pyshiro.phonemes import load_table, convert_lyric_file

    if len(sys.argv) < 3:
        print('Usage: python shiro/align.py <wav> <lyric.txt>')
        sys.exit(1)

    wav_path = Path(sys.argv[1])
    txt_path = Path(sys.argv[2])

    print('モデル読み込み...')
    model    = load_hsmm('SHIRO-Models-Japanese/intunist-jp6_generic.hsmm')
    phonemap = load_phonemap('SHIRO-Models-Japanese/intunist-jp6_phonemap.json')
    table    = load_table(Path('data/kana2phonemes.table'))

    print('特徴抽出...')
    streams  = extract_mfcc_from_file(wav_path)
    T        = streams[0].shape[0]

    print('音素変換...')
    phonemes = convert_lyric_file(txt_path, table)
    print(f'  音素列 ({len(phonemes)}): {" ".join(phonemes)}')

    print(f'アライメント中... (T={T}, N_states={len(phonemes)*3})')
    state_seq = build_state_sequence(phonemes, phonemap, T)
    segments  = forced_align(model, streams, state_seq)

    # 音素単位で集約（3状態 → 1音素）
    print('\n--- 結果 ---')
    for i, ph in enumerate(phonemes):
        s0, _  = segments[i * 3]
        _, e2  = segments[i * 3 + 2]
        t0 = s0 * HT
        t1 = e2 * HT
        print(f'  {t0:.3f} - {t1:.3f}  {ph}')
