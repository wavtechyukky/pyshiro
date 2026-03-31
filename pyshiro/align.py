"""
align.py

SHIRO 互換 HSMM 強制アライメント（Viterbi デコーディング）

phonemap.json で定義された音素→状態マッピングと
.hsmm モデルの GMM + 継続時間分布を使い、
音素列の各音素がどのフレーム区間に対応するかを推定する。
"""

import json
import math
from dataclasses import dataclass, field
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
    phoneme:       str
    state_idx:     int   # 音素内での状態番号 (0/1/2)
    dur_idx:       int   # モデルの継続時間状態インデックス
    out_idx:       int   # GMM インデックス（全ストリーム共通）
    floor:         int   # 最小継続フレーム数
    ceil:          int   # 最大継続フレーム数（0=無制限）
    pskip_sources: list  = field(default_factory=list)
    # ゼロフレームスキップ: [(src_state_idx, log_p), ...]  ← pskip 用
    topo_sources:  list  = field(default_factory=list)
    # 継続時間保存スキップ: [(src_state_idx, log_p), ...]  ← topology 用


def build_state_sequence(phonemes: List[str], phonemap: dict,
                         total_frames: int) -> List[StateInfo]:
    """
    音素列 → StateInfo のリストに展開する。
    各音素は 3 状態（デフォルト）を持つ。

    スキップエッジ:
    - pskip (ゼロフレーム) : phonemap に "pskip" > 0 が指定された音素を丸ごと省略。
      前の音素の最終状態 → 次の音素の先頭状態（フレーム消費なし）。
    - topology（継続時間保存）: phonemap に "topology" が指定された音素の内部構造。
      - "type-a" (デフォルト): スキップなし
      - "type-b": 各状態 k から最終状態へのスキップ
      - "type-c": 各状態 k から状態 k+2 へのスキップ
      - "skip-boundary": 前の音素最終状態→現音素状態1、現音素状態(nst-2)→次音素先頭
    """
    states: List[StateInfo] = []
    phoneme_starts: List[int] = []

    def _get_entry(ph: str) -> dict:
        """phonemap からエントリを取得する。
        トライフォン 'L-ph+R' が無ければモノフォン 'ph' にフォールバックする。"""
        if ph in phonemap:
            return phonemap[ph]
        # 中心音素を抽出してフォールバック
        core = ph
        if '+' in core:
            core = core.split('+')[0]
        if '-' in core:
            core = core.split('-')[1]
        if core in phonemap:
            return phonemap[core]
        raise ValueError(
            f"音素 '{ph}' が phonemap に存在しません"
            + (f"（モノフォン '{core}' も見つかりません）" if core != ph else "")
        )

    # パス1: 全状態を構築（スキップエッジなし）
    for ph in phonemes:
        entry  = _get_entry(ph)
        nst    = len(entry['states'])
        floors = entry.get('durfloor', [0.0] * nst)
        ceils  = entry.get('durceil',  [0.0] * nst)

        phoneme_starts.append(len(states))

        for k, st in enumerate(entry['states']):
            fl_sec = floors[k] if k < len(floors) else 0.0
            ce_sec = ceils[k]  if k < len(ceils)  else 0.0

            floor_f = max(1, round(fl_sec * FPS)) if fl_sec > 0 else 1
            ceil_f  = round(ce_sec * FPS)          if ce_sec > 0 else total_frames

            states.append(StateInfo(
                phoneme   = ph,
                state_idx = k,
                dur_idx   = st['dur'],
                out_idx   = st['out'][0],
                floor     = floor_f,
                ceil      = ceil_f,
            ))

    # パス2: スキップエッジを設定
    for i, ph in enumerate(phonemes):
        entry    = _get_entry(ph)
        nst      = len(entry['states'])
        ph_start = phoneme_starts[i]

        # --- pskip（ゼロフレームスキップ）---
        # 音素 i に pskip > 0 があれば前の音素最終状態 → 次の音素先頭状態
        pskip = float(entry.get('pskip', 0.0))
        if pskip > 0.0 and 0 < i < len(phonemes) - 1:
            src = phoneme_starts[i] - 1       # 前の音素の最終状態
            dst = phoneme_starts[i + 1]       # 次の音素の先頭状態
            states[dst].pskip_sources.append((src, math.log(max(pskip, 1e-30))))

        # --- topology（継続時間保存スキップ）---
        topology = entry.get('topology', 'type-a')

        if topology == 'type-b':
            # 状態 k → 最終状態（状態 nst-1）へのスキップ（k=0..nst-3）
            # SHIRO: for k=1..nst-2: state[ph_start+k-1].jmp += {d=nst-k}
            for k in range(nst - 2):          # k=0..nst-3 (0-indexed)
                src = ph_start + k
                dst = ph_start + nst - 1
                states[dst].topo_sources.append((src, 0.0))

        elif topology == 'type-c':
            # 状態 k → 状態 k+2 へのスキップ（k=0..nst-3）
            # SHIRO: for k=1..nst-2: state[ph_start+k-1].jmp += {d=2}
            for k in range(nst - 2):
                src = ph_start + k
                dst = ph_start + k + 2
                states[dst].topo_sources.append((src, 0.0))

        elif topology == 'skip-boundary':
            # 前の音素最終状態 → 現音素状態1（現音素の先頭をスキップ）
            # SHIRO: states[#states-nst].jmp += {d=2}
            if i > 0:
                src = phoneme_starts[i] - 1   # 前の音素最終状態
                dst = ph_start + 1            # 現音素状態1
                if dst < len(states):
                    states[dst].topo_sources.append((src, 0.0))
            # 現音素状態(nst-2) → 次の音素先頭（現音素末尾をスキップ）
            # SHIRO: states[#states-1].jmp += {d=2}
            if i + 1 < len(phonemes):
                src = ph_start + nst - 2      # 現音素の後ろから2番目の状態
                dst = phoneme_starts[i + 1]   # 次の音素先頭
                states[dst].topo_sources.append((src, 0.0))

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
                cum_ll_mat,   # (n_unique_out, T+1) float64
                out_row,      # (N,) int32  — cum_ll_mat の行インデックス
                floors,       # (N,) int32
                ceils,        # (N,) int32
                dur_means,    # (N,) float64
                dur_vars,     # (N,) float64
                t_lo_arr,     # (N,) int32
                t_hi_arr,     # (N,) int32
                use_duration,
                topo_src,     # (N, max_topo) int32  -1=番兵  topology スキップ起点
                topo_log,     # (N, max_topo) float64          log(skip prob)
                pskip_src,    # (N, max_pskip) int32 -1=番兵  pskip ゼロフレーム起点
                pskip_log):   # (N, max_pskip) float64         log(pskip)
    """HSMM Viterbi DP の内側ループ（純粋 numpy 配列のみ受け取る）。

    スキップ遷移:
    - topo: 継続時間保存。状態 n が通常の d フレームを消費しつつ、
            n-1 以外の先行状態から遷移する。back_src に先行状態を記録。
    - pskip: ゼロフレーム。フレーム消費なしで src → n に遷移。
             back_d=0 かつ back_src=src で識別する。

    Returns
    -------
    score    : (N, T+1) float64
    back_d   : (N, T+1) int32  — 継続フレーム数（pskip 時は 0）
    back_src : (N, T+1) int32  — 先行状態インデックス（通常時は n-1）
    """
    NEG_INF  = -1e30
    TWO_PI   = 2.0 * math.pi
    max_topo  = topo_src.shape[1]
    max_pskip = pskip_src.shape[1]

    score    = np.full((N, T + 1), NEG_INF)
    back_d   = np.zeros((N, T + 1), dtype=np.int32)
    back_src = np.full((N, T + 1), -1, dtype=np.int32)

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
        score[0, t]    = s
        back_d[0, t]   = d
        back_src[0, t] = -1   # 先行なし（始端）

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
            best_src   = n - 1
            d_max = ce if ce < t else t

            # 通常遷移 + topology スキップ（どちらも d フレームを消費）
            for d in range(fl, d_max + 1):
                prev_t = t - d
                ll = cum_ll_mat[row, t] - cum_ll_mat[row, prev_t]
                if use_dur_n:
                    ll += -0.5 * ((d - dm) ** 2 / dv + log_denom)

                # 前の状態 n-1 から（通常遷移）
                ps = score[n - 1, prev_t]
                if ps > NEG_INF:
                    lp = ps + ll
                    if lp > best_score:
                        best_score = lp
                        best_d     = d
                        best_src   = n - 1

                # topology スキップ: 別の先行状態から同じ d フレーム消費で遷移
                for ki in range(max_topo):
                    src = topo_src[n, ki]
                    if src < 0:
                        break
                    ps = score[src, prev_t]
                    if ps > NEG_INF:
                        lp = ps + topo_log[n, ki] + ll
                        if lp > best_score:
                            best_score = lp
                            best_d     = d
                            best_src   = src

            # pskip ゼロフレームスキップ（フレーム消費なし）
            for ki in range(max_pskip):
                src = pskip_src[n, ki]
                if src < 0:
                    break
                ps = score[src, t]
                if ps > NEG_INF:
                    lp = ps + pskip_log[n, ki]
                    if lp > best_score:
                        best_score = lp
                        best_d     = 0   # ゼロフレーム
                        best_src   = src

            score[n, t]    = best_score
            back_d[n, t]   = best_d
            back_src[n, t] = best_src

    return score, back_d, back_src


# ---------------------------------------------------------------------------
# Viterbi 強制アライメント
# ---------------------------------------------------------------------------

def forced_align(model: HSMMModel, streams: List[np.ndarray],
                 state_seq: List[StateInfo],
                 use_duration: bool = True,
                 initial_segments: List[Tuple[int, int]] = None,
                 window: int = 60,
                 daem_temp: float = 1.0,
                 nodur_phonemes: set = None,
                 hmm_cap: int = None) -> List[Tuple[int, int]]:
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
    daem_temp        : DAEM 温度 (0, 1]。観測対数尤度をこの値でスケール。
                       1.0 のとき通常の Viterbi（デフォルト）。
    nodur_phonemes   : 継続時間モデルを無効化する音素集合（例: {"pau", "br"}）。
                       指定した音素の dur_var を 0 とみなし一様分布扱いにする。
    hmm_cap          : HMM モード（pass 1）での状態あたり最大フレーム数。
                       None（デフォルト）のとき制限なし。訓練の初期段階で
                       収束を助けるために小さい値（例: 200）を指定できる。

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
    # DAEM: 観測対数尤度を daem_temp でスケールして分布を平滑化
    cum_ll_mat = np.zeros((n_unique, T + 1), dtype=np.float64)
    for idx in unique_out:
        ll = compute_frame_loglik(model, idx, streams)
        if daem_temp != 1.0:
            ll = ll * daem_temp
        cum_ll_mat[out_to_row[idx], 1:] = np.cumsum(ll)

    # --- 状態ごとの配列を組み立て ---
    floors    = np.array([s.floor for s in state_seq], dtype=np.int32)
    ceils     = np.array([min(s.ceil, T) for s in state_seq], dtype=np.int32)
    # HMM モード（第1パス）は継続時間を探索しないため上限を制限。
    # これにより計算量が O(T²×N) から O(T×N×cap) に削減される。
    # cap は少なくとも T/N（全状態が T フレームをカバーできる最小値）を確保する。
    # nodur_phonemes の状態は継続時間が不定のためキャップしない。
    if not use_duration and hmm_cap is not None:
        effective_cap = max(hmm_cap, math.ceil(T / N))
        for n in range(N):
            if nodur_phonemes and state_seq[n].phoneme in nodur_phonemes:
                pass  # cap しない
            else:
                ceils[n] = min(ceils[n], effective_cap)
    out_row   = np.array([out_to_row[s.out_idx] for s in state_seq], dtype=np.int32)
    dur_means = np.array([model.durations[s.dur_idx].mean for s in state_seq], dtype=np.float64)
    dur_vars  = np.array([model.durations[s.dur_idx].var  for s in state_seq], dtype=np.float64)
    # nodur_phonemes に含まれる音素の dur_var を 0 にして一様分布扱いにする
    if nodur_phonemes:
        for n, s in enumerate(state_seq):
            if s.phoneme in nodur_phonemes:
                dur_vars[n] = 0.0

    # topo / pskip の 2D 配列を組み立て（-1 で番兵）
    max_topo  = max((len(s.topo_sources)  for s in state_seq), default=0)
    max_pskip = max((len(s.pskip_sources) for s in state_seq), default=0)
    max_topo  = max(max_topo,  1)   # Numba に 0 列配列を渡さないようにする
    max_pskip = max(max_pskip, 1)

    topo_src  = np.full((N, max_topo),  -1, dtype=np.int32)
    topo_log  = np.zeros((N, max_topo),     dtype=np.float64)
    pskip_src = np.full((N, max_pskip), -1, dtype=np.int32)
    pskip_log = np.zeros((N, max_pskip),    dtype=np.float64)

    for n, s in enumerate(state_seq):
        for ki, (src, lp) in enumerate(s.topo_sources):
            topo_src[n, ki]  = src
            topo_log[n, ki]  = lp
        for ki, (src, lp) in enumerate(s.pskip_sources):
            pskip_src[n, ki] = src
            pskip_log[n, ki] = lp

    # --- t の探索範囲 ---
    t_lo_arr = np.empty(N, dtype=np.int32)
    t_hi_arr = np.empty(N, dtype=np.int32)
    for n, sn in enumerate(state_seq):
        if initial_segments is None:
            t_lo_arr[n] = sn.floor
            t_hi_arr[n] = T
        else:
            # nodur_phonemes の状態は pass1 の境界が大幅にずれうるため
            # ウィンドウを使わず全範囲を探索する
            if nodur_phonemes and sn.phoneme in nodur_phonemes:
                t_lo_arr[n] = sn.floor
                t_hi_arr[n] = T
            else:
                e0 = initial_segments[n][1]
                lo = max(sn.floor, e0 - window)
                hi = e0 + window if n < N - 1 else T
                t_lo_arr[n] = lo
                t_hi_arr[n] = min(T, hi)

    # --- Numba JIT カーネルで DP 実行 ---
    score, back_d, back_src = _viterbi_dp(
        N, T, cum_ll_mat, out_row,
        floors, ceils, dur_means, dur_vars,
        t_lo_arr, t_hi_arr, use_duration,
        topo_src, topo_log, pskip_src, pskip_log)

    # --- 強制アライメント: 最後の状態は必ずフレーム T で終わる ---
    best_score = float(score[N - 1, T])
    if best_score <= -1e30:
        raise RuntimeError("アライメント失敗: 有効なパスが見つかりませんでした。"
                           " 音素列と音声の長さを確認してください。")

    # --- バックトラック ---
    # back_d[n,t]=0 かつ back_src[n,t]!=n-1: pskip ゼロフレーム
    # back_d[n,t]>0 かつ back_src[n,t]!=n-1: topology スキップ（中間状態をゼロ埋め）
    # back_d[n,t]>0 かつ back_src[n,t]==n-1: 通常遷移
    segments: List[Tuple[int, int]] = [None] * N  # type: ignore[list-item]
    t = T
    n = N - 1
    while n >= 0:
        d   = int(back_d[n, t])
        src = int(back_src[n, t])

        if d == 0:
            # pskip ゼロフレームスキップ: src → n（フレーム消費なし）
            # src+1..n の全状態に空区間を割り当て
            for k in range(src + 1, n + 1):
                segments[k] = (t, t)
            n = src
        else:
            start = t - d
            segments[n] = (start, t)
            # topology スキップ: src が n-1 でなければ中間状態を空区間に
            for k in range(src + 1, n):
                segments[k] = (start, start)
            n   = src
            t   = start

    return segments, best_score


def forced_align_2pass(model: HSMMModel, streams: List[np.ndarray],
                       state_seq: List[StateInfo],
                       window: int = 60,
                       daem_temp: float = 1.0,
                       nodur_phonemes: set = None,
                       hmm_cap: int = None) -> List[Tuple[int, int]]:
    """
    2パス強制アライメント（SHIRO 方式）。

    1パス目: HMM モード（継続時間分布なし）で粗いアライメント
    2パス目: HSMM モードで 1パス目の境界付近を精細化

    Parameters
    ----------
    window          : 2パス目で 1パス目境界から探索するフレーム幅（デフォルト 60 = 0.3s）
    daem_temp       : DAEM 温度（forced_align に渡す）
    nodur_phonemes  : 継続時間モデルを無効化する音素集合（forced_align に渡す）
    hmm_cap         : HMM pass 1 の状態あたり最大フレーム数（forced_align に渡す）
    """
    initial, _ = forced_align(model, streams, state_seq, use_duration=False,
                              daem_temp=daem_temp, nodur_phonemes=nodur_phonemes,
                              hmm_cap=hmm_cap)
    refined, loglik = forced_align(model, streams, state_seq, use_duration=True,
                                   initial_segments=initial, window=window,
                                   daem_temp=daem_temp, nodur_phonemes=nodur_phonemes,
                                   hmm_cap=hmm_cap)
    return refined, loglik


# ---------------------------------------------------------------------------
# CLI エントリポイント
# ---------------------------------------------------------------------------

def main():
    import argparse
    from pyshiro.features import extract_mfcc_from_file
    from pyshiro.model    import load_hsmm
    from pyshiro.phonemes import convert_lyric_file
    from pyshiro.labels   import segments_to_phoneme_intervals, write_lab

    parser = argparse.ArgumentParser(description="SHIRO HSMM 強制アライメント")
    parser.add_argument("wav",        type=Path, help="入力 WAV ファイル")
    parser.add_argument("lyrics",     type=Path, help="歌詞 .txt ファイル")
    parser.add_argument("--model",    type=Path, required=True,
                        help=".hsmm モデルファイル")
    parser.add_argument("--phonemap", type=Path, required=True,
                        help="phonemap.json ファイル")
    parser.add_argument("--out",      type=Path, default=None,
                        help="出力 .lab ファイル（省略時は WAV と同名）")
    parser.add_argument("--format",   choices=["lab", "textgrid", "audacity"],
                        default="lab", help="出力形式（デフォルト: lab）")
    args = parser.parse_args()

    out_path = args.out or args.wav.with_suffix(
        ".lab" if args.format == "lab" else
        ".TextGrid" if args.format == "textgrid" else ".txt"
    )

    print(f"モデル読み込み: {args.model}", flush=True)
    model    = load_hsmm(args.model)
    phonemap = load_phonemap(args.phonemap)

    print(f"特徴抽出: {args.wav}", flush=True)
    streams  = extract_mfcc_from_file(args.wav)
    T        = streams[0].shape[0]

    print(f"音素変換: {args.lyrics}", flush=True)
    phonemes = convert_lyric_file(args.lyrics)
    print(f"  音素列 ({len(phonemes)}): {' '.join(phonemes)}", flush=True)

    print(f"アライメント中... (T={T}, N={len(phonemes)})", flush=True)
    state_seq = build_state_sequence(phonemes, phonemap, T)
    segments, loglik = forced_align_2pass(model, streams, state_seq,
                                          nodur_phonemes={"pau", "br"})
    print(f"  対数尤度: {loglik / T:.4f} nats/frame", flush=True)

    intervals = segments_to_phoneme_intervals(phonemes, segments)

    if args.format == "lab":
        write_lab(intervals, out_path)
    elif args.format == "textgrid":
        from pyshiro.labels import write_textgrid
        write_textgrid(intervals, out_path)
    else:
        from pyshiro.labels import write_audacity
        write_audacity(intervals, out_path)

    print(f"保存: {out_path}", flush=True)


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
    segments, loglik = forced_align(model, streams, state_seq)
    print(f'  対数尤度: {loglik / T:.4f} nats/frame')

    # 音素単位で集約（3状態 → 1音素）
    print('\n--- 結果 ---')
    for i, ph in enumerate(phonemes):
        s0, _  = segments[i * 3]
        _, e2  = segments[i * 3 + 2]
        t0 = s0 * HT
        t1 = e2 * HT
        print(f'  {t0:.3f} - {t1:.3f}  {ph}')
