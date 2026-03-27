"""
train.py

SHIRO HSMM 訓練スクリプト

.lab（音素ラベル）+ WAV コーパスから MFCC を抽出し、
Viterbi 訓練（E-step=強制アライメント, M-step=最尤推定）で
.hsmm モデルを学習する。

使い方:
  python -m hsmm.train \\
      --wav_dir  data/kiritan_cut \\
      --lab_dir  data/output_hsmm \\
      --phonemap SHIRO-Models-Japanese/intunist-jp6_phonemap.json \\
      --out      my_model.hsmm \\
      --iters    5

  # 初期モデルを指定する場合（追加学習）:
  python -m hsmm.train ... --init_model SHIRO-Models-Japanese/intunist-jp6_generic.hsmm
"""

import argparse
import json
import math
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import msgpack
import numpy as np

from pyshiro.features import extract_mfcc_from_file
from pyshiro.model import GMM, Duration, HSMMModel, Stream, load_hsmm

# フレームレート（hsmm/align.py と共通）
HOP_TIME = 80 / 16000   # 0.005 s/frame
FPS      = 1.0 / HOP_TIME

# 分散フロア（ゼロ分散防止）
VAR_FLOOR_RATIO = 0.01   # 全体分散の 1%


# ---------------------------------------------------------------------------
# .lab 読み込み
# ---------------------------------------------------------------------------

def read_lab(path: Path) -> List[Tuple[float, float, str]]:
    """ENUNU .lab → [(start_sec, end_sec, phoneme), ...]"""
    ivs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        ivs.append((float(parts[0]), float(parts[1]), parts[2]))
    return ivs


# ---------------------------------------------------------------------------
# フレームを状態に割り当て
# ---------------------------------------------------------------------------

def assign_frames(
    intervals: List[Tuple[float, float, str]],
    phonemap: dict,
    total_frames: int,
) -> Dict[int, List[int]]:
    """
    .lab の音素区間を3状態に3等分してフレームを割り当てる。

    Returns
    -------
    out_idx -> [frame_index, ...] の辞書
    """
    assignment: Dict[int, List[int]] = defaultdict(list)

    for start_sec, end_sec, ph in intervals:
        if ph not in phonemap:
            continue
        entry  = phonemap[ph]
        states = entry["states"]   # 3要素
        nst    = len(states)

        start_f = round(start_sec * FPS)
        end_f   = min(round(end_sec * FPS), total_frames)
        dur     = end_f - start_f
        if dur <= 0:
            continue

        # 3状態に均等分割
        boundaries = [start_f + round(dur * k / nst) for k in range(nst + 1)]

        for k, st in enumerate(states):
            out_idx = st["out"][0]
            for f in range(boundaries[k], boundaries[k + 1]):
                assignment[out_idx].append(f)

    return assignment


# ---------------------------------------------------------------------------
# 並列ワーカー: 1ファイル → 充分統計量
# ---------------------------------------------------------------------------

# SuffStats  : {(stream_idx, out_idx): (count, sum_x, sum_sq)}
# DurStats   : {dur_idx: [frame_counts]}

def _collect_one_file(args):
    """
    ProcessPoolExecutor ワーカー。
    1 ファイルを処理して充分統計量を返す。
    モジュールレベル関数である必要がある（pickle 制約）。
    """
    (wav_path_str, lab_path_str, phonemap,
     model, use_align, use_duration, daem_temp, hmm_cap) = args

    from pyshiro.features import extract_mfcc_from_file

    wav_path = Path(wav_path_str)
    lab_path = Path(lab_path_str)

    streams_feat = extract_mfcc_from_file(wav_path)
    T = streams_feat[0].shape[0]

    intervals = read_lab(lab_path)

    suff: Dict = {}   # (stream_idx, out_idx) -> [(count_m, sum_x_m, sum_sq_m), ...]
    durs: Dict = {}   # dur_idx -> [frame_counts]

    def _accum(stream_idx, out_idx, frames):
        if len(frames) == 0:
            return
        key = (stream_idx, out_idx)
        gmm = model.streams[stream_idx].gmms[out_idx]
        nmix = gmm.nmix
        frames_f = frames.astype(np.float64)

        if nmix == 1:
            entry = [(float(frames_f.shape[0]),
                      frames_f.sum(axis=0),
                      (frames_f ** 2).sum(axis=0))]
        else:
            # E-step: 混合成分の責任度を計算
            N = frames_f.shape[0]
            log_resp = np.zeros((N, nmix))
            for m in range(nmix):
                mu  = gmm.means[m].astype(np.float64)
                var = np.maximum(gmm.vars[m].astype(np.float64), 1e-6)
                diff = frames_f - mu
                log_resp[:, m] = (math.log(max(float(gmm.weights[m]), 1e-30))
                                  - 0.5 * (np.sum(diff ** 2 / var, axis=1)
                                           + np.sum(np.log(2 * math.pi * var))))
            log_sum = np.logaddexp.reduce(log_resp, axis=1, keepdims=True)
            resp = np.exp(log_resp - log_sum)   # (N, nmix)
            entry = []
            for m in range(nmix):
                r = resp[:, m]          # (N,)
                cm  = float(r.sum())
                sxm = (r[:, None] * frames_f).sum(axis=0)
                ssqm = (r[:, None] * frames_f ** 2).sum(axis=0)
                entry.append((cm, sxm, ssqm))

        if key in suff:
            suff[key] = [(c0 + c1, sx0 + sx1, ssq0 + ssq1)
                         for (c0, sx0, ssq0), (c1, sx1, ssq1)
                         in zip(suff[key], entry)]
        else:
            suff[key] = entry

    loglik = 0.0

    if use_align:
        from pyshiro.align import (build_state_sequence,
                                   forced_align, forced_align_2pass)
        phonemes = [ph for _, _, ph in intervals]
        state_seq = build_state_sequence(phonemes, phonemap, T)
        if use_duration:
            segments, loglik = forced_align_2pass(model, streams_feat, state_seq,
                                                  daem_temp=daem_temp,
                                                  hmm_cap=hmm_cap)
        else:
            segments, loglik = forced_align(model, streams_feat, state_seq,
                                            use_duration=False, daem_temp=daem_temp,
                                            hmm_cap=hmm_cap)

        for n, (s, e) in enumerate(segments):
            st_info = state_seq[n]
            out_idx = st_info.out_idx
            dur_idx = st_info.dur_idx
            durs.setdefault(dur_idx, []).append(e - s)
            for stream_idx, feat in enumerate(streams_feat):
                _accum(stream_idx, out_idx, feat[s:e])
    else:
        assign = assign_frames(intervals, phonemap, T)
        out_to_dur = {st["out"][0]: st["dur"]
                      for entry in phonemap.values()
                      for st in entry["states"]}
        for out_idx, frame_list in assign.items():
            dur_idx = out_to_dur.get(out_idx, out_idx)
            durs.setdefault(dur_idx, []).append(len(frame_list))
            idx_arr = np.array(frame_list, dtype=np.int32)
            for stream_idx, feat in enumerate(streams_feat):
                _accum(stream_idx, out_idx, feat[idx_arr])

    return suff, durs, loglik, T


# ---------------------------------------------------------------------------
# M-step: GMM パラメータ推定
# ---------------------------------------------------------------------------

def estimate_gmm(frames: np.ndarray, varfloor: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    nmix=1 GMM の最尤推定。

    Parameters
    ----------
    frames    : (N, ndim)
    varfloor  : (ndim,)

    Returns
    -------
    mean : (ndim,)
    var  : (ndim,)
    """
    if len(frames) == 0:
        return np.zeros(varfloor.shape), varfloor.copy()
    mean = frames.mean(axis=0)
    var  = frames.var(axis=0)
    var  = np.maximum(var, varfloor)
    return mean, var


# ---------------------------------------------------------------------------
# M-step: 継続時間パラメータ推定
# ---------------------------------------------------------------------------

def estimate_duration(
    dur_frames: List[int],
) -> Tuple[float, float]:
    """
    状態の継続フレーム数リストからガウス継続時間分布を推定する。

    Returns
    -------
    mean, var（フレーム単位）
    """
    if not dur_frames:
        return 10.0, 100.0
    arr  = np.array(dur_frames, dtype=np.float64)
    mean = float(arr.mean())
    var  = float(arr.var())
    if var < 1.0:
        var = 1.0   # 最小分散
    return mean, var


# ---------------------------------------------------------------------------
# GMM 混合数増加（分割 + EM）
# ---------------------------------------------------------------------------

def split_gmm(gmm: GMM, perturb: float = 0.2) -> GMM:
    """
    各混合成分を2分割して nmix を倍にする（EM 前の初期化）。
    摂動量は標準偏差の perturb 倍。
    """
    m0 = gmm.means.astype(np.float64)    # (nmix, ndim)
    v0 = gmm.vars.astype(np.float64)
    vf = gmm.varfloors.astype(np.float64)
    w0 = gmm.weights.astype(np.float64)
    nmix, ndim = m0.shape

    new_means   = np.empty((nmix * 2, ndim), dtype=np.float32)
    new_vars    = np.empty((nmix * 2, ndim), dtype=np.float32)
    new_vf      = np.empty((nmix * 2, ndim), dtype=np.float32)
    new_weights = np.empty(nmix * 2,         dtype=np.float32)

    for m in range(nmix):
        delta = perturb * np.sqrt(np.maximum(v0[m], vf[m]))
        new_means[2 * m]     = (m0[m] + delta).astype(np.float32)
        new_means[2 * m + 1] = (m0[m] - delta).astype(np.float32)
        new_vars[2 * m]      = v0[m].astype(np.float32)
        new_vars[2 * m + 1]  = v0[m].astype(np.float32)
        new_vf[2 * m]        = vf[m].astype(np.float32)
        new_vf[2 * m + 1]    = vf[m].astype(np.float32)
        new_weights[2 * m]   = w0[m] / 2
        new_weights[2 * m + 1] = w0[m] / 2

    return GMM(nmix=nmix * 2, ndim=ndim,
               weights=new_weights, means=new_means,
               vars=new_vars, varfloors=new_vf)


def estimate_gmm_em(
    frames: np.ndarray,
    init_gmm: GMM,
    em_iters: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    nmix > 1 GMM の EM 推定。

    Parameters
    ----------
    frames   : (N, ndim)
    init_gmm : 初期 GMM（split_gmm で用意した物）
    em_iters : EM 反復回数

    Returns
    -------
    means   : (nmix, ndim) float32
    vars_   : (nmix, ndim) float32
    weights : (nmix,)      float32
    """
    nmix = init_gmm.nmix
    means   = init_gmm.means.astype(np.float64)
    vars_   = init_gmm.vars.astype(np.float64)
    varfloors = init_gmm.varfloors.astype(np.float64)
    weights = init_gmm.weights.astype(np.float64)
    N, ndim = frames.shape

    for _ in range(em_iters):
        # E-step: log responsibilities (N, nmix)
        log_resp = np.zeros((N, nmix))
        for m in range(nmix):
            v = np.maximum(vars_[m], varfloors[m])
            diff = frames - means[m]
            log_resp[:, m] = (np.log(max(weights[m], 1e-30))
                              - 0.5 * (np.sum(diff ** 2 / v, axis=1)
                                       + np.sum(np.log(2 * math.pi * v))))
        log_sum = np.logaddexp.reduce(log_resp, axis=1, keepdims=True)
        resp = np.exp(log_resp - log_sum)   # (N, nmix)

        # M-step
        Nk = resp.sum(axis=0)
        for m in range(nmix):
            if Nk[m] > 1e-6:
                means[m]  = (resp[:, m:m+1] * frames).sum(axis=0) / Nk[m]
                diff = frames - means[m]
                vars_[m]  = np.maximum(
                    (resp[:, m:m+1] * diff ** 2).sum(axis=0) / Nk[m],
                    varfloors[m],
                )
        weights = np.maximum(Nk / N, 1e-30)
        weights /= weights.sum()

    return means.astype(np.float32), vars_.astype(np.float32), weights.astype(np.float32)


def split_model(model: HSMMModel) -> HSMMModel:
    """モデル内の全 GMM を分割して nmix を倍にした新しいモデルを返す。"""
    new_streams = []
    for stream in model.streams:
        new_gmms = [split_gmm(g) for g in stream.gmms]
        new_streams.append(Stream(weight=stream.weight, gmms=new_gmms))
    return HSMMModel(streams=new_streams, durations=model.durations)


# ---------------------------------------------------------------------------
# モデル初期化（フラットスタート）
# ---------------------------------------------------------------------------

def init_model_flat(
    phonemap: dict,
    ndim: int,
    global_mean: np.ndarray,
    global_var: np.ndarray,
    varfloor: np.ndarray,
    nstream: int = 3,
    stream_weights: List[float] = None,
) -> HSMMModel:
    """
    全状態を全体統計で初期化したモデルを返す。
    アライメント済みデータがある場合は init_model_from_alignment を使う方が良い。
    """
    if stream_weights is None:
        stream_weights = [1.0] * nstream

    # 全ユニーク out_idx / dur_idx を収集して最大値を確認
    all_out = set()
    all_dur = set()
    for entry in phonemap.values():
        for st in entry["states"]:
            all_out.add(st["out"][0])
            all_dur.add(st["dur"])
    n_out = max(all_out) + 1
    n_dur = max(all_dur) + 1

    streams = []
    for w in stream_weights:
        gmms = [GMM(nmix=1, ndim=ndim,
                    weights=np.array([1.0], dtype=np.float32),
                    means=global_mean[np.newaxis].astype(np.float32),
                    vars=global_var[np.newaxis].astype(np.float32),
                    varfloors=varfloor[np.newaxis].astype(np.float32))
                for _ in range(n_out)]
        streams.append(Stream(weight=w, gmms=gmms))

    durations = [Duration(mean=20.0, var=100.0,
                          floor=-1, ceil=-1,
                          fixed_mean=-1, vfloor=0.0)
                 for _ in range(n_dur)]

    return HSMMModel(streams=streams, durations=durations)


# ---------------------------------------------------------------------------
# モデル保存
# ---------------------------------------------------------------------------

def save_hsmm(model: HSMMModel, path: Path) -> None:
    """HSMMModel を .hsmm (MessagePack) に書き出す。"""

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

    def _stream_to_obj(s: Stream):
        return [s.weight, [_gmm_to_obj(g) for g in s.gmms]]

    def _dur_to_obj(d: Duration):
        return [d.mean, d.var, d.floor, d.ceil, d.fixed_mean, d.vfloor]

    obj = [
        [_stream_to_obj(s) for s in model.streams],
        [_dur_to_obj(d)    for d in model.durations],
    ]
    path.write_bytes(msgpack.packb(obj))
    print(f"  保存: {path}", flush=True)


# ---------------------------------------------------------------------------
# 1 エポック分の統計収集
# ---------------------------------------------------------------------------

def collect_stats(
    wav_files: List[Path],
    lab_dir: Path,
    phonemap: dict,
    model: HSMMModel,
    use_align: bool,
    use_duration: bool = True,
    daem_temp: float = 1.0,
    n_jobs: int = 1,
    hmm_cap: int = None,
) -> Tuple[Dict, Dict]:
    """
    全ファイルを処理して、充分統計量と継続フレーム数を返す。

    Returns
    -------
    suff_stats : {(stream_idx, out_idx): (count, sum_x, sum_sq)}
    dur_pool   : {dur_idx: [frame_counts]}

    nmix=1 専用の充分統計量形式。n_jobs>1 で並列処理。
    """
    args_list = []
    n_no_lab = 0
    for wav_path in sorted(wav_files):
        lab_path = lab_dir / (wav_path.stem + ".lab")
        if not lab_path.exists():
            n_no_lab += 1
            continue
        args_list.append((str(wav_path), str(lab_path), phonemap,
                          model, use_align, use_duration, daem_temp, hmm_cap))

    if n_no_lab:
        print(f"  lab なしスキップ: {n_no_lab} 件", flush=True)

    suff_stats: Dict = {}
    dur_pool:   Dict = defaultdict(list)
    n_ok = n_err = 0
    n_total    = len(args_list)
    total_ll   = 0.0
    total_frames = 0
    show_tb = True   # 最初のエラーのみトレースバックを表示

    def _merge(suff, durs, ll, n_frames):
        nonlocal total_ll, total_frames
        for key, entry in suff.items():
            if key in suff_stats:
                suff_stats[key] = [(c0 + c1, sx0 + sx1, ssq0 + ssq1)
                                   for (c0, sx0, ssq0), (c1, sx1, ssq1)
                                   in zip(suff_stats[key], entry)]
            else:
                suff_stats[key] = entry
        for dur_idx, ds in durs.items():
            dur_pool[dur_idx].extend(ds)
        total_ll     += ll
        total_frames += n_frames

    if n_jobs <= 1:
        for i, args in enumerate(args_list):
            if i % 100 == 0:
                print(f"    {i}/{n_total} ({Path(args[0]).name})", flush=True)
            try:
                suff, durs, ll, nf = _collect_one_file(args)
                _merge(suff, durs, ll, nf)
                n_ok += 1
            except Exception as ex:
                import traceback
                if show_tb:
                    print(traceback.format_exc(), flush=True)
                    show_tb = False
                print(f"  スキップ ({Path(args[0]).name}): {ex}", flush=True)
                n_err += 1
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futs = {ex.submit(_collect_one_file, a): a for a in args_list}
            done = 0
            for fut in as_completed(futs):
                done += 1
                if done % 100 == 0:
                    print(f"    {done}/{n_total}", flush=True)
                try:
                    suff, durs, ll, nf = fut.result()
                    _merge(suff, durs, ll, nf)
                    n_ok += 1
                except Exception as ex:
                    import traceback
                    if show_tb:
                        traceback.print_exc()
                        show_tb = False
                    print(f"  スキップ ({Path(futs[fut][0]).name}): {ex}", flush=True)
                    n_err += 1

    print(f"  処理: {n_ok} 件, スキップ: {n_err} 件", flush=True)
    return suff_stats, dur_pool, total_ll, total_frames


# ---------------------------------------------------------------------------
# テスト対数尤度の評価
# ---------------------------------------------------------------------------

def _eval_one_file(args):
    """
    ProcessPoolExecutor ワーカー。
    1 ファイルを HSMM アライメントして (loglik, n_frames) を返す。
    """
    wav_path_str, lab_path_str, phonemap, model = args
    from pyshiro.features import extract_mfcc_from_file
    from pyshiro.align import build_state_sequence, forced_align_2pass

    wav_path = Path(wav_path_str)
    lab_path = Path(lab_path_str)

    streams_feat = extract_mfcc_from_file(wav_path)
    T = streams_feat[0].shape[0]
    intervals = read_lab(lab_path)
    phonemes  = [ph for _, _, ph in intervals]
    state_seq = build_state_sequence(phonemes, phonemap, T)
    _, loglik = forced_align_2pass(model, streams_feat, state_seq)
    return loglik, T


def compute_test_ll(
    test_wav_dir: Path,
    test_lab_dir: Path,
    phonemap: dict,
    model: HSMMModel,
    n_jobs: int = 1,
) -> float:
    """
    テストデータの HSMM 対数尤度（nats/frame）を返す。
    アライメントのみ行い、モデルは更新しない。
    """
    args_list = []
    for wav_path in sorted(test_wav_dir.glob("*.wav")):
        lab_path = test_lab_dir / (wav_path.stem + ".lab")
        if lab_path.exists():
            args_list.append((str(wav_path), str(lab_path), phonemap, model))

    if not args_list:
        print("  テストデータなし", flush=True)
        return float("nan")

    total_ll = 0.0
    total_frames = 0
    n_ok = n_err = 0

    if n_jobs <= 1:
        for args in args_list:
            try:
                ll, nf = _eval_one_file(args)
                total_ll += ll
                total_frames += nf
                n_ok += 1
            except Exception as ex:
                print(f"  テスト評価スキップ ({Path(args[0]).name}): {ex}", flush=True)
                n_err += 1
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futs = {ex.submit(_eval_one_file, a): a for a in args_list}
            for fut in as_completed(futs):
                try:
                    ll, nf = fut.result()
                    total_ll += ll
                    total_frames += nf
                    n_ok += 1
                except Exception:
                    n_err += 1

    if total_frames == 0:
        return float("nan")
    return total_ll / total_frames


# ---------------------------------------------------------------------------
# M-step: モデルパラメータ更新
# ---------------------------------------------------------------------------

def update_model(
    model: HSMMModel,
    suff_stats: Dict,
    dur_pool: Dict[int, List[int]],
    global_varfloors: List[np.ndarray],
) -> HSMMModel:
    """
    充分統計量からモデルパラメータを更新した新しい HSMMModel を返す。

    suff_stats: {(stream_idx, out_idx): [(count_m, sum_x_m, sum_sq_m), ...]}
    nmix=1 および nmix>1 の両方に対応。
    """
    ndim    = model.ndim
    nstream = model.nstream
    n_out   = len(model.streams[0].gmms)

    new_streams = []
    for s_idx, stream in enumerate(model.streams):
        varfloor_s = global_varfloors[s_idx]
        new_gmms = []
        for out_idx in range(n_out):
            key     = (s_idx, out_idx)
            stats   = suff_stats.get(key)   # [(cm, sxm, ssqm), ...]
            gmm_cur = stream.gmms[out_idx]
            nmix    = gmm_cur.nmix

            if stats is not None:
                total_count = sum(cm for cm, _, _ in stats)
                new_weights = np.empty(nmix, dtype=np.float32)
                new_means   = np.empty((nmix, ndim), dtype=np.float32)
                new_vars    = np.empty((nmix, ndim), dtype=np.float32)
                for m, (cm, sxm, ssqm) in enumerate(stats):
                    if cm > 1e-6:
                        mean_m = (sxm / cm).astype(np.float64)
                        var_m  = (ssqm / cm - mean_m ** 2).astype(np.float64)
                        var_m  = np.maximum(var_m, varfloor_s)
                        w_m    = cm / max(total_count, 1e-30)
                    else:
                        mean_m = gmm_cur.means[m].astype(np.float64)
                        var_m  = gmm_cur.vars[m].astype(np.float64)
                        w_m    = float(gmm_cur.weights[m])
                    new_weights[m] = w_m
                    new_means[m]   = mean_m.astype(np.float32)
                    new_vars[m]    = var_m.astype(np.float32)
                # 重みを正規化
                new_weights /= new_weights.sum()
                new_gmms.append(GMM(
                    nmix=nmix, ndim=ndim,
                    weights=new_weights,
                    means=new_means,
                    vars=new_vars,
                    varfloors=gmm_cur.varfloors.copy(),
                ))
            else:
                # データなし: 既存パラメータを保持
                new_gmms.append(gmm_cur)
        new_streams.append(Stream(weight=stream.weight, gmms=new_gmms))

    new_durations = []
    for dur_idx, dur in enumerate(model.durations):
        durs = dur_pool.get(dur_idx, [])
        if durs:
            mean, var = estimate_duration(durs)
        else:
            mean, var = dur.mean, dur.var
        new_durations.append(Duration(
            mean=mean, var=var,
            floor=dur.floor, ceil=dur.ceil,
            fixed_mean=dur.fixed_mean, vfloor=dur.vfloor,
        ))

    return HSMMModel(streams=new_streams, durations=new_durations)


# ---------------------------------------------------------------------------
# メイン訓練ループ
# ---------------------------------------------------------------------------

def train(
    wav_dir:    Path,
    lab_dir:    Path,
    phonemap_path: Path,
    out_path:   Path,
    iters:      int  = 5,
    hmm_iters:  int  = 0,
    daem:       bool = False,
    nmix:       int  = 1,
    init_model: Path = None,
    start_iter: int  = 0,
    n_jobs:     int  = 1,
    cap_relax_iter: int = None,
    test_wav_dir: Path = None,
    test_lab_dir: Path = None,
) -> None:
    """
    HSMM 訓練メインループ。

    Parameters
    ----------
    wav_dir    : WAV ファイルディレクトリ
    lab_dir    : 初期 .lab ファイルディレクトリ（HSMM 出力）
    phonemap_path : phonemap.json
    out_path   : 出力 .hsmm パス
    iters      : EM 反復回数合計（1回目は .lab 固定、2回目以降は再アライメント）
    hmm_iters  : HMM ブートストラップ反復回数（shiro-rest -g 相当）。
                 最初の hmm_iters 回は継続時間分布を無視した HMM モードで訓練し、
                 GMM パラメータを先に収束させてから HSMM に切り替える。
                 0 のとき（デフォルト）はブートストラップなし。
    daem       : True のとき DAEM（Deterministic Annealing EM）を有効化。
                 アライメント反復ごとに温度 T = sqrt(k / n_align) で観測尤度をスケール
                 （k: アライメント反復番号、n_align: アライメント反復総数）。
                 フラットスタートの局所最適を回避する効果がある。
    nmix       : GMM 混合数の上限（1, 2, 4, 8 など2の累乗推奨）。
                 1 より大きい場合、iters を均等に分割して段階的に混合数を倍増する
                 （shiro-stats -n 相当）。
    init_model      : 初期モデル（省略時はフラットスタートで構築）
    cap_relax_iter  : このイテレーション以降、HMM pass1 の cap を 200→1000 に解放する。
                      None のとき常に 200。早期収束後に long tone / long pau を正確に扱う。
    """
    phonemap_raw = json.loads(phonemap_path.read_text(encoding="utf-8"))
    phonemap     = phonemap_raw["phone_map"]

    wav_files = sorted(wav_dir.glob("*.wav"))
    print(f"訓練ファイル数: {len(wav_files)}", flush=True)

    # --- グローバル統計（分散フロア計算用）--- キャッシュあれば再利用
    stats_cache = out_path.with_suffix(".globalstats.npz")
    if stats_cache.exists():
        print(f"グローバル統計キャッシュを読み込み: {stats_cache}", flush=True)
        cache = np.load(stats_cache)
        global_means    = [cache[f"mean_{s}"] for s in range(3)]
        global_vars     = [cache[f"var_{s}"]  for s in range(3)]
    else:
        print("グローバル統計を計算中...", flush=True)
        all_feats = [[] for _ in range(3)]
        for i, wav in enumerate(wav_files):
            if i % 500 == 0:
                print(f"  {i}/{len(wav_files)} ({wav.name})", flush=True)
            try:
                feats = extract_mfcc_from_file(wav)
                for s, f in enumerate(feats):
                    all_feats[s].append(f)
            except Exception:
                pass
        global_means = [np.concatenate(f).mean(axis=0) for f in all_feats]
        global_vars  = [np.concatenate(f).var(axis=0)  for f in all_feats]
        np.savez(stats_cache,
                 **{f"mean_{s}": global_means[s] for s in range(3)},
                 **{f"var_{s}":  global_vars[s]  for s in range(3)})
        print(f"  グローバル統計を保存: {stats_cache}", flush=True)
    global_varfloors = [np.maximum(v * VAR_FLOOR_RATIO, 1e-6) for v in global_vars]

    # --- 初期モデルの準備 ---
    if init_model:
        print(f"初期モデル読み込み: {init_model}", flush=True)
        model = load_hsmm(init_model)
    else:
        print("フラットスタートで初期モデルを構築...", flush=True)
        model = init_model_flat(
            phonemap, ndim=12,
            global_mean=global_means[0],
            global_var=global_vars[0],
            varfloor=global_varfloors[0],
            nstream=3,
            stream_weights=[1.0, 1.0, 1.0],
        )

    # --- GMM 分割スケジュールの計算 ---
    # nmix > 1 の場合、iters を (n_splits+1) 段階に均等分割して段階的に倍増する
    # 例: iters=10, nmix=4 → n_splits=2, 分割は iter 3 と iter 6 の前に実行
    n_splits    = max(int(math.log2(nmix)), 0) if nmix > 1 else 0
    split_iters = set()
    if n_splits > 0:
        step = iters // (n_splits + 1)
        for k in range(1, n_splits + 1):
            split_iters.add(k * step)

    # --- EM 反復 ---
    # イテレーション 0            : .lab 固定（初期化）
    # イテレーション 1..hmm_iters : HMM モード（ブートストラップ）
    # イテレーション hmm_iters+1..: HSMM モード
    # DAEM: アライメント反復 k/n_align に応じて温度 T = sqrt(k/n_align) を付与
    n_align = max(iters - 1, 1)  # アライメントを行う反復数
    align_iter = start_iter - 1 if start_iter > 0 else 0
    prev_ll: float = None  # 前イテレーションの対数尤度（nats/frame）

    log_path = out_path.with_name(out_path.stem + "_log_loglikelihood.txt")
    log_mode = "a" if start_iter > 0 else "w"
    log_file = open(log_path, log_mode, buffering=1)
    log_file.write("iter\ttrain_ll\ttest_ll\n" if log_mode == "w" else "")

    for it in range(start_iter, iters):
        # GMM 分割（指定イテレーション到達時）
        if it in split_iters:
            cur_nmix = model.streams[0].gmms[0].nmix
            print(f"\n--- GMM 分割: nmix {cur_nmix} → {cur_nmix * 2} ---", flush=True)
            model = split_model(model)
        use_align    = (it > 0)
        use_duration = (it > hmm_iters)

        # DAEM 温度: アライメントを行う反復のみカウント
        if use_align:
            align_iter += 1
            daem_temp = math.sqrt(align_iter / n_align) if daem else 1.0
        else:
            daem_temp = 1.0

        if not use_align:
            mode = ".lab 固定"
        elif not use_duration:
            mode = f"HMM ブートストラップ" + (f" (T={daem_temp:.3f})" if daem else "")
        else:
            mode = f"HSMM" + (f" (T={daem_temp:.3f})" if daem else "")
        print(f"\n=== イテレーション {it + 1}/{iters} ({mode}) ===", flush=True)

        if cap_relax_iter is None:
            hmm_cap = None          # cap なし（デフォルト）
        elif it >= cap_relax_iter:
            hmm_cap = None          # cap 解放
        else:
            hmm_cap = 200           # 初期は制約あり
        suff_stats, dur_pool, total_ll, total_frames = collect_stats(
            wav_files, lab_dir, phonemap, model,
            use_align=use_align, use_duration=use_duration, daem_temp=daem_temp,
            n_jobs=n_jobs, hmm_cap=hmm_cap,
        )

        if use_align and total_frames > 0:
            ll_per_frame = total_ll / total_frames
            if prev_ll is not None:
                diff = ll_per_frame - prev_ll
                sign = "+" if diff >= 0 else ""
                print(f"  訓練対数尤度: {ll_per_frame:.4f} nats/frame  "
                      f"(前回比: {sign}{diff:.4f})", flush=True)
            else:
                print(f"  訓練対数尤度: {ll_per_frame:.4f} nats/frame", flush=True)
            prev_ll = ll_per_frame
        else:
            ll_per_frame = float("nan")

        model = update_model(model, suff_stats, dur_pool, global_varfloors)

        # テスト対数尤度
        test_ll = float("nan")
        if test_wav_dir is not None and test_lab_dir is not None:
            test_ll = compute_test_ll(test_wav_dir, test_lab_dir, phonemap, model,
                                      n_jobs=n_jobs)
            print(f"  テスト対数尤度: {test_ll:.4f} nats/frame", flush=True)

        # ログファイルに記録
        log_file.write(f"{it + 1}\t{ll_per_frame:.6f}\t{test_ll:.6f}\n")

        # イテレーションごとにチェックポイントを保存
        ckpt = out_path.with_suffix(f".iter{it + 1}.hsmm")
        save_hsmm(model, ckpt)

    log_file.close()

    # --- 保存 ---
    save_hsmm(model, out_path)
    print(f"\n訓練完了: {out_path}", flush=True)
    print(f"ログ: {log_path}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SHIRO HSMM 訓練")
    parser.add_argument("--wav_dir",   type=Path, required=True,
                        help="WAV ファイルディレクトリ")
    parser.add_argument("--lab_dir",   type=Path, required=True,
                        help="初期 .lab ファイルディレクトリ")
    parser.add_argument("--phonemap",  type=Path,
                        default=Path("SHIRO-Models-Japanese/intunist-jp6_phonemap.json"))
    parser.add_argument("--out",       type=Path, default=Path("my_model.hsmm"),
                        help="出力 .hsmm パス")
    parser.add_argument("--iters",     type=int, default=5,
                        help="EM 反復回数合計（デフォルト: 5）")
    parser.add_argument("--hmm_iters", type=int, default=0,
                        help="HMM ブートストラップ反復回数（デフォルト: 0＝なし）")
    parser.add_argument("--daem", action="store_true",
                        help="DAEM 訓練を有効化（温度アニーリングで局所最適を回避）")
    parser.add_argument("--nmix", type=int, default=1,
                        help="GMM 混合数の上限（1, 2, 4, 8 など。デフォルト: 1）")
    parser.add_argument("--init_model", type=Path, default=None,
                        help="初期モデル（省略時はフラットスタート）")
    parser.add_argument("--start_iter", type=int, default=0,
                        help="再開するイテレーション番号（0始まり）。--init_model と併用")
    parser.add_argument("--jobs", type=int, default=os.cpu_count(),
                        help=f"並列ワーカー数（デフォルト: CPU コア数）")
    parser.add_argument("--cap_relax_iter", type=int, default=None,
                        help="このイテレーション以降 hmm_cap を 200→1000 に緩和（デフォルト: None=常に200）")
    parser.add_argument("--test_wav_dir", type=Path, default=None,
                        help="テスト WAV ディレクトリ（--test_lab_dir と併用でテスト対数尤度を表示）")
    parser.add_argument("--test_lab_dir", type=Path, default=None,
                        help="テスト LAB ディレクトリ（--test_wav_dir と併用でテスト対数尤度を表示）")
    args = parser.parse_args()

    train(
        wav_dir=args.wav_dir,
        lab_dir=args.lab_dir,
        phonemap_path=args.phonemap,
        out_path=args.out,
        iters=args.iters,
        hmm_iters=args.hmm_iters,
        daem=args.daem,
        nmix=args.nmix,
        init_model=args.init_model,
        start_iter=args.start_iter,
        n_jobs=args.jobs,
        cap_relax_iter=args.cap_relax_iter,
        test_wav_dir=args.test_wav_dir,
        test_lab_dir=args.test_lab_dir,
    )


if __name__ == "__main__":
    main()
