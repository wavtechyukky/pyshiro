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
from collections import defaultdict
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
        ivs.append((int(parts[0]) / 1e7, int(parts[1]) / 1e7, parts[2]))
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

    # 全ユニーク out_idx を収集して最大値を確認
    all_out = set()
    for entry in phonemap.values():
        for st in entry["states"]:
            all_out.add(st["out"][0])
    n_out = max(all_out) + 1

    gmm_proto = GMM(
        nmix=1, ndim=ndim,
        weights=np.array([1.0], dtype=np.float32),
        means=global_mean[np.newaxis].astype(np.float32),
        vars_=global_var[np.newaxis].astype(np.float32),
        varfloors=varfloor[np.newaxis].astype(np.float32),
    )
    # dataclass フィールド名を合わせる
    gmm_proto = GMM(
        nmix=1, ndim=ndim,
        weights=np.array([1.0], dtype=np.float32),
        means=global_mean[np.newaxis].astype(np.float32),
        vars=global_var[np.newaxis].astype(np.float32),
        varfloors=varfloor[np.newaxis].astype(np.float32),
    )

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
                 for _ in range(n_out)]

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
    print(f"  保存: {path}")


# ---------------------------------------------------------------------------
# 1 エポック分の統計収集
# ---------------------------------------------------------------------------

def collect_stats(
    wav_files: List[Path],
    lab_dir: Path,
    phonemap: dict,
    model: HSMMModel,
    use_align: bool,
) -> Tuple[Dict[int, List[np.ndarray]], Dict[int, List[int]]]:
    """
    全ファイルを処理して、状態ごとのフレーム集合と継続フレーム数を返す。

    Parameters
    ----------
    use_align : True のとき強制アライメントで境界を更新（2回目以降）
                False のとき .lab の境界をそのまま使う（初回）
    """
    from pyshiro.align import (build_state_sequence, forced_align_2pass,
                             load_phonemap)
    from pyshiro.phonemes import convert_lyric_file, load_table

    frame_pool: Dict[int, List[np.ndarray]] = defaultdict(list)
    dur_pool:   Dict[int, List[int]]         = defaultdict(list)

    n_ok = n_skip = 0

    for wav_path in sorted(wav_files):
        lab_path = lab_dir / (wav_path.stem + ".lab")
        if not lab_path.exists():
            n_skip += 1
            continue

        try:
            streams_feat = extract_mfcc_from_file(wav_path)
            T = streams_feat[0].shape[0]
            intervals = read_lab(lab_path)

            if use_align:
                # アライメントで境界を更新
                phonemes = [ph for _, _, ph in intervals]
                state_seq = build_state_sequence(phonemes, phonemap, T)
                segments  = forced_align_2pass(model, streams_feat, state_seq)

                # 状態ごとにフレームを収集
                for n, (s, e) in enumerate(segments):
                    out_idx = state_seq[n].out_idx
                    dur_pool[out_idx].append(e - s)
                    for stream_idx, feat in enumerate(streams_feat):
                        for f in range(s, e):
                            frame_pool[(stream_idx, out_idx)].append(feat[f])
            else:
                # .lab の境界を3等分して収集
                assign = assign_frames(intervals, phonemap, T)
                for out_idx, frame_list in assign.items():
                    # 継続フレーム数（音素単位の概算）
                    dur_pool[out_idx].append(len(frame_list))
                    for f in frame_list:
                        for stream_idx, feat in enumerate(streams_feat):
                            frame_pool[(stream_idx, out_idx)].append(feat[f])

            n_ok += 1
        except Exception as ex:
            print(f"  スキップ ({wav_path.name}): {ex}")
            n_skip += 1

    print(f"  処理: {n_ok} 件, スキップ: {n_skip} 件")
    return frame_pool, dur_pool


# ---------------------------------------------------------------------------
# M-step: モデルパラメータ更新
# ---------------------------------------------------------------------------

def update_model(
    model: HSMMModel,
    frame_pool: Dict,
    dur_pool: Dict[int, List[int]],
    global_varfloors: List[np.ndarray],
) -> HSMMModel:
    """統計からモデルパラメータを更新した新しい HSMMModel を返す。"""
    ndim    = model.ndim
    nstream = model.nstream
    n_out   = len(model.streams[0].gmms)

    new_streams = []
    for s_idx, stream in enumerate(model.streams):
        varfloor_s = global_varfloors[s_idx]
        new_gmms = []
        for out_idx in range(n_out):
            key    = (s_idx, out_idx)
            frames = frame_pool.get(key, [])
            if frames:
                arr        = np.stack(frames)
                mean, var  = estimate_gmm(arr, varfloor_s)
                varfloor   = stream.gmms[out_idx].varfloors[0].copy()
            else:
                # データなし: 既存パラメータを保持
                mean     = stream.gmms[out_idx].means[0].copy()
                var      = stream.gmms[out_idx].vars[0].copy()
                varfloor = stream.gmms[out_idx].varfloors[0].copy()

            new_gmms.append(GMM(
                nmix=1, ndim=ndim,
                weights=np.array([1.0], dtype=np.float32),
                means=mean[np.newaxis].astype(np.float32),
                vars=var[np.newaxis].astype(np.float32),
                varfloors=varfloor[np.newaxis].astype(np.float32),
            ))
        new_streams.append(Stream(weight=stream.weight, gmms=new_gmms))

    new_durations = []
    for out_idx, dur in enumerate(model.durations):
        durs = dur_pool.get(out_idx, [])
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
    init_model: Path = None,
) -> None:
    """
    HSMM 訓練メインループ。

    Parameters
    ----------
    wav_dir   : WAV ファイルディレクトリ
    lab_dir   : 初期 .lab ファイルディレクトリ（HSMM 出力）
    phonemap_path : phonemap.json
    out_path  : 出力 .hsmm パス
    iters     : EM 反復回数（1回目は .lab 固定、2回目以降は再アライメント）
    init_model: 初期モデル（省略時はフラットスタートで構築）
    """
    phonemap_raw = json.loads(phonemap_path.read_text(encoding="utf-8"))
    phonemap     = phonemap_raw["phone_map"]

    wav_files = sorted(wav_dir.glob("*.wav"))
    print(f"訓練ファイル数: {len(wav_files)}")

    # --- グローバル統計（分散フロア計算用）---
    print("グローバル統計を計算中...")
    all_feats = [[] for _ in range(3)]
    for wav in wav_files:
        try:
            feats = extract_mfcc_from_file(wav)
            for s, f in enumerate(feats):
                all_feats[s].append(f)
        except Exception:
            pass
    global_means    = [np.concatenate(f).mean(axis=0) for f in all_feats]
    global_vars     = [np.concatenate(f).var(axis=0)  for f in all_feats]
    # ストリームごとの分散フロア（各 (12,)）
    global_varfloors = [np.maximum(v * VAR_FLOOR_RATIO, 1e-6) for v in global_vars]

    # --- 初期モデルの準備 ---
    if init_model:
        print(f"初期モデル読み込み: {init_model}")
        model = load_hsmm(init_model)
    else:
        print("フラットスタートで初期モデルを構築...")
        model = init_model_flat(
            phonemap, ndim=12,
            global_mean=global_means[0],
            global_var=global_vars[0],
            varfloor=global_varfloors[0],
            nstream=3,
            stream_weights=[1.0, 1.0, 1.0],
        )

    # --- EM 反復 ---
    for it in range(iters):
        use_align = (it > 0)   # 初回は .lab 固定
        mode = "再アライメント" if use_align else ".lab 固定"
        print(f"\n=== イテレーション {it + 1}/{iters} ({mode}) ===")

        frame_pool, dur_pool = collect_stats(
            wav_files, lab_dir, phonemap, model, use_align=use_align
        )
        model = update_model(model, frame_pool, dur_pool, global_varfloors)

    # --- 保存 ---
    save_hsmm(model, out_path)
    print(f"\n訓練完了: {out_path}")


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
                        help="EM 反復回数（デフォルト: 5）")
    parser.add_argument("--init_model", type=Path, default=None,
                        help="初期モデル（省略時はフラットスタート）")
    args = parser.parse_args()

    train(
        wav_dir=args.wav_dir,
        lab_dir=args.lab_dir,
        phonemap_path=args.phonemap,
        out_path=args.out,
        iters=args.iters,
        init_model=args.init_model,
    )


if __name__ == "__main__":
    main()
