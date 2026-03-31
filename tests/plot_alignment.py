"""
tests/plot_alignment.py

WAV + GT lab ディレクトリを受け取り、アライメント結果を視覚的に確認するスクリプト。
各 WAV ファイルをエネルギーベースで短いチャンクに分割し、
波形・メルスペクトログラム・正解ラベル（GT）・推定ラベル（Est）を PNG で出力する。

使い方:
  python tests/plot_alignment.py \
      --wav_dir  /path/to/test/wav \
      --lab_dir  /path/to/test/lab \
      --model    ckpt/pyshiro-jp-no-natsume.hsmm \
      --phonemap ckpt/pyshiro-jp_phonemap.json \
      --out_dir  test_plots
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


# ---------------------------------------------------------------------------
# 音声エネルギーベースのセグメント分割
# ---------------------------------------------------------------------------

@dataclass
class _Region:
    start: int
    end: int
    is_silence: bool

    @property
    def n_samples(self) -> int:
        return self.end - self.start


def _detect_regions(
    audio: np.ndarray,
    sr: int,
    silence_thresh_db: float = -50.0,
    min_silence_dur: float = 0.10,
    frame_dur: float = 0.01,
) -> List[_Region]:
    frame_size = max(1, int(sr * frame_dur))
    min_silence_frames = max(1, round(min_silence_dur / frame_dur))
    silent_flags = []
    pos = 0
    while pos < len(audio):
        frame = audio[pos: pos + frame_size]
        rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
        db = 20.0 * np.log10(max(rms, 1e-10))
        silent_flags.append(db < silence_thresh_db)
        pos += frame_size

    if not silent_flags:
        return []

    raw = []
    seg_start = 0
    for i in range(1, len(silent_flags) + 1):
        if i == len(silent_flags) or silent_flags[i] != silent_flags[seg_start]:
            raw.append((seg_start, i, silent_flags[seg_start]))
            seg_start = i

    merged = [(s, e, False) if is_sil and (e - s) < min_silence_frames else (s, e, is_sil)
              for s, e, is_sil in raw]

    coalesced = []
    for item in merged:
        if coalesced and coalesced[-1][2] == item[2]:
            coalesced[-1] = (coalesced[-1][0], item[1], item[2])
        else:
            coalesced.append(list(item))

    regions = []
    n_total = len(audio)
    for idx, (sf_i, ef_i, is_sil) in enumerate(coalesced):
        start_s = sf_i * frame_size
        end_s = ef_i * frame_size if idx < len(coalesced) - 1 else n_total
        end_s = min(end_s, n_total)
        if start_s < end_s:
            regions.append(_Region(start=start_s, end=end_s, is_silence=is_sil))
    return regions


def _build_segments(
    regions: List[_Region],
    sr: int,
    max_dur: float = 10.0,
    long_silence: float = 1.0,
    pad: float = 0.05,
    total_samples: int = 0,
) -> List[Tuple[int, int]]:
    max_samples = int(max_dur * sr)
    long_silence_samples = int(long_silence * sr)
    pad_samples = int(pad * sr)
    segments = []
    i = 0

    while i < len(regions) and regions[i].is_silence:
        i += 1

    while i < len(regions):
        if regions[i].is_silence:
            i += 1
            continue

        seg_voiced_start = regions[i].start
        seg_voiced_end = regions[i].end
        seg_len = regions[i].n_samples
        i += 1

        while i < len(regions):
            if not regions[i].is_silence:
                seg_voiced_end = regions[i].end
                seg_len += regions[i].n_samples
                i += 1
                continue
            sil = regions[i]
            if sil.n_samples >= long_silence_samples:
                break
            if i + 1 >= len(regions):
                break
            next_voiced = regions[i + 1]
            if next_voiced.is_silence:
                i += 1
                continue
            next_len = sil.n_samples + next_voiced.n_samples
            if seg_len + next_len > max_samples:
                break
            seg_voiced_end = next_voiced.end
            seg_len += next_len
            i += 2

        padded_start = max(0, seg_voiced_start - pad_samples)
        padded_end = min(total_samples, seg_voiced_end + pad_samples)
        if padded_end > padded_start:
            segments.append((padded_start, padded_end))

        while i < len(regions) and regions[i].is_silence:
            i += 1

    return segments


# ---------------------------------------------------------------------------
# ラベル読み込み / チャンク抽出
# ---------------------------------------------------------------------------

def _read_lab(path: Path) -> List[Tuple[float, float, str]]:
    """HTK (.lab)、秒単位 .lab、TextGrid、Audacity (.txt) を自動判定して読む。
    返り値は常に秒単位の (start, end, phoneme) リスト。
    """
    import re as _re

    text = path.read_text(encoding="utf-8")

    # --- Praat TextGrid ---
    if 'File type = "ooTextFile"' in text or path.suffix.lower() == ".textgrid":
        xmin_vals = _re.findall(r'xmin\s*=\s*([0-9.e+\-]+)', text)
        xmax_vals = _re.findall(r'xmax\s*=\s*([0-9.e+\-]+)', text)
        text_vals = _re.findall(r'text\s*=\s*"([^"]*)"', text)
        n = len(text_vals)
        intervals = []
        for i in range(n):
            s = float(xmin_vals[2 + i])
            e = float(xmax_vals[2 + i])
            intervals.append((s, e, text_vals[i]))
        return intervals

    intervals = []

    # --- Audacity: TAB 区切り ---
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines and "\t" in lines[0]:
        for line in lines:
            parts = line.split("\t")
            if len(parts) >= 2:
                s, e = float(parts[0]), float(parts[1])
                ph = parts[2] if len(parts) >= 3 else ""
                intervals.append((s, e, ph))
        return intervals

    # --- HTK / 秒単位 .lab: スペース区切り ---
    # 先頭が 0 の場合は 2列目の終了時刻で判定
    first_parts = lines[0].split() if lines else []
    first_val = float(first_parts[1]) if len(first_parts) >= 2 else 0.0
    is_htk = first_val > 1000.0
    for line in lines:
        parts = line.split()
        if len(parts) < 3:
            continue
        s, e = float(parts[0]), float(parts[1])
        if is_htk:
            s *= 1e-7
            e *= 1e-7
        intervals.append((s, e, parts[2]))
    return intervals


def _clip_intervals(intervals, t_start: float, t_end: float):
    result = []
    for s, e, ph in intervals:
        if e <= t_start or s >= t_end:
            continue
        result.append((max(s, t_start) - t_start, min(e, t_end) - t_start, ph))
    return result


# ---------------------------------------------------------------------------
# アライメント
# ---------------------------------------------------------------------------

HOP_TIME = 80 / 16000  # pyshiro の MFCC ホップ幅（秒）


def _align(wav_arr: np.ndarray, sr: int, gt_intervals, model, phonemap):
    from pyshiro.features import extract_mfcc
    from pyshiro.align import build_state_sequence, forced_align_2pass

    phonemes = [ph for _, _, ph in gt_intervals]
    streams = extract_mfcc(wav_arr, sr)
    T = streams[0].shape[0]
    state_seq = build_state_sequence(phonemes, phonemap, T)
    segments, _ = forced_align_2pass(model, streams, state_seq,
                                     nodur_phonemes={"pau", "br"},
                                     hmm_cap=1000)

    result = []
    i = 0
    while i < len(state_seq):
        ph = state_seq[i].phoneme
        nst = 0
        # state_idx==0 は音素インスタンスの先頭。同名音素の連続を誤合算しないよう
        # 最初の状態から始まり、次の音素先頭（state_idx==0）が来たら止める。
        while i + nst < len(state_seq) and state_seq[i + nst].phoneme == ph:
            nst += 1
            if i + nst < len(state_seq) and state_seq[i + nst].state_idx == 0:
                break
        s_frame = segments[i][0]
        e_frame = segments[i + nst - 1][1]
        result.append((s_frame * HOP_TIME, e_frame * HOP_TIME, ph))
        i += nst
    return result


# ---------------------------------------------------------------------------
# メルスペクトログラム
# ---------------------------------------------------------------------------

def _mel_filterbank(sr, n_fft, n_mels, fmin=0.0, fmax=None):
    if fmax is None:
        fmax = sr / 2.0

    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_min, mel_max = hz_to_mel(fmin), hz_to_mel(fmax)
    mel_pts = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)
    bin_pts = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    n_bins = n_fft // 2 + 1
    fb = np.zeros((n_mels, n_bins))
    for m in range(1, n_mels + 1):
        fl, fc, fr = bin_pts[m - 1], bin_pts[m], bin_pts[m + 1]
        for k in range(fl, fc):
            if fc > fl:
                fb[m - 1, k] = (k - fl) / (fc - fl)
        for k in range(fc, fr):
            if fr > fc:
                fb[m - 1, k] = (fr - k) / (fr - fc)
    return fb


def _compute_melspec(wav, sr, n_fft=1024, hop=256, n_mels=80):
    win = np.hanning(n_fft)
    n_frames = (len(wav) - n_fft) // hop + 1
    if n_frames <= 0:
        return np.zeros((n_mels, 1))
    frames = np.stack([wav[i * hop: i * hop + n_fft] * win for i in range(n_frames)])
    spec = np.abs(np.fft.rfft(frames, n=n_fft)) ** 2
    fb = _mel_filterbank(sr, n_fft, n_mels)
    return np.log10(np.maximum(fb @ spec.T, 1e-10)) * 10


# ---------------------------------------------------------------------------
# プロット
# ---------------------------------------------------------------------------

_PALETTE = plt.cm.tab20.colors
_COLORS = {}


def _ph_color(ph):
    if ph not in _COLORS:
        _COLORS[ph] = _PALETTE[len(_COLORS) % len(_PALETTE)]
    return _COLORS[ph]


def _draw_row(ax, intervals, label, duration):
    ax.set_xlim(0, duration)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel(label, fontsize=8, rotation=0, labelpad=36, va="center")
    seen = set()
    for s, e, ph in intervals:
        color = _ph_color(ph)
        ax.add_patch(mpatches.FancyBboxPatch(
            (s, 0.05), e - s, 0.9,
            boxstyle="square,pad=0",
            facecolor=(*color[:3], 0.55),
            edgecolor="white", linewidth=0.5,
        ))
        ax.text((s + e) / 2, 0.5, ph,
                ha="center", va="center",
                fontsize=max(4, min(8, (e - s) * 60)),
                color="black", clip_on=True)
        for t in (s, e):
            if t not in seen:
                ax.axvline(t, color="white", linewidth=0.4, alpha=0.7)
                seen.add(t)


def _plot_chunk(wav_arr, sr, gt_intervals, est_intervals, title, out_path):
    duration = len(wav_arr) / sr
    melspec = _compute_melspec(wav_arr, sr)

    fig, axes = plt.subplots(
        4, 1,
        figsize=(max(10, duration * 1.5), 6),
        gridspec_kw={"height_ratios": [1.2, 2, 0.6, 0.6]},
    )
    ax_wav, ax_mel, ax_gt, ax_est = axes

    times = np.linspace(0, duration, len(wav_arr))
    ax_wav.plot(times, wav_arr, color="steelblue", linewidth=0.4, rasterized=True)
    ax_wav.set_xlim(0, duration)
    ax_wav.set_ylabel("Wav", fontsize=8)
    ax_wav.set_yticks([])
    ax_wav.tick_params(labelbottom=False)

    ax_mel.imshow(melspec, aspect="auto", origin="lower",
                  extent=[0, duration, 0, melspec.shape[0]],
                  cmap="magma", interpolation="nearest")
    ax_mel.set_ylabel("Mel", fontsize=8)
    ax_mel.set_yticks([])
    ax_mel.tick_params(labelbottom=False)

    _draw_row(ax_gt,  gt_intervals,  "GT",  duration)
    _draw_row(ax_est, est_intervals, "Est", duration)

    for ax in (ax_wav, ax_mel, ax_gt):
        ax.tick_params(labelbottom=False)
    ax_est.set_xlabel("Time (s)", fontsize=8)

    fig.suptitle(title, fontsize=10, y=1.01)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}", flush=True)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wav_dir",   type=Path, required=True, help="WAV ディレクトリ")
    parser.add_argument("--lab_dir",   type=Path, required=True, help="GT lab ディレクトリ")
    parser.add_argument("--model",     type=Path, required=True, help=".hsmm モデル")
    parser.add_argument("--phonemap",  type=Path, required=True, help="phonemap.json")
    parser.add_argument("--out_dir",   type=Path, default=Path("test_plots"))
    parser.add_argument("--max_dur",   type=float, default=10.0, help="チャンク最大秒数")
    parser.add_argument("--silence_thresh", type=float, default=-50.0)
    parser.add_argument("--min_silence_dur", type=float, default=0.10)
    parser.add_argument("--long_silence", type=float, default=1.0)
    parser.add_argument("--pad",       type=float, default=0.05)
    args = parser.parse_args()

    from pyshiro.model import load_hsmm
    from pyshiro.align import load_phonemap

    print("モデル読み込み...", flush=True)
    model   = load_hsmm(args.model)
    phonemap = load_phonemap(args.phonemap)

    wav_files = sorted(args.wav_dir.glob("*.wav"))
    print(f"{len(wav_files)} ファイルを処理中...", flush=True)

    for wav_path in wav_files:
        # .lab / .TextGrid / .txt を順に探す
        lab_path = None
        for ext in (".lab", ".TextGrid", ".txt"):
            candidate = args.lab_dir / (wav_path.stem + ext)
            if candidate.exists():
                lab_path = candidate
                break
        if lab_path is None:
            print(f"  [SKIP] lab なし: {wav_path.name}", flush=True)
            continue

        wav_full, sr = sf.read(wav_path, dtype="float32")
        gt_all = _read_lab(lab_path)

        regions = _detect_regions(wav_full, sr,
                                  silence_thresh_db=args.silence_thresh,
                                  min_silence_dur=args.min_silence_dur)
        chunks = _build_segments(regions, sr,
                                 max_dur=args.max_dur,
                                 long_silence=args.long_silence,
                                 pad=args.pad,
                                 total_samples=len(wav_full))

        print(f"\n{wav_path.stem} → {len(chunks)} チャンク", flush=True)

        for ci, (s_samp, e_samp) in enumerate(chunks):
            wav_chunk = wav_full[s_samp:e_samp]
            t_start, t_end = s_samp / sr, e_samp / sr

            gt_chunk = _clip_intervals(gt_all, t_start, t_end)
            if not gt_chunk:
                continue

            title = f"{wav_path.stem}_c{ci + 1:02d}  ({t_start:.1f}s–{t_end:.1f}s)"
            print(f"  チャンク {ci + 1}/{len(chunks)}: {t_end - t_start:.1f}s", flush=True)

            try:
                est_intervals = _align(wav_chunk, sr, gt_chunk, model, phonemap)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"  [ERR] {e}", flush=True)
                est_intervals = gt_chunk

            out_path = args.out_dir / f"{wav_path.stem}_c{ci + 1:02d}.png"
            _plot_chunk(wav_chunk, sr, gt_chunk, est_intervals, title, out_path)

    print(f"\n完了。出力先: {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
