"""
workflow/05_check_consonant_outliers.py

同一種類の子音のうち duration が外れ値のものを検出し、
該当箇所の波形と音素ラベルを 1 枚の画像にまとめてプロットする。

短すぎる子音はラベリングに失敗している可能性が高い、という経験則に基づく
チェック用ツール。実用上は短い側（--side short, デフォルト）を見ることが
ほとんど。長い側（--side long）は閉鎖区間込みの正常な伸長が多く、明確な
失敗の信号にはなりにくい。

使い方:
  python workflow/05_check_consonant_outliers.py audio.wav labels.lab
  python workflow/05_check_consonant_outliers.py audio.wav labels.TextGrid \\
      --side short --out plots/cons.png \\
      --iqr_k 1.5 --plosive_iqr_k 3.0 --max_dur_ms 50 --floor_ms 40

入力ラベルは .lab（HTK 100ns / 秒自動判定）/ .TextGrid / .txt(Audacity)。
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))
from pyshiro.labels import read_lab, read_textgrid, read_audacity

# ── 音素分類 ──────────────────────────────────────────────────────────────────

VOWELS         = {"a", "i", "u", "e", "o", "N", "A", "I", "U", "E", "O"}
SILENCE        = {"pau", "sil", "cl", "br", ""}
NON_CONSONANTS = VOWELS | SILENCE


def is_consonant(ph: str) -> bool:
    return bool(ph) and ph not in NON_CONSONANTS


# 破裂音・破擦音（閉鎖を含むため duration が二峰性になりやすく、
# 通常の子音より duration のばらつきが大きい → 大きめの IQR 倍率を使う）
PLOSIVES = {
    "k", "t", "p", "b", "d", "g",
    "ky", "ty", "py", "by", "dy", "gy",
    "ch", "ts", "j",
}


def _ph_color(ph: str, highlight: bool = False) -> str:
    if highlight:
        return "#ff3333"
    if ph in SILENCE:
        return "#cccccc"
    if ph in VOWELS:
        return "#ffaa88"
    return "#88aaff"


# ── ラベル読み込み（拡張子で自動判別） ────────────────────────────────────────

def load_intervals(path: Path) -> List[Tuple[float, float, str]]:
    suf = path.suffix.lower()
    if suf == ".textgrid":
        return read_textgrid(path)
    if suf == ".txt":
        return read_audacity(path)
    return read_lab(path)


# ── 外れ値検出 ────────────────────────────────────────────────────────────────

def detect_outliers(
    intervals: List[Tuple[float, float, str]],
    side: str,
    iqr_k: float,
    plosive_iqr_k: float,
    max_dur_ms: float,
    floor_ms: float,
    skip: set,
) -> List[Tuple[int, str, float, float]]:
    """
    Returns: [(interval_idx, phoneme, duration_sec, ratio), ...]
      ratio = （短い側）median / duration、（長い側）duration / median

    side="short" の判定条件（いずれか）:
      A. duration が floor_ms 未満（絶対的に異常に短い → 無条件でフラグ）
      B. duration が max_dur_ms 未満 かつ 同種子音内で IQR 下側の外れ値
    side="long" の判定条件:
      同種子音内で IQR 上側の外れ値（Q3 + k*IQR を上回る）
    破裂音・破擦音は plosive_iqr_k、それ以外は iqr_k を使う。
    """
    cons = [(i, e - s, ph) for i, (s, e, ph) in enumerate(intervals)
            if is_consonant(ph) and ph not in skip]

    by_ph: dict = defaultdict(list)
    for i, dur, ph in cons:
        by_ph[ph].append((i, dur))

    outliers = []
    for ph, entries in by_ph.items():
        durs   = np.array([d for _, d in entries])
        median = float(np.median(durs))
        k = plosive_iqr_k if ph in PLOSIVES else iqr_k
        if len(entries) >= 4:
            q1, q3 = np.percentile(durs, 25), np.percentile(durs, 75)
            lo = q1 - k * (q3 - q1)
            hi = q3 + k * (q3 - q1)
        else:
            lo, hi = -np.inf, np.inf  # サンプルが少なければ IQR 判定はスキップ
        for idx, dur in entries:
            ms = dur * 1000
            if side == "short":
                cond_a = ms < floor_ms                  # 無条件（異常に短い）
                cond_b = ms < max_dur_ms and dur < lo   # 短い かつ 統計的外れ値
                hit    = cond_a or cond_b
                ratio  = (median / dur) if dur > 0 else float("inf")
            else:  # long
                hit    = dur > hi
                ratio  = (dur / median) if median > 0 else float("inf")
            if hit:
                outliers.append((idx, ph, dur, ratio))

    outliers.sort(key=lambda x: -x[3])  # ratio 降順（より極端なものを先に）
    return outliers


# ── プロット ──────────────────────────────────────────────────────────────────

def plot_all(
    audio: np.ndarray,
    sr: int,
    intervals: List[Tuple[float, float, str]],
    outliers: List[Tuple[int, str, float, float]],
    context_sec: float,
    out_path: Path,
    side: str,
    ncols: int = 4,
) -> None:
    n = len(outliers)
    if n == 0:
        print("外れ値は検出されませんでした。")
        return

    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.0, nrows * 2.4))
    axes = np.atleast_1d(axes).ravel()

    total_dur = len(audio) / sr

    for k, (idx, ph, dur, ratio) in enumerate(outliers):
        ax = axes[k]
        s_out, e_out, _ = intervals[idx]
        center = (s_out + e_out) / 2
        t0 = max(0.0, center - context_sec)
        t1 = min(total_dur, center + context_sec)

        i0, i1 = int(t0 * sr), int(t1 * sr)
        seg = audio[i0:i1]
        tw  = np.linspace(t0, t1, len(seg))

        ax.plot(tw, seg, color="#333333", linewidth=0.4)
        ax.axvspan(s_out, e_out, color="#ff3333", alpha=0.22)
        ax.set_xlim(t0, t1)
        ax.set_yticks([])
        amp = float(np.abs(seg).max()) if len(seg) else 1.0
        ax.set_ylim(-amp * 1.1, amp * 1.1)

        # 可視範囲の音素ラベルを下部にバー表示
        y_bar = -amp * 0.9
        for s, e, p in intervals:
            if s < t1 and e > t0:
                sc, ec = max(s, t0), min(e, t1)
                hl = (s == s_out and e == e_out and p == ph)
                ax.barh(y_bar, ec - sc, left=sc, height=amp * 0.18,
                        color=_ph_color(p, hl), edgecolor="white", linewidth=0.4)
                ax.text((sc + ec) / 2, y_bar, p, ha="center", va="center",
                        fontsize=6, fontweight="bold" if hl else "normal")

        ax.set_title(f"[{ph}] {dur*1000:.0f}ms ({ratio:.1f}x)  @{s_out:.2f}s",
                     fontsize=8, color="#cc0000")
        ax.tick_params(labelsize=6)

    for k in range(n, len(axes)):
        axes[k].axis("off")

    label = "Short" if side == "short" else "Long"
    fig.suptitle(f"{label}-side outlier consonants: {n}  ({out_path.name})",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"保存: {out_path}  ({n} 件)")


# ── メイン ────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("wav",    type=Path, help="WAV ファイル")
    p.add_argument("labels", type=Path, help=".lab / .TextGrid / .txt")
    p.add_argument("--side", choices=["short", "long"], default="short",
                   help="検出する外れ値の側（デフォルト: short）。\n"
                        "short は失敗の信号として有効。long は正常な伸長が多い。")
    p.add_argument("--out",  type=Path, default=None,
                   help="出力 PNG（省略時は <labels名>_<side>_cons.png）")
    p.add_argument("--iqr_k", type=float, default=1.5,
                   help="通常子音の IQR 倍率（小さいほど多く検出, デフォルト 1.5）")
    p.add_argument("--plosive_iqr_k", type=float, default=3.0,
                   help="破裂音・破擦音の IQR 倍率（二峰性のため大きめ, デフォルト 3.0）")
    p.add_argument("--max_dur_ms", type=float, default=50.0,
                   help="[short] この長さ（ms）未満かつ外れ値を検出（デフォルト 50）")
    p.add_argument("--floor_ms", type=float, default=40.0,
                   help="[short] この長さ（ms）未満は無条件でフラグ（デフォルト 40）")
    p.add_argument("--context", type=float, default=0.2,
                   help="前後コンテキスト幅（秒, デフォルト 0.2）")
    p.add_argument("--ncols", type=int, default=4,
                   help="プロットの列数（デフォルト 4）")
    p.add_argument("--skip", type=str, default="",
                   help="検出から除外する音素（カンマ区切り）")
    args = p.parse_args()

    out_path = args.out or Path(f"{args.labels.stem}_{args.side}_cons.png")
    skip = set(s for s in args.skip.split(",") if s) if args.skip else set()

    intervals = load_intervals(args.labels)

    audio, sr = sf.read(str(args.wav), always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)

    outliers = detect_outliers(intervals, args.side, args.iqr_k, args.plosive_iqr_k,
                               args.max_dur_ms, args.floor_ms, skip)

    print(f"子音の{args.side}側外れ値: {len(outliers)} 件")
    for idx, ph, dur, ratio in outliers:
        s = intervals[idx][0]
        print(f"  [{ph:4s}] {dur*1000:6.1f}ms  ({ratio:4.1f}x median)  @ {s:.3f}s")

    plot_all(audio, sr, intervals, outliers, args.context, out_path,
             args.side, args.ncols)


if __name__ == "__main__":
    main()
