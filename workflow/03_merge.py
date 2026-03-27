"""
workflow/03_merge.py

手修正済みのセグメントラベルを 1 曲分の HTK .lab に結合する。

オフセットはサンプル単位の整数演算で適用するため、浮動小数点誤差が生じない。
  HTK 単位 (100ns) = サンプル数 × (10_000_000 // sample_rate)
  ※ 16kHz なら 1 サンプル = 625 HTK 単位

入力ラベル形式の自動判定:
  .TextGrid   → Praat TextGrid として解析
  その他      → HTK .lab (100ns 整数) または 秒単位 .lab を自動判定
                先頭行の開始値が 1000 より大きければ HTK 単位とみなす

使い方:
  python workflow/03_merge.py \\
      --cuts    example/cut_wav/akai_kutsu_cuts.json \\
      --lab_dir example/labels \\
      --out     example/akai_kutsu_merged.lab

オプション:
  --format    入力ラベルの拡張子: lab（デフォルト）/ textgrid / audacity
  --fill_pau  セグメント間の隙間を pau で埋める（デフォルト: True）
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Tuple

# HTK 単位 (100ns) での 1 サンプルあたりの値 (sample_rate ごとに変わる)
def _htk_per_sample(sample_rate: int) -> int:
    return 10_000_000 // sample_rate  # 16kHz → 625


# ---------------------------------------------------------------------------
# ラベル読み込み
# ---------------------------------------------------------------------------

def _read_lab_htk(path: Path) -> List[Tuple[int, int, str]]:
    """HTK .lab (100ns 整数) または 秒単位 .lab を読み込む。
    Returns: [(start_htk, end_htk, phoneme), ...]
    """
    lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines()
             if l.strip()]
    if not lines:
        return []

    # 先頭行の値で単位を判定
    first_val = float(lines[0].split()[0])
    is_htk = first_val > 1000.0

    result = []
    for line in lines:
        parts = line.split()
        if len(parts) < 3:
            continue
        s, e, ph = float(parts[0]), float(parts[1]), parts[2]
        if is_htk:
            result.append((int(s), int(e), ph))
        else:
            # 秒 → HTK 変換（浮動小数点誤差が残る可能性があるが、入力が秒の場合は仕方ない）
            result.append((round(s * 10_000_000), round(e * 10_000_000), ph))
    return result


def _read_textgrid_htk(path: Path) -> List[Tuple[int, int, str]]:
    """Praat TextGrid の最初の IntervalTier を読む。
    Returns: [(start_htk, end_htk, phoneme), ...]
    """
    text = path.read_text(encoding="utf-8")
    xmin_vals = re.findall(r'xmin\s*=\s*([0-9.e+\-]+)', text)
    xmax_vals = re.findall(r'xmax\s*=\s*([0-9.e+\-]+)', text)
    text_vals = re.findall(r'text\s*=\s*"([^"]*)"', text)

    # IntervalTier のヘッダ 2 個（file xmin/xmax, tier xmin/xmax）を除いて
    # intervals の xmin/xmax を取り出す
    # シンプルに: text の数だけ interval がある想定
    n = len(text_vals)
    # xmin_vals[0]=file xmin, [1]=tier xmin, [2..2+n-1]=interval xmin
    # xmax_vals[0]=file xmax, [1]=tier xmax, [2..2+n-1]=interval xmax
    offset = 2
    result = []
    for i in range(n):
        s = float(xmin_vals[offset + i])
        e = float(xmax_vals[offset + i])
        ph = text_vals[i]
        result.append((round(s * 10_000_000), round(e * 10_000_000), ph))
    return result


def _read_audacity_htk(path: Path) -> List[Tuple[int, int, str]]:
    """Audacity ラベル (秒, TAB 区切り) を読む。"""
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split("\t")
        if len(parts) < 2:
            continue
        s, e = float(parts[0]), float(parts[1])
        ph = parts[2] if len(parts) >= 3 else ""
        result.append((round(s * 10_000_000), round(e * 10_000_000), ph))
    return result


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--cuts",     type=Path, required=True,
                        help="{stem}_cuts.json")
    parser.add_argument("--lab_dir",  type=Path, required=True,
                        help="手修正済みラベルのディレクトリ")
    parser.add_argument("--out",      type=Path, required=True,
                        help="出力 HTK .lab パス")
    parser.add_argument("--format",   choices=["lab", "textgrid", "audacity"],
                        default="lab")
    parser.add_argument("--no_fill_pau", action="store_true",
                        help="セグメント間の隙間を pau で埋めない")
    args = parser.parse_args()

    cuts = json.loads(args.cuts.read_text(encoding="utf-8"))
    _EXT = {"lab": ".lab", "textgrid": ".TextGrid", "audacity": ".txt"}
    ext = _EXT[args.format]

    merged: List[Tuple[int, int, str]] = []
    prev_end_htk: int = 0  # 0 から始まるよう初期化（先頭の pau を補填）

    for entry in cuts:
        name        = entry["name"]
        start_samp  = entry["start_sample"]
        sr          = entry["sample_rate"]
        hps         = _htk_per_sample(sr)
        offset_htk  = start_samp * hps  # 整数演算、誤差なし

        lab_path = args.lab_dir / f"{name}{ext}"
        if not lab_path.exists():
            print(f"  [SKIP] ラベルなし: {lab_path.name}")
            continue

        if args.format == "textgrid":
            intervals = _read_textgrid_htk(lab_path)
        elif args.format == "audacity":
            intervals = _read_audacity_htk(lab_path)
        else:
            intervals = _read_lab_htk(lab_path)

        if not intervals:
            print(f"  [SKIP] 空ラベル: {lab_path.name}")
            continue

        # セグメント内での開始オフセット（通常 0 だが TextGrid 等では非 0 の場合もある）
        seg_origin = intervals[0][0]

        # 絶対 HTK 時刻に変換
        abs_intervals = [
            (offset_htk + (s - seg_origin),
             offset_htk + (e - seg_origin),
             ph)
            for s, e, ph in intervals
        ]

        # 前のセグメントとの隙間を pau で埋める
        if not args.no_fill_pau and prev_end_htk is not None:
            gap = abs_intervals[0][0] - prev_end_htk
            if gap > 0:
                # 前後が pau なら結合、そうでなければ新規 pau を挿入
                if merged and merged[-1][2] == "pau":
                    merged[-1] = (merged[-1][0], abs_intervals[0][0], "pau")
                elif abs_intervals[0][2] == "pau":
                    abs_intervals[0] = (prev_end_htk, abs_intervals[0][1], "pau")
                else:
                    merged.append((prev_end_htk, abs_intervals[0][0], "pau"))

        # 先頭・末尾の pau を前後のセグメントと結合
        if merged and merged[-1][2] == "pau" and abs_intervals[0][2] == "pau":
            merged[-1] = (merged[-1][0], abs_intervals[0][1], "pau")
            abs_intervals = abs_intervals[1:]

        merged.extend(abs_intervals)
        prev_end_htk = merged[-1][1]
        print(f"  {name}: {len(intervals)} intervals, "
              f"offset={offset_htk} HTK ({start_samp} samples)")

    if not merged:
        print("結合できるラベルがありませんでした。")
        return

    # 末尾を元WAVの最後まで pau で埋める
    last_entry = cuts[-1]
    if "source_total_samples" in last_entry:
        hps = _htk_per_sample(last_entry["sample_rate"])
        source_end_htk = last_entry["source_total_samples"] * hps
        if merged[-1][1] < source_end_htk:
            if merged[-1][2] == "pau":
                merged[-1] = (merged[-1][0], source_end_htk, "pau")
            else:
                merged.append((merged[-1][1], source_end_htk, "pau"))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # ENUNU .lab 形式（秒単位、小数点以下7桁）で出力
    lines = [f"{s * 1e-7:.7f} {e * 1e-7:.7f} {ph}" for s, e, ph in merged]
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    total_sec = merged[-1][1] * 1e-7
    print(f"\n{len(merged)} intervals, 総時間 {total_sec:.2f}s → {args.out}")


if __name__ == "__main__":
    main()
