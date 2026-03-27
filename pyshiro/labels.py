"""
labels.py

アライメント結果 → ENUNU .lab / Praat TextGrid 出力
"""

from pathlib import Path
from typing import List, Tuple

HOP_SIZE   = 80
SAMPLE_RATE = 16000
HOP_TIME   = HOP_SIZE / SAMPLE_RATE  # 0.005 s/frame

# 時刻の丸め精度: フレーム単位 (5ms) なので小数点以下4桁で十分
_PREC = 4


def _f(sec: float) -> str:
    """秒数を丸めて文字列化（浮動小数点誤差を除去）"""
    return str(round(sec, _PREC))


def segments_to_phoneme_intervals(
    phonemes: List[str],
    segments: List[Tuple[int, int]],
) -> List[Tuple[int, int, str]]:
    """
    3状態×N音素のセグメントリストを音素単位の区間に集約する。
    Returns: [(start_frame, end_frame, phoneme), ...]
    フレーム単位のまま返す（変換は書き出し時のみ）。
    """
    intervals = []
    for i, ph in enumerate(phonemes):
        s0, _  = segments[i * 3]
        _, e2  = segments[i * 3 + 2]
        intervals.append((s0, e2, ph))
    return intervals


# ---------------------------------------------------------------------------
# ENUNU .lab 形式 (秒単位)
# ---------------------------------------------------------------------------

def write_lab(intervals: List[Tuple[int, int, str]],
              out_path: Path,
              htk: bool = False) -> None:
    """
    ENUNU .lab ファイルを書き出す。

    htk=False（デフォルト）: 秒単位（小数点以下7桁）
    htk=True              : HTK 100ns 整数単位（1秒 = 10,000,000）
    """
    ns_per_frame = int(HOP_SIZE * 1e7 / SAMPLE_RATE)  # 500_000
    lines = []
    for start, end, ph in intervals:
        if htk:
            lines.append(f"{start * ns_per_frame} {end * ns_per_frame} {ph}")
        else:
            lines.append(f"{start * HOP_TIME:.7f} {end * HOP_TIME:.7f} {ph}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Praat TextGrid 形式
# ---------------------------------------------------------------------------

def write_textgrid(intervals: List[Tuple[int, int, str]],
                   out_path: Path,
                   tier_name: str = "phoneme") -> None:
    """
    Praat TextGrid (.TextGrid) を書き出す。
    時刻はフレーム→秒変換し、丸めてから書き出す。
    """
    xmin_s = _f(intervals[0][0]  * HOP_TIME)
    xmax_s = _f(intervals[-1][1] * HOP_TIME)
    n      = len(intervals)

    lines = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        "",
        f"xmin = {xmin_s}",
        f"xmax = {xmax_s}",
        "tiers? <exists>",
        "size = 1",
        "item []:",
        "    item [1]:",
        '        class = "IntervalTier"',
        f'        name = "{tier_name}"',
        f"        xmin = {xmin_s}",
        f"        xmax = {xmax_s}",
        f"        intervals: size = {n}",
    ]

    for i, (start, end, ph) in enumerate(intervals, 1):
        lines += [
            f"        intervals [{i}]:",
            f"            xmin = {_f(start * HOP_TIME)}",
            f"            xmax = {_f(end   * HOP_TIME)}",
            f'            text = "{ph}"',
        ]

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Audacity ラベル形式
# ---------------------------------------------------------------------------

def write_audacity(intervals: List[Tuple[int, int, str]],
                   out_path: Path) -> None:
    """
    Audacity ラベルファイル (.txt) を書き出す。
    形式: start_sec <TAB> end_sec <TAB> label  （1行1ラベル）
    """
    lines = []
    for start, end, ph in intervals:
        s = round(start * HOP_TIME, _PREC)
        e = round(end   * HOP_TIME, _PREC)
        lines.append(f"{s}\t{e}\t{ph}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_audacity(path: Path) -> List[Tuple[float, float, str]]:
    """
    Audacity ラベルファイルを読み込む。

    Returns
    -------
    list of (start_sec, end_sec, label)
    """
    intervals = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        start = float(parts[0])
        end   = float(parts[1])
        label = parts[2] if len(parts) >= 3 else ""
        intervals.append((start, end, label))
    return intervals
