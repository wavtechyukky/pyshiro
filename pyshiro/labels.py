"""
labels.py

アライメント結果 → ENUNU .lab / Praat TextGrid 出力
ラベル読み込み → .lab (HTK / 秒自動判定)、Praat TextGrid、Audacity
"""

import re
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
# ENUNU .lab 形式 (HTK: 100ns 単位)
# ---------------------------------------------------------------------------

def write_lab(intervals: List[Tuple[int, int, str]],
              out_path: Path) -> None:
    """
    HTK/ENUNU 標準形式の .lab ファイルを書き出す。
    時刻単位: 100ナノ秒整数 (1秒 = 10,000,000)
    NNSVS / ENUNU / Sinsy / vLabeler に直接渡せる形式。
    """
    ns_per_frame = int(HOP_SIZE * 1e7 / SAMPLE_RATE)  # 500_000
    lines = [f"{s * ns_per_frame} {e * ns_per_frame} {ph}"
             for s, e, ph in intervals]
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


def read_textgrid(path: Path,
                  tier: int = 0) -> List[Tuple[float, float, str]]:
    """
    Praat TextGrid (.TextGrid) の IntervalTier を読み込む。

    Parameters
    ----------
    path : Path
        TextGrid ファイルパス。
    tier : int
        読み込む IntervalTier のインデックス（0 始まり）。複数 tier がある場合に使用。

    Returns
    -------
    list of (start_sec, end_sec, label)
    """
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-16")
    xmin_vals = re.findall(r'xmin\s*=\s*([0-9.e+\-]+)', text)
    xmax_vals = re.findall(r'xmax\s*=\s*([0-9.e+\-]+)', text)
    text_vals = re.findall(r'text\s*=\s*"([^"]*)"', text)

    # tier ヘッダ数（file xmin/xmax + 各 tier xmin/xmax）を飛ばして interval を取得
    # tier ごとに xmin/xmax が 1 ペアずつ先頭にある
    n = len(text_vals)
    offset = 2 + tier  # file(1) + tier headers up to target tier
    intervals = []
    for i in range(n):
        s = float(xmin_vals[offset + i])
        e = float(xmax_vals[offset + i])
        intervals.append((s, e, text_vals[i]))
    return intervals


def read_lab(path: Path) -> List[Tuple[float, float, str]]:
    """
    HTK .lab または秒単位 .lab を読み込む。

    先頭行の終了値が 1000 より大きければ HTK 100ns 整数とみなして秒に変換する。

    Returns
    -------
    list of (start_sec, end_sec, label)
    """
    lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines()
             if l.strip()]
    if not lines:
        return []
    first_parts = lines[0].split()
    first_end = float(first_parts[1]) if len(first_parts) >= 2 else 0.0
    is_htk = first_end > 1000.0
    intervals = []
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
