"""
workflow/01_cut.py

16kHz モノラル WAV をフレーズ単位のセグメントに分割する。
  - エネルギーベースで無音区間を検出し、長い pau を跨がないように分割
  - セグメントを WAV ファイルとして書き出す
  - cuts.json にオフセット情報を保存（後で 03_merge.py が使用）
  - --lyrics_dir を指定すると、セグメントと同名の空の歌詞テンプレート（.txt）を生成する

使い方:
  python workflow/01_cut.py \\
      --wav_dir    example/wav_16k \\
      --out_dir    example/cut_wav \\
      --lyrics_dir example/lyrics

オプション:
  --max_dur        セグメント最大秒数（デフォルト: 15）
  --min_dur        これより短いセグメントは直前のセグメントに結合（デフォルト: 2）
  --silence_thresh 無音判定の閾値 dB（デフォルト: -50）
  --min_silence    無音と判定する最小継続時間 秒（デフォルト: 0.10）
  --long_silence   セグメント分割点とみなす無音の長さ 秒（デフォルト: 1.0）
  --pad            セグメント前後のパディング 秒（デフォルト: 0.05）
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf


# ---------------------------------------------------------------------------
# エネルギーベースのセグメント検出
# (tests/plot_alignment.py と同じロジック)
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
    max_dur: float = 15.0,
    long_silence: float = 1.0,
    min_split_silence: float = 0.3,
    pad: float = 0.05,
    total_samples: int = 0,
) -> List[Tuple[int, int]]:
    max_samples = int(max_dur * sr)
    long_silence_samples = int(long_silence * sr)
    min_split_samples = int(min_split_silence * sr)
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
                # cl のような短い無音では切らない。min_split_silence 以上の無音でのみ分割。
                if sil.n_samples >= min_split_samples:
                    break
                # 短すぎる無音はmax_durを多少超えても飲み込む
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
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--wav_dir",    type=Path, required=True, help="16kHz WAV ディレクトリ")
    parser.add_argument("--out_dir",    type=Path, required=True, help="セグメント WAV 出力ディレクトリ")
    parser.add_argument("--lyrics_dir", type=Path, default=None,
                        help="歌詞テンプレート（空 .txt）の出力先。省略時は生成しない")
    parser.add_argument("--cuts_dir",   type=Path, default=None,
                        help="{stem}_cuts.json の保存先ディレクトリ（デフォルト: --out_dir と同じ）")
    parser.add_argument("--max_dur",        type=float, default=15.0)
    parser.add_argument("--min_dur",        type=float, default=2.0,
                        help="これより短いセグメントは直前に結合")
    parser.add_argument("--silence_thresh", type=float, default=-50.0)
    parser.add_argument("--min_silence",    type=float, default=0.10)
    parser.add_argument("--long_silence",      type=float, default=1.0)
    parser.add_argument("--min_split_silence", type=float, default=0.3,
                        help="max_dur 超過時に分割を許可する最小無音長（秒）。cl などの短い無音で切れるのを防ぐ")
    parser.add_argument("--pad",               type=float, default=0.05)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cuts_dir = args.cuts_dir or args.out_dir
    cuts_dir.mkdir(parents=True, exist_ok=True)
    if args.lyrics_dir:
        args.lyrics_dir.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(args.wav_dir.glob("*.wav"))
    if not wav_files:
        print(f"WAV ファイルが見つかりません: {args.wav_dir}")
        return

    for wav_path in wav_files:
        wav, sr = sf.read(wav_path, dtype="float32", always_2d=False)
        print(f"\n{wav_path.name}  ({len(wav)/sr:.1f}s, {sr} Hz)")

        regions = _detect_regions(
            wav, sr,
            silence_thresh_db=args.silence_thresh,
            min_silence_dur=args.min_silence,
        )
        raw_segs = _build_segments(
            regions, sr,
            max_dur=args.max_dur,
            long_silence=args.long_silence,
            min_split_silence=args.min_split_silence,
            pad=args.pad,
            total_samples=len(wav),
        )

        # min_dur より短いセグメントを直前に結合
        min_samples = int(args.min_dur * sr)
        segs: List[Tuple[int, int]] = []
        for s, e in raw_segs:
            if segs and (e - s) < min_samples:
                segs[-1] = (segs[-1][0], e)
            else:
                segs.append((s, e))

        print(f"  → {len(segs)} セグメント")

        source_total_samples = len(wav)
        song_cuts = []
        for idx, (s_samp, e_samp) in enumerate(segs, 1):
            name = f"{wav_path.stem}_s{idx:03d}"
            dur = (e_samp - s_samp) / sr
            out_wav = args.out_dir / f"{name}.wav"
            sf.write(out_wav, wav[s_samp:e_samp], sr, subtype="PCM_16")
            print(f"  {name}  {s_samp/sr:.2f}s – {e_samp/sr:.2f}s  ({dur:.1f}s)")

            song_cuts.append({
                "name":                name,
                "source_wav":          wav_path.name,
                "start_sample":        s_samp,
                "end_sample":          e_samp,
                "source_total_samples": source_total_samples,
                "sample_rate":         sr,
                # 参考用（マージには使わない）
                "start_sec":           round(s_samp / sr, 6),
                "end_sec":             round(e_samp / sr, 6),
            })

            if args.lyrics_dir:
                lyrics_path = args.lyrics_dir / f"{name}.txt"
                if not lyrics_path.exists():
                    lyrics_path.write_text("", encoding="utf-8")

        cuts_json_path = cuts_dir / f"{wav_path.stem}_cuts.json"
        cuts_json_path.write_text(
            json.dumps(song_cuts, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  → {cuts_json_path.name} を保存しました")

    if args.lyrics_dir:
        print(f"\n歌詞テンプレートを生成しました: {args.lyrics_dir}")
        print("各 .txt ファイルにひらがなで歌詞を記入してから 02_align.py を実行してください。")

    print("\n完了。")


if __name__ == "__main__":
    main()
