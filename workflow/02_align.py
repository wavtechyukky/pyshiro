"""
workflow/02_align.py

カット済みセグメントに対して強制アライメントを実行し、ラベルファイルを生成する。

使い方:
  python workflow/02_align.py \\
      --wav_dir    example/cut_wav \\
      --lyrics_dir example/lyrics \\
      --model      models/intunist-jp6_generic.hsmm \\
      --phonemap   models/intunist-jp6_phonemap.json \\
      --out_dir    example/labels

オプション:
  --format    出力形式: lab（デフォルト）/ textgrid / audacity
  --overwrite 既存のラベルを上書きする（デフォルト: スキップ）
  --jobs      並列ワーカー数（デフォルト: CPU コア数）
"""

import argparse
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


def _align_one(wav_path: Path, lyrics_path: Path, model_path: Path,
               phonemap_path: Path, out_path: Path, fmt: str) -> str:
    """1 セグメントをアライメントしてラベルを書き出す（サブプロセス用）。"""
    import pyshiro
    from pyshiro.labels import (segments_to_phoneme_intervals,
                                write_lab, write_textgrid, write_audacity)

    model    = pyshiro.load_hsmm(model_path)
    phonemap = pyshiro.load_phonemap(phonemap_path)
    table    = pyshiro.load_table()

    phonemes = pyshiro.convert_lyric_file(lyrics_path, table)
    if not phonemes:
        return f"  [SKIP] 音素列が空: {wav_path.name}"

    streams  = pyshiro.extract_mfcc_from_file(wav_path)
    T        = streams[0].shape[0]
    state_seq = pyshiro.build_state_sequence(phonemes, phonemap, T)
    segments, _ = pyshiro.forced_align_2pass(
        model, streams, state_seq,
        nodur_phonemes={"pau", "br"},
        hmm_cap=1000,
    )

    intervals = segments_to_phoneme_intervals(phonemes, segments)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "textgrid":
        write_textgrid(intervals, out_path)
    elif fmt == "audacity":
        write_audacity(intervals, out_path)
    else:
        write_lab(intervals, out_path)

    return f"  saved: {out_path.name}"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--wav_dir",    type=Path, required=True)
    parser.add_argument("--lyrics_dir", type=Path, required=True)
    parser.add_argument("--model",      type=Path, required=True)
    parser.add_argument("--phonemap",   type=Path, required=True)
    parser.add_argument("--out_dir",    type=Path, required=True)
    parser.add_argument("--format",     choices=["lab", "textgrid", "audacity"],
                        default="lab")
    parser.add_argument("--overwrite",  action="store_true")
    parser.add_argument("--jobs",       type=int,
                        default=os.cpu_count() or 1)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    _EXT = {"lab": ".lab", "textgrid": ".TextGrid", "audacity": ".txt"}
    ext = _EXT[args.format]

    wav_files = sorted(args.wav_dir.glob("*.wav"))
    if not wav_files:
        print(f"WAV ファイルが見つかりません: {args.wav_dir}")
        return

    tasks = []
    for wav_path in wav_files:
        lyrics_path = args.lyrics_dir / f"{wav_path.stem}.txt"
        if not lyrics_path.exists():
            print(f"  [SKIP] 歌詞なし: {lyrics_path.name}")
            continue
        out_path = args.out_dir / f"{wav_path.stem}{ext}"
        if out_path.exists() and not args.overwrite:
            print(f"  [SKIP] 既存: {out_path.name}")
            continue
        tasks.append((wav_path, lyrics_path, out_path))

    print(f"{len(tasks)} セグメントをアライメント中 (jobs={args.jobs}) ...")

    if args.jobs == 1:
        for wav_path, lyrics_path, out_path in tasks:
            print(f"\n{wav_path.stem}", flush=True)
            try:
                msg = _align_one(wav_path, lyrics_path,
                                 args.model, args.phonemap,
                                 out_path, args.format)
                print(msg, flush=True)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"  [ERR] {e}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {
                pool.submit(_align_one, wav_path, lyrics_path,
                            args.model, args.phonemap,
                            out_path, args.format): wav_path.stem
                for wav_path, lyrics_path, out_path in tasks
            }
            for future in as_completed(futures):
                stem = futures[future]
                try:
                    print(f"{stem}: {future.result()}", flush=True)
                except Exception as e:
                    print(f"{stem}: [ERR] {e}", flush=True)

    print("\n完了。")


if __name__ == "__main__":
    main()
