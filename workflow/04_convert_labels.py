"""
workflow/05_convert_labels.py

外部ラベル（他のツール・基準でアノテーションされた .lab / TextGrid）を
このモデルの作風に変換する。

例: 東北きりたん歌声DB の .lab は子音の閉鎖期を含まないが、
pyshiro で訓練したモデルは閉鎖期も子音ラベルに含める。
本スクリプトはその差異を自動的に補正する。

アルゴリズム:
  1. 外部ラベルから「長い母音（≥ anchor_vowel_min s）」や
     「長い pau（≥ anchor_pau_min s）」の時刻をアンカーとして検出する。
  2. 隣接アンカー間（最大 max_seg_sec 秒）を forced_align_2pass で再アライメントする。
  3. エネルギーオンセットで pau/子音境界を後補正する（--no_refine で無効化）。

使い方:
  python workflow/05_convert_labels.py audio.wav external.lab \\
      --model    checkpoint/pyshiro-jp-v1.hsmm \\
      --phonemap checkpoint/pyshiro-jp-v1_phonemap.json \\
      --out      converted.lab

  python workflow/05_convert_labels.py audio.wav external.TextGrid \\
      --model    checkpoint/pyshiro-jp-v1.hsmm \\
      --phonemap checkpoint/pyshiro-jp-v1_phonemap.json \\
      --format   textgrid \\
      --out      converted.TextGrid \\
      --phoneme_map phoneme_map.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))

import pyshiro
from pyshiro.labels import (read_lab, read_textgrid, read_audacity,
                             write_lab, write_lab_sec, write_textgrid, write_audacity)
from pyshiro.align  import realign_external_labels, refine_pau_boundaries, SILENCE_PHONES


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('wav',    type=Path, help='入力 WAV ファイル')
    p.add_argument('labels', type=Path, help='外部ラベルファイル (.lab / .TextGrid / .txt)')
    p.add_argument('--model',    type=Path, required=True, help='.hsmm モデルファイル')
    p.add_argument('--phonemap', type=Path, required=True, help='phonemap.json ファイル')
    p.add_argument('--out',      type=Path, default=None,
                   help='出力ファイルパス（省略時は WAV と同名）')
    p.add_argument('--format',   choices=['lab', 'lab_sec', 'textgrid', 'audacity'],
                   default='lab',
                   help='出力形式（デフォルト: lab）\n'
                        '  lab      : HTK 100ns 整数\n'
                        '  lab_sec  : 秒単位スペース区切り\n'
                        '  textgrid : Praat TextGrid\n'
                        '  audacity : Audacity タブ区切り')
    p.add_argument('--anchor_vowel_min',  type=float, default=0.5,
                   help='母音アンカーの最小長（秒, デフォルト: 0.5）')
    p.add_argument('--anchor_pau_min',    type=float, default=0.5,
                   help='pau アンカーの最小長（秒, デフォルト: 0.5）')
    p.add_argument('--anchor_pau_offset', type=float, default=0.25,
                   help='pau 内アンカー位置（pau 先頭からの秒数, デフォルト: 0.25）')
    p.add_argument('--max_seg_sec',       type=float, default=30.0,
                   help='アンカー間の最大秒数（超える区間はスキップ, デフォルト: 30.0）')
    p.add_argument('--phoneme_map',       type=Path, default=None,
                   help='音素名変換テーブル JSON（外部→pyshiro）。省略時は変換なし。\n'
                        '例: {"q": "cl", "nn": "N"}')
    p.add_argument('--fix_transitions', type=str, default=None,
                   metavar='FROM-TO,FROM-TO,...',
                   help='この移行タイプの音素境界のみ再アライメントで修正する。\n'
                        '指定しない移行タイプは元ラベルを維持（アンカー固定）。\n'
                        '省略時は全ての音素境界を修正（デフォルト動作）。\n'
                        '形式: "FROM-TO" をカンマ区切りで列挙（空白不可）。\n'
                        'FROM/TO は silence / consonant / vowel\n'
                        '例: --fix_transitions vowel-consonant,silence-consonant,vowel-silence')
    p.add_argument('--silence_phones', type=str, default=None,
                   help='無音クラスとして扱う音素（カンマ区切り）。\n'
                        'デフォルト: pau,sil,cl,br')
    p.add_argument('--unknown_phoneme_map', type=str, default=None,
                   help='phonemap に存在しない未知音素のアライメント時の代替（カンマ区切りの KEY=VAL）。\n'
                        '出力ラベルは元の音素名のまま。\n'
                        '例: vy=v,fy=f\n'
                        '未登録の未知音素はアンカー扱い（境界固定）。')
    p.add_argument('--no_refine', action='store_true',
                   help='pau/子音境界のエネルギーオンセット補正を無効化')
    args = p.parse_args()

    out_path = args.out or args.wav.with_suffix(
        '.lab'      if args.format in ('lab', 'lab_sec') else
        '.TextGrid' if args.format == 'textgrid'         else '.txt'
    )

    # ── モデル・phonemap 読み込み ─────────────────────────────────────────────
    print(f'モデル読み込み: {args.model}', flush=True)
    model    = pyshiro.load_hsmm(args.model)
    phonemap = pyshiro.load_phonemap(args.phonemap)

    # ── 音素名変換テーブル ────────────────────────────────────────────────────
    phoneme_map = None
    if args.phoneme_map:
        phoneme_map = json.loads(args.phoneme_map.read_text(encoding='utf-8'))
        print(f'音素名変換テーブル: {phoneme_map}', flush=True)

    # ── 外部ラベル読み込み ────────────────────────────────────────────────────
    print(f'外部ラベル読み込み: {args.labels}', flush=True)
    suf = args.labels.suffix.lower()
    if suf == '.textgrid':
        intervals = read_textgrid(args.labels)
    elif suf == '.txt':
        intervals = read_audacity(args.labels)
    else:
        intervals = read_lab(args.labels)

    n_ph = len(intervals)
    total_sec = intervals[-1][1] - intervals[0][0] if intervals else 0.0
    print(f'  音素数: {n_ph}, 総時間: {total_sec:.1f}s', flush=True)

    # ── 特徴抽出 ─────────────────────────────────────────────────────────────
    print(f'特徴抽出: {args.wav}', flush=True)
    audio, sr = sf.read(str(args.wav), always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    streams = pyshiro.extract_mfcc_from_file(args.wav)
    T = streams[0].shape[0]
    print(f'  T={T} frames ({T * 0.005:.3f}s)', flush=True)

    # ── fix_transitions のパース（論理反転: 指定以外を ignore に変換） ─────────
    _ALL_CLASSES = ('silence', 'vowel', 'consonant')
    _ALL_TRANSITIONS = frozenset(
        (a, b) for a in _ALL_CLASSES for b in _ALL_CLASSES
    )

    _VALID_CLASSES = set(_ALL_CLASSES)
    ignore_transitions = None
    if args.fix_transitions:
        fix_set = set()
        # カンマ区切りの単一文字列をパース（空白が混ざっても許容）
        for tok in args.fix_transitions.replace(' ', '').split(','):
            if not tok:
                continue
            a, _, b = tok.partition('-')
            if a not in _VALID_CLASSES or b not in _VALID_CLASSES:
                raise SystemExit(
                    f"ERROR: 不正な移行タイプ '{tok}'。"
                    f" FROM-TO 形式で各クラスは {sorted(_VALID_CLASSES)} のいずれか。"
                )
            fix_set.add((a, b))
        # fix 指定以外の全移行タイプをアンカー（ignore）にする
        ignore_transitions = _ALL_TRANSITIONS - fix_set or None
        print(f'修正する移行タイプ: {sorted(fix_set)}', flush=True)

    silence_phones = (frozenset(args.silence_phones.split(','))
                      if args.silence_phones else SILENCE_PHONES)

    unknown_phoneme_map = None
    if args.unknown_phoneme_map:
        unknown_phoneme_map = {}
        for pair in args.unknown_phoneme_map.split(','):
            k, _, v = pair.partition('=')
            if k and v:
                unknown_phoneme_map[k.strip()] = v.strip()
        print(f'未知音素マップ: {unknown_phoneme_map}', flush=True)

    # ── 再アライメント ────────────────────────────────────────────────────────
    print('再アライメント中...', flush=True)
    result = realign_external_labels(
        model, streams, intervals, phonemap,
        anchor_vowel_min_sec=args.anchor_vowel_min,
        anchor_pau_min_sec=args.anchor_pau_min,
        anchor_pau_offset_sec=args.anchor_pau_offset,
        max_seg_sec=args.max_seg_sec,
        phoneme_map=phoneme_map,
        nodur_phonemes={'pau', 'br', 'cl'},
        ignore_transitions=ignore_transitions,
        silence_phones=silence_phones,
        unknown_phoneme_map=unknown_phoneme_map,
    )
    print(f'  完了: {len(result)} 音素区間', flush=True)

    # ── pau/子音境界エネルギーオンセット補正 ──────────────────────────────────
    if not args.no_refine:
        before = list(result)
        result = refine_pau_boundaries(result, audio, sr)
        n_fixed = sum(1 for (s1, _, _), (s2, _, _) in zip(before, result) if s1 != s2)
        if n_fixed:
            print(f'  境界補正: {n_fixed} 箇所', flush=True)

    # ── 書き出し ──────────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.format == 'lab':
        write_lab(result, out_path)
    elif args.format == 'lab_sec':
        write_lab_sec(result, out_path)
    elif args.format == 'textgrid':
        write_textgrid(result, out_path)
    else:
        write_audacity(result, out_path)
    print(f'保存: {out_path}', flush=True)

    print('\n--- 結果 ---')
    for start_f, end_f, ph in result:
        print(f'  {start_f * 0.005:.4f}s - {end_f * 0.005:.4f}s  {ph}')


if __name__ == '__main__':
    main()
