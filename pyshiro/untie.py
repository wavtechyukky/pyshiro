"""
untie.py

モノフォン → トライフォン展開（shiro-untie 相当）

コーパスの .lab ファイルから出現したすべてのトライフォンコンテキスト
（L-ph+R 形式）を収集し、モノフォン音素マップとモデルを展開して
トライフォン用の phonemap.json と .hsmm を生成する。

使い方:
  python -m pyshiro.untie \\
      --phonemap  models/intunist-jp6_phonemap.json \\
      --model     models/intunist-jp6_generic.hsmm \\
      --lab_dir   data/lab \\
      --out_phonemap  my_tri_phonemap.json \\
      --out_model     my_tri_model.hsmm

アライメント・訓練時:
  - build_state_sequence() はトライフォンエントリを優先し、
    未見のコンテキストはモノフォンにフォールバックする（align.py 参照）。
  - train.py の --phonemap にトライフォン phonemap を指定して追加訓練する。
"""

import argparse
import copy
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

from pyshiro.model import GMM, Duration, HSMMModel, Stream, load_hsmm, save_hsmm


# ---------------------------------------------------------------------------
# .lab 読み込み（train.py と同じ形式）
# ---------------------------------------------------------------------------

def _read_lab_phones(path: Path) -> List[str]:
    """HTK .lab から音素列を返す（時間情報は破棄）。"""
    phones = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) >= 3:
            phones.append(parts[2])
    return phones


def _core_phone(triphone: str) -> str:
    """'L-ph+R' または 'L-ph' または 'ph+R' または 'ph' から中心音素を返す。"""
    ph = triphone
    if '+' in ph:
        ph = ph.split('+')[0]
    if '-' in ph:
        ph = ph.split('-')[1]
    return ph


# ---------------------------------------------------------------------------
# トライフォンコンテキストの収集
# ---------------------------------------------------------------------------

def collect_triphones(
    lab_dir: Path,
    sil_phones: Set[str] = None,
) -> Set[Tuple[str, str, str]]:
    """
    lab_dir 内の全 .lab ファイルからトライフォンコンテキストを収集する。

    Parameters
    ----------
    sil_phones : コンテキストに含めない音素（pau, sil など）

    Returns
    -------
    set of (L, center, R) tuples
    """
    if sil_phones is None:
        sil_phones = {'pau', 'sil', 'br'}

    triphones: Set[Tuple[str, str, str]] = set()

    for lab_path in sorted(lab_dir.glob("*.lab")):
        phones = _read_lab_phones(lab_path)
        for i, ph in enumerate(phones):
            if ph in sil_phones:
                continue
            L = phones[i - 1] if i > 0 else 'pau'
            R = phones[i + 1] if i < len(phones) - 1 else 'pau'
            triphones.add((L, ph, R))

    return triphones


# ---------------------------------------------------------------------------
# phonemap 展開
# ---------------------------------------------------------------------------

def build_triphone_phonemap(
    mono_phonemap: dict,
    triphones: Set[Tuple[str, str, str]],
) -> dict:
    """
    モノフォン phonemap をトライフォン展開した新しい phone_map を返す。

    - モノフォンエントリはそのまま残す（未見コンテキストのフォールバック用）
    - 各トライフォン L-ph+R に対して新しい out_idx を割り当てる
    - dur_idx はモノフォンを共有する（継続時間は音素ごとに一致）
    - 新しいエントリは "triphone": true フラグを持つ

    Returns
    -------
    dict  新しい phone_map
      （out_idx の最大値を max_out_idx キーで記録）
    """
    # 現在の最大 out_idx を求める
    max_out = 0
    for entry in mono_phonemap.values():
        for st in entry['states']:
            for o in st['out']:
                if o > max_out:
                    max_out = o
    next_out = max_out + 1

    new_phonemap = copy.deepcopy(mono_phonemap)

    for L, ph, R in sorted(triphones):
        if ph not in mono_phonemap:
            continue
        mono_entry = mono_phonemap[ph]
        tri_key    = f"{L}-{ph}+{R}"
        if tri_key in new_phonemap:
            continue  # 既存エントリはスキップ

        # 新しい out_idx をモノフォンと同数分割り当て
        tri_states = []
        for st in mono_entry['states']:
            new_out = [next_out + k for k, _ in enumerate(st['out'])]
            next_out += len(st['out'])
            tri_states.append({'dur': st['dur'], 'out': new_out})

        tri_entry = {
            'states':   tri_states,
            'triphone': True,
        }
        # オプション情報をコピー
        for key in ('topology', 'pskip', 'durfloor', 'durceil'):
            if key in mono_entry:
                tri_entry[key] = mono_entry[key]

        new_phonemap[tri_key] = tri_entry

    return new_phonemap, next_out


# ---------------------------------------------------------------------------
# モデル展開
# ---------------------------------------------------------------------------

def expand_model(
    mono_model: HSMMModel,
    mono_phonemap: dict,
    tri_phonemap: dict,
    total_out: int,
) -> HSMMModel:
    """
    モノフォンモデルをトライフォンに展開する。

    各トライフォン状態の GMM はモノフォンの GMM をコピーして初期化する。
    継続時間分布はモノフォンと共有（同一インデックス）。
    """
    n_stream = mono_model.nstream
    ndim     = mono_model.ndim

    # 新しい GMM 配列（既存モノフォン + 新トライフォン）
    new_streams = []
    for s_idx, stream in enumerate(mono_model.streams):
        mono_gmms = stream.gmms   # インデックス = out_idx
        n_mono    = len(mono_gmms)

        new_gmms = list(mono_gmms)   # モノフォンをそのまま残す

        # トライフォンごとに GMM を追加（モノフォンからコピー）
        # tri_phonemap の全エントリをスキャンして新 out_idx を収集
        added: Dict[int, GMM] = {}
        for key, entry in tri_phonemap.items():
            if not entry.get('triphone'):
                continue
            # 対応するモノフォン out_idx を参照
            core = _core_phone(key)
            if core not in mono_phonemap:
                continue
            mono_entry = mono_phonemap[core]
            for k, tri_st in enumerate(entry['states']):
                mono_out = mono_entry['states'][k]['out'][0]
                for tri_out in tri_st['out']:
                    if tri_out < n_mono:
                        continue  # モノフォン範囲内（重複割り当て不可のはずだが）
                    if tri_out not in added:
                        # モノフォン GMM をディープコピー
                        src = mono_gmms[mono_out]
                        added[tri_out] = GMM(
                            nmix=src.nmix, ndim=src.ndim,
                            weights=src.weights.copy(),
                            means=src.means.copy(),
                            vars=src.vars.copy(),
                            varfloors=src.varfloors.copy(),
                        )

        # added を out_idx の順に並べて追加
        n_new = total_out - n_mono
        for out_idx in range(n_mono, total_out):
            if out_idx in added:
                new_gmms.append(added[out_idx])
            else:
                # 万一欠番があれば先頭 GMM のコピーで埋める
                src = mono_gmms[0]
                new_gmms.append(GMM(
                    nmix=src.nmix, ndim=src.ndim,
                    weights=src.weights.copy(),
                    means=src.means.copy(),
                    vars=src.vars.copy(),
                    varfloors=src.varfloors.copy(),
                ))

        new_streams.append(Stream(weight=stream.weight, gmms=new_gmms))

    return HSMMModel(streams=new_streams, durations=list(mono_model.durations))


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def untie(
    phonemap_path: Path,
    model_path:    Path,
    lab_dir:       Path,
    out_phonemap:  Path,
    out_model:     Path,
    sil_phones:    Set[str] = None,
) -> None:
    """
    モノフォン → トライフォン展開を実行する。

    Parameters
    ----------
    phonemap_path : モノフォン phonemap.json
    model_path    : モノフォン .hsmm
    lab_dir       : コーパス .lab ディレクトリ（トライフォン収集用）
    out_phonemap  : 出力トライフォン phonemap.json
    out_model     : 出力トライフォン .hsmm
    sil_phones    : コンテキスト境界として使う無音音素（デフォルト: pau, sil, br）
    """
    phonemap_raw = json.loads(phonemap_path.read_text(encoding="utf-8"))
    mono_phonemap = phonemap_raw["phone_map"]
    mono_model    = load_hsmm(model_path)

    print(f"モノフォン音素数: {len(mono_phonemap)}")
    print(f"モデル GMM 数:  {mono_model.nstate}")

    print("トライフォンコンテキスト収集中...")
    triphones = collect_triphones(lab_dir, sil_phones)
    print(f"  ユニークトライフォン数: {len(triphones)}")

    print("phonemap 展開中...")
    tri_phonemap, total_out = build_triphone_phonemap(mono_phonemap, triphones)
    n_tri = sum(1 for e in tri_phonemap.values() if e.get('triphone'))
    print(f"  トライフォンエントリ数: {n_tri}")
    print(f"  出力 GMM 総数: {total_out}")

    print("モデル展開中...")
    tri_model = expand_model(mono_model, mono_phonemap, tri_phonemap, total_out)

    # phonemap JSON 書き出し
    out_raw = {"phone_map": tri_phonemap}
    out_phonemap.write_text(json.dumps(out_raw, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    print(f"  phonemap 保存: {out_phonemap}")

    # モデル保存
    save_hsmm(tri_model, out_model)
    print(f"  model 保存: {out_model}")
    print("完了")


def main():
    parser = argparse.ArgumentParser(description="モノフォン → トライフォン展開")
    parser.add_argument("--phonemap",     type=Path, required=True,
                        help="モノフォン phonemap.json")
    parser.add_argument("--model",        type=Path, required=True,
                        help="モノフォン .hsmm")
    parser.add_argument("--lab_dir",      type=Path, required=True,
                        help="コーパス .lab ディレクトリ")
    parser.add_argument("--out_phonemap", type=Path, required=True,
                        help="出力トライフォン phonemap.json")
    parser.add_argument("--out_model",    type=Path, required=True,
                        help="出力トライフォン .hsmm")
    parser.add_argument("--sil", nargs="*", default=["pau", "sil", "br"],
                        help="コンテキスト境界音素（デフォルト: pau sil br）")
    args = parser.parse_args()

    untie(
        phonemap_path=args.phonemap,
        model_path=args.model,
        lab_dir=args.lab_dir,
        out_phonemap=args.out_phonemap,
        out_model=args.out_model,
        sil_phones=set(args.sil),
    )


if __name__ == "__main__":
    main()
