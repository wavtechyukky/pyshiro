"""
phonemes.py

ひらがなテキスト → ENUNU/SHIRO 音素列変換

■ 入力フォーマット（.txt）:
  - ひらがなのみ
  - 空行または改行 → pau（ポーズ）

■ 出力:
  - 音素のリスト（pau 含む）
  - 文頭・文末にも pau を付加

■ 変換テーブル:
  pyshiro/data/kana2phonemes.table（ENUNU の kana2phonemes_003_oto2lab.table）
"""

from pathlib import Path
from typing import List

# パッケージ同梱のデフォルトテーブル
_DEFAULT_TABLE = Path(__file__).parent / "data" / "kana2phonemes.table"

# 母音と長母音の対応（ー処理用）
_VOWEL_TO_LONG = {'a': 'A', 'i': 'I', 'u': 'U', 'e': 'E', 'o': 'O',
                  'A': 'A', 'I': 'I', 'U': 'U', 'E': 'E', 'O': 'O'}


def load_table(table_path: Path = None) -> dict:
    """kana2phonemes.table を読み込み、かな→音素リストの辞書を返す。
    table_path を省略するとパッケージ同梱のテーブルを使用する。
    """
    if table_path is None:
        table_path = _DEFAULT_TABLE
    table = {}
    with open(table_path, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            parts = line.split(' ')
            kana = parts[0]
            phonemes = parts[1:]
            table[kana] = phonemes
    return table


def kana_to_phonemes(text: str, table: dict) -> List[str]:
    """ひらがな1行分を音素列に変換する。"""
    sorted_keys = sorted(table.keys(), key=len, reverse=True)

    phonemes: List[str] = []
    last_vowel = None
    i = 0

    while i < len(text):
        if text[i] == 'ー':
            if last_vowel and last_vowel in _VOWEL_TO_LONG:
                phonemes.append(_VOWEL_TO_LONG[last_vowel])
            else:
                phonemes.append('a')
            i += 1
            continue

        matched = False
        for key in sorted_keys:
            if text[i:i + len(key)] == key:
                phs = table[key]
                phonemes.extend(phs)
                if phs[-1] in _VOWEL_TO_LONG:
                    last_vowel = phs[-1]
                i += len(key)
                matched = True
                break

        if not matched:
            print(f"  [WARNING] 未対応の文字: '{text[i]}' (U+{ord(text[i]):04X}) → スキップ")
            i += 1

    return phonemes


def _expand_inline_tokens(line: str, table: dict) -> List[str]:
    """
    1行のテキストを音素列に変換する。
    [xxx] 形式および裸の ASCII 英字列を音素トークンとして直接展開する。
    例: "きっとbrとべば"  → ['k','i','cl','t','o','br','t','o','b','e','b','a']
    例: "きっと[br]とべば" → 同上
    """
    import re
    result = []
    # [xxx] または裸の ASCII 英字列でトークン分割
    for part in re.split(r'(\[[^\]]+\]|[A-Za-z]+)', line):
        if not part:
            continue
        if part.startswith('[') and part.endswith(']'):
            result.append(part[1:-1].strip())
        elif part.isascii() and part.isalpha():
            result.append(part)
        else:
            result.extend(kana_to_phonemes(part, table))
    return result


def text_to_phonemes(text: str, table: dict) -> List[str]:
    """複数行テキスト（歌詞ファイル全体）を音素列に変換する。
    空行 → pau、文頭・文末にも pau を付加。
    行中に [br] / [pau] のようなインライントークンを埋め込める。
    """
    result: List[str] = ['pau']

    for line in text.splitlines():
        line = line.strip()
        if not line:
            if result[-1] != 'pau':
                result.append('pau')
        else:
            phs = _expand_inline_tokens(line, table)
            if phs:
                result.extend(phs)
                if result[-1] != 'pau':
                    result.append('pau')

    if result[-1] != 'pau':
        result.append('pau')

    return result


def _is_phoneme_text(text: str) -> bool:
    """スペース・改行・ASCII 英字のみ → 音素直書きファイルと判断。"""
    return all(c in ' \n\r' or ('a' <= c <= 'z') or ('A' <= c <= 'Z') for c in text)


def _parse_phoneme_text(text: str) -> List[str]:
    """スペース区切り音素列テキスト → 音素リスト。改行 = pau。"""
    result: List[str] = ['pau']
    for line in text.splitlines():
        tokens = [t for t in line.strip().split() if t]
        if not tokens:
            if result[-1] != 'pau':
                result.append('pau')
        else:
            result.extend(tokens)
            if result[-1] != 'pau':
                result.append('pau')
    if result[-1] != 'pau':
        result.append('pau')
    return result


def convert_lyric_file(txt_path: Path, table: dict = None) -> List[str]:
    """歌詞 .txt ファイルを読み込み、音素列を返す。
    table を省略するとパッケージ同梱のテーブルを使用する。
    """
    if table is None:
        table = load_table()
    text = txt_path.read_text(encoding='utf-8')
    if _is_phoneme_text(text):
        return _parse_phoneme_text(text)
    return text_to_phonemes(text, table)
