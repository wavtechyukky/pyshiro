"""
DAEM・topology・トライフォン（untie）の動作確認テスト。
実音声不要 — すべて合成データで実行できる。
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from pyshiro.model import GMM, Duration, HSMMModel, Stream, save_hsmm, load_hsmm
from pyshiro.align import load_phonemap, build_state_sequence, forced_align
from pyshiro.train import train
from pyshiro.untie import collect_triphones, build_triphone_phonemap, expand_model


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------

N_STREAMS = 3
N_DIM = 12

# 3音素 × 3状態 = 9状態
PHONEMES = ["pau", "a", "k"]

def _phonemap_data(topologies: dict = None) -> dict:
    """
    pau / a / k の3音素、各3状態の最小 phonemap dict を返す。
    dur/out インデックスは 0〜8 の連番。
    """
    topologies = topologies or {}
    phone_map = {}
    for i, ph in enumerate(PHONEMES):
        entry = {
            "states": [
                {"dur": i * 3 + k, "out": [i * 3 + k] * N_STREAMS}
                for k in range(3)
            ]
        }
        if ph in topologies:
            entry["topology"] = topologies[ph]
        phone_map[ph] = entry
    return {"phone_map": phone_map}


def _make_model() -> HSMMModel:
    """phonemap の 9状態に対応する最小 HSMMModel を返す。"""
    n_states = len(PHONEMES) * 3  # 9

    streams = []
    for _ in range(N_STREAMS):
        gmms = [
            GMM(
                nmix=1,
                ndim=N_DIM,
                weights=np.array([1.0]),
                means=np.zeros((1, N_DIM)),
                vars=np.ones((1, N_DIM)),
                varfloors=np.ones((1, N_DIM)) * 1e-4,
            )
            for _ in range(n_states)
        ]
        streams.append(Stream(weight=1.0, gmms=gmms))

    durations = [
        Duration(mean=5.0, var=4.0, floor=1, ceil=0, fixed_mean=0, vfloor=1e-4)
        for _ in range(n_states)
    ]
    return HSMMModel(streams=streams, durations=durations)


def _make_features(T: int = 60) -> list:
    """T フレームのダミー MFCC ストリームを返す（N_STREAMS × N_DIM）。"""
    rng = np.random.default_rng(42)
    return [rng.standard_normal((T, N_DIM)).astype(np.float32) for _ in range(N_STREAMS)]


def _write_corpus(tmp_path: Path) -> tuple[Path, Path, Path]:
    """ダミーコーパス（wav + lab + phonemap）を tmp_path に書き出す。"""
    wav_dir = tmp_path / "wav"
    lab_dir = tmp_path / "lab"
    wav_dir.mkdir(); lab_dir.mkdir()

    audio = np.zeros(32000, dtype=np.float32)
    sf.write(wav_dir / "utt1.wav", audio, 16000)

    # pau a k pau — 各状態5フレーム程度の長さ
    (lab_dir / "utt1.lab").write_text(
        "0 500000 pau\n"
        "500000 1000000 a\n"
        "1000000 1500000 k\n"
        "1500000 2000000 pau\n"
    )

    pmap_path = tmp_path / "phonemap.json"
    pmap_path.write_text(json.dumps(_phonemap_data()))

    return wav_dir, lab_dir, pmap_path


# ---------------------------------------------------------------------------
# 1. DAEM テスト
# ---------------------------------------------------------------------------

class TestDAEM:
    def test_daem_runs_without_error(self, tmp_path):
        """DAEM=True で train() が例外なく完了すること。"""
        wav_dir, lab_dir, pmap_path = _write_corpus(tmp_path)

        model = _make_model()
        init_path = tmp_path / "init.hsmm"
        save_hsmm(model, init_path)

        out_path = tmp_path / "out.hsmm"
        train(
            wav_dir=wav_dir,
            lab_dir=lab_dir,
            phonemap_path=pmap_path,
            out_path=out_path,
            iters=2,
            hmm_iters=1,
            daem=True,
            n_jobs=1,
            init_model=init_path,
        )
        assert out_path.exists(), "モデルが保存されていない"

    def test_daem_temp_low(self, tmp_path):
        """温度 0.1 の低温 DAEM でも forced_align が完走すること。"""
        pmap_data = _phonemap_data()
        pmap_path = tmp_path / "phonemap.json"
        pmap_path.write_text(json.dumps(pmap_data))

        phonemap = load_phonemap(pmap_path)
        model = _make_model()
        streams = _make_features(60)
        T = streams[0].shape[0]

        phonemes_seq = ["pau", "a", "k", "pau"]
        state_seq = build_state_sequence(phonemes_seq, phonemap, T)
        segs, _ = forced_align(model, streams, state_seq, use_duration=False, daem_temp=0.1)
        assert len(segs) == len(state_seq)
        assert segs[-1][1] == T


# ---------------------------------------------------------------------------
# 2. topology テスト
# ---------------------------------------------------------------------------

class TestTopology:
    @pytest.mark.parametrize("topo", ["type-a", "type-b", "type-c", "skip-boundary"])
    def test_topology_alignment(self, tmp_path, topo):
        """各 topology で forced_align が正常完了し、T フレームをカバーすること。"""
        pmap_data = _phonemap_data(topologies={"a": topo})
        pmap_path = tmp_path / "phonemap.json"
        pmap_path.write_text(json.dumps(pmap_data))

        phonemap = load_phonemap(pmap_path)
        model = _make_model()
        streams = _make_features(60)
        T = streams[0].shape[0]

        phonemes_seq = ["pau", "a", "pau"]
        state_seq = build_state_sequence(phonemes_seq, phonemap, T)
        segs, _ = forced_align(model, streams, state_seq, use_duration=False)

        assert len(segs) == len(state_seq)
        assert segs[-1][1] == T

    def test_topology_state_seq_structure(self, tmp_path):
        """type-b の topo_sources が正しく設定されること。"""
        pmap_data = _phonemap_data(topologies={"a": "type-b"})
        pmap_path = tmp_path / "phonemap.json"
        pmap_path.write_text(json.dumps(pmap_data))

        phonemap = load_phonemap(pmap_path)
        T = 60
        phonemes_seq = ["pau", "a", "pau"]
        state_seq = build_state_sequence(phonemes_seq, phonemap, T)

        # 音素 "a" の最終状態（インデックス5: pau×3 + a×3 の5番目）に
        # topo_sources があること
        a_states = [s for s in state_seq if s.phoneme == "a"]
        # type-b: 最終状態に中間状態からのスキップが来る
        final_a = a_states[-1]
        assert len(final_a.topo_sources) > 0, "type-b の topo_sources が空"


# ---------------------------------------------------------------------------
# 3. トライフォン（untie）テスト
# ---------------------------------------------------------------------------

class TestUntie:
    def _setup_lab(self, lab_dir: Path):
        (lab_dir / "utt1.lab").write_text(
            "0 500000 pau\n500000 1000000 a\n1000000 1500000 k\n1500000 2000000 pau\n"
        )
        (lab_dir / "utt2.lab").write_text(
            "0 500000 pau\n500000 1000000 k\n1000000 1500000 a\n1500000 2000000 pau\n"
        )

    def test_collect_triphones(self, tmp_path):
        """collect_triphones が正しくトライフォンを収集すること。"""
        lab_dir = tmp_path / "lab"
        lab_dir.mkdir()
        self._setup_lab(lab_dir)

        triphones = collect_triphones(lab_dir)
        # pau-a+k, a-k+pau などが含まれるはず
        phones = {t[1] for t in triphones}
        assert "a" in phones
        assert "k" in phones

    def test_build_triphone_phonemap(self, tmp_path):
        """build_triphone_phonemap がトライフォンエントリを生成すること。"""
        lab_dir = tmp_path / "lab"
        lab_dir.mkdir()
        self._setup_lab(lab_dir)

        pmap_data = _phonemap_data()
        mono_phonemap = pmap_data["phone_map"]
        triphones = collect_triphones(lab_dir)

        tri_phonemap, _ = build_triphone_phonemap(mono_phonemap, triphones)

        # トライフォンエントリが存在すること
        tri_entries = [p for p in tri_phonemap if "-" in p or "+" in p]
        assert len(tri_entries) > 0, "トライフォンエントリが生成されていない"

        # モノフォンも残っていること（フォールバック用）
        assert "pau" in tri_phonemap
        assert "a" in tri_phonemap

    def test_expand_model(self, tmp_path):
        """expand_model がトライフォン数に対応したモデルを生成すること。"""
        lab_dir = tmp_path / "lab"
        lab_dir.mkdir()
        self._setup_lab(lab_dir)

        pmap_data = _phonemap_data()
        mono_phonemap = pmap_data["phone_map"]
        mono_model = _make_model()
        triphones = collect_triphones(lab_dir)
        tri_phonemap, total_out = build_triphone_phonemap(mono_phonemap, triphones)

        tri_model = expand_model(mono_model, mono_phonemap, tri_phonemap, total_out)

        # 状態数がモノフォンより多いこと
        assert len(tri_model.durations) >= len(mono_model.durations)

    def test_untie_alignment(self, tmp_path):
        """トライフォン phonemap でアライメントが動作すること。"""
        lab_dir = tmp_path / "lab"
        lab_dir.mkdir()
        self._setup_lab(lab_dir)

        pmap_data = _phonemap_data()
        mono_phonemap = pmap_data["phone_map"]
        mono_model = _make_model()
        triphones = collect_triphones(lab_dir)
        tri_phonemap_raw, total_out = build_triphone_phonemap(mono_phonemap, triphones)
        tri_model = expand_model(mono_model, mono_phonemap, tri_phonemap_raw, total_out)

        # phonemap JSON として保存して再読み込み
        pmap_path = tmp_path / "tri_phonemap.json"
        pmap_path.write_text(json.dumps({"phone_map": tri_phonemap_raw}))
        tri_phonemap = load_phonemap(pmap_path)

        streams = _make_features(60)
        T = streams[0].shape[0]

        phonemes_seq = ["pau", "a", "k", "pau"]
        state_seq = build_state_sequence(phonemes_seq, tri_phonemap, T)
        segs, _ = forced_align(tri_model, streams, state_seq, use_duration=False)

        assert segs[-1][1] == T
