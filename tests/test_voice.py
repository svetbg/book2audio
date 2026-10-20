from types import SimpleNamespace

import pytest

from book2audio.exceptions import InputError
from book2audio.tts.voice import (VOICE_FIELDS, load_book_voice, save_book_voice,
                                  save_named_voice, use_named_voice, voices_dir)


def voice_args(**kw):
    d = dict(ref=None, ref_text=None, voice=None, seed=1234, design_model="m", ref_max_secs=20,
             narrator=None)
    d.update(kw)
    return SimpleNamespace(**d)


def test_voices_dir_follows_xdg(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert voices_dir() == tmp_path / "book2audio" / "voices"


def test_book_voice_round_trip(tmp_path):
    ref = tmp_path / "s.wav"
    ref.write_bytes(b"x")
    save_book_voice(voice_args(ref=str(ref), ref_text="здравей", seed=7), tmp_path)
    assert (tmp_path / "voice_settings.json").exists()
    new = voice_args(seed=1234)
    load_book_voice(new, tmp_path)
    assert {k: getattr(new, k) for k in VOICE_FIELDS} == dict(
        ref=str(ref.resolve()), ref_text="здравей", voice=None, seed=7, design_model="m",
        ref_max_secs=20)


def test_book_voice_not_loaded_when_voice_given(tmp_path):
    save_book_voice(voice_args(seed=7), tmp_path)
    new = voice_args(voice="female")
    load_book_voice(new, tmp_path)
    assert new.seed == 1234


def test_saved_named_voice_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    wav = tmp_path / "s.wav"
    wav.write_bytes(b"x")
    save_named_voice("nv", {"ref_audio": str(wav), "ref_text": "текст"})
    assert (tmp_path / "xdg" / "book2audio" / "voices" / "nv" / "reference.wav").exists()
    args = voice_args(narrator="nv")
    use_named_voice(args)
    assert args.ref.endswith("voices/nv/reference.wav")
    assert args.ref_text == "текст"


def test_unknown_named_voice_lists_saved(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    with pytest.raises(InputError, match="No saved voice named 'ghost'.*none yet"):
        use_named_voice(voice_args(narrator="ghost"))
