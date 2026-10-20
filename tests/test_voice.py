from types import SimpleNamespace

import pytest

from book2audio.exceptions import InputError
from book2audio.tts.voice import VoiceLibrary
from book2audio.workdir import VOICE_FIELDS, WorkDir


def voice_args(**kw):
    d = dict(ref=None, ref_text=None, voice=None, seed=1234, design_model="m", ref_max_secs=20,
             narrator=None)
    d.update(kw)
    return SimpleNamespace(**d)


def test_library_root_follows_xdg(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert VoiceLibrary().root == tmp_path / "book2audio" / "voices"


def test_book_voice_round_trip(tmp_path):
    ref = tmp_path / "s.wav"
    ref.write_bytes(b"x")
    WorkDir(tmp_path).save_voice_settings(voice_args(ref=str(ref), ref_text="здравей", seed=7))
    assert (tmp_path / "voice_settings.json").exists()
    new = voice_args(seed=1234)
    WorkDir(tmp_path).load_voice_settings(new)
    assert {k: getattr(new, k) for k in VOICE_FIELDS} == dict(
        ref=str(ref.resolve()), ref_text="здравей", voice=None, seed=7, design_model="m",
        ref_max_secs=20)


def test_book_voice_not_loaded_when_voice_given(tmp_path):
    WorkDir(tmp_path).save_voice_settings(voice_args(seed=7))
    new = voice_args(voice="female")
    WorkDir(tmp_path).load_voice_settings(new)
    assert new.seed == 1234


def test_saved_named_voice_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    wav = tmp_path / "s.wav"
    wav.write_bytes(b"x")
    library = VoiceLibrary()
    library.save("nv", {"ref_audio": str(wav), "ref_text": "текст"})
    assert (tmp_path / "xdg" / "book2audio" / "voices" / "nv" / "reference.wav").exists()
    voice = library.get("nv")
    assert voice["ref_audio"].endswith("voices/nv/reference.wav")
    assert voice["ref_text"] == "текст"


def test_unknown_named_voice_lists_saved(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    with pytest.raises(InputError, match="No saved voice named 'ghost'.*none yet"):
        VoiceLibrary().get("ghost")
