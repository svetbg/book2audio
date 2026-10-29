import pytest

from book2audio.tts.remote import RemoteNarrator


@pytest.fixture(autouse=True)
def forget_uploads():
    RemoteNarrator._uploaded.clear()      # per process in real use; per test here, as each test has its own server
