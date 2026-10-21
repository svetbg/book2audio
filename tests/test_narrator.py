import sys

import numpy as np
import pytest

from book2audio.exceptions import MissingDependency
from book2audio.tts.narrator import Narrator
from stubs import stub_modules

SR = 24000
TEXT = "а" * 130                                   # ~10 s expected


class FakeNarrator(Narrator):
    sr = SR

    def __init__(self, secs):
        self.secs = list(secs)
        self.seeds = []

    def generate(self, text, seed, **voice):
        self.seeds.append(seed)
        return np.full(int(self.secs.pop(0) * SR), 0.3, dtype=np.float32)


def test_synth_chunk_retries_with_another_seed_until_plausible():
    narr = FakeNarrator([1.0, 10.0])
    assert len(narr.synth_chunk(TEXT, {}, 5, 2)) > 9 * SR
    assert narr.seeds == [5, 1005]


def test_synth_chunk_keeps_best_after_all_tries():
    narr = FakeNarrator([1.0, 40.0, 3.0])
    a = narr.synth_chunk(TEXT, {}, 5, 2)
    assert narr.seeds == [5, 1005, 2005]
    assert 2.9 < len(a) / SR < 3.2                  # closest to the expected length


def test_missing_omnivoice_names_the_quoted_extra(monkeypatch):
    stub_modules(monkeypatch)
    monkeypatch.setitem(sys.modules, "omnivoice", None)             # import fails
    with pytest.raises(MissingDependency, match=r'pip install "\.\[local\]"  \(or: pip install "book2audio\[local\]"\)'):
        Narrator("m")
