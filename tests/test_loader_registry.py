from pathlib import Path

from book2audio.loaders import EpubLoader, TxtLoader, for_path


def test_loader_picked_by_suffix_case_insensitively():
    assert isinstance(for_path(Path("a.EPUB"), 1), EpubLoader)
    assert isinstance(for_path(Path("a.text"), 1), TxtLoader)
    assert for_path(Path("a.txt"), 7).min_chars == 7
