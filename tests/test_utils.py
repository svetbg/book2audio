from book2audio.utils import compact_ranges, parse_chapters


def test_parse_chapters():
    assert parse_chapters(None, 4) == [1, 2, 3, 4]
    assert parse_chapters("1-3,5", 10) == [1, 2, 3, 5]
    assert parse_chapters("3-", 5) == [3, 4, 5]
    assert parse_chapters("-2", 5) == [1, 2]
    assert parse_chapters("0,2,99", 5) == [2]


def test_compact_ranges():
    assert compact_ranges([1, 2, 3, 5, 7, 8]) == "1-3, 5, 7-8"
    assert compact_ranges([4]) == "4"
