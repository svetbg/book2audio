from book2audio.text.lexicon import Lexicon


def test_replacement_and_capital(tmp_path):
    f = tmp_path / "lex.txt"
    f.write_text("бутона=буто́на\n", encoding="utf-8")
    lex = Lexicon.from_file(str(f))
    assert lex.apply("бутона") == "буто́на"
    assert lex.apply("Бутона цъфти") == "Буто́на цъфти"
    assert lex.apply("бутонаха") == "бутонаха"   # whole words only


def test_comments_and_both_separators(tmp_path):
    f = tmp_path / "lex.txt"
    f.write_text("# comment\n\nкат\tкота\nкуче=пес\nno separator\n", encoding="utf-8")
    lex = Lexicon.from_file(str(f))
    assert len(lex.rules) == 2
    assert lex.apply("кат и куче") == "кота и пес"


def test_no_path():
    assert Lexicon.from_file(None).rules == []
