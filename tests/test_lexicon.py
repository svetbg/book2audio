from book2audio.text.lexicon import apply_lexicon, load_lexicon


def test_replacement_and_capital(tmp_path):
    f = tmp_path / "lex.txt"
    f.write_text("бутона=буто́на\n", encoding="utf-8")
    rules = load_lexicon(str(f))
    assert apply_lexicon("бутона", rules) == "буто́на"
    assert apply_lexicon("Бутона цъфти", rules) == "Буто́на цъфти"
    assert apply_lexicon("бутонаха", rules) == "бутонаха"   # whole words only


def test_comments_and_both_separators(tmp_path):
    f = tmp_path / "lex.txt"
    f.write_text("# comment\n\nкат\tкота\nкуче=пес\nno separator\n", encoding="utf-8")
    rules = load_lexicon(str(f))
    assert len(rules) == 2
    assert apply_lexicon("кат и куче", rules) == "кота и пес"


def test_no_path():
    assert load_lexicon(None) == []
