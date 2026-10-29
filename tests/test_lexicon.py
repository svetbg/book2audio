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


def test_stress_dict(tmp_path):
    d = tmp_path / "stress-dict.txt"
    # вятър, вятъра (shares 5 letters), абзац (two stresses: skipped), сън (one vowel: skipped)
    d.write_text("0вятър:2\n5а;2\n0абзац:1,4\n0сън:2\n", encoding="utf-8")
    lex_file = tmp_path / "lex.txt"
    lex_file.write_text("вятъра=вятъра́\n", encoding="utf-8")
    lex = Lexicon.from_file(str(lex_file), str(d))
    assert lex.apply("Вятър, вятъра, абзац, сън, вя́тър, Вятърът.") == \
        "Вя́тър, вятъра́, абзац, сън, вя́тър, Вятърът."   # lexicon wins; marked words kept
    assert Lexicon.from_file(None, str(d)).apply("вятъра") == "вя́търа"
