from ebooklib import epub

from book2audio.loaders import load_book

BODY = "Дълъг абзац с текст. " * 20


def make_epub(path, docs, toc_titles):
    b = epub.EpubBook()
    b.set_identifier("x")
    b.set_title("Заглавие")
    b.add_author("Автор")
    b.set_language("bg")
    items = []
    for i, html in enumerate(docs, 1):
        c = epub.EpubHtml(title=toc_titles.get(i, ""), file_name=f"c{i}.xhtml")
        c.content = f"<html><body>{html}</body></html>"
        b.add_item(c)
        items.append(c)
    b.toc = tuple(c for i, c in enumerate(items, 1) if i in toc_titles)
    b.spine = items
    b.add_item(epub.EpubNcx())
    b.add_item(epub.EpubNav())
    epub.write_epub(str(path), b)


def test_metadata_toc_titles_and_title_fallback(tmp_path):
    f = tmp_path / "b.epub"
    make_epub(f, [f"<p>{BODY}</p>",                       # TOC title
                  f"<h2>Заглавие от h2</h2><p>{BODY}</p>",   # first h1-h3
                  f"<p>Кратко начало</p><p>{BODY}</p>"],      # short first paragraph
              {1: "Първа"})
    book = load_book(f, 100)
    assert (book.title, book.author) == ("Заглавие", "Автор")
    assert [c.title for c in book.chapters] == ["Първа", "Заглавие от h2", "Кратко начало"]


def test_leaf_blocks_not_duplicated_and_min_chars(tmp_path):
    f = tmp_path / "b.epub"
    make_epub(f, [f"<div><p>{BODY}</p></div>", "<p>кратко</p>"], {})
    book = load_book(f, 100)
    assert len(book.chapters) == 1
    assert book.chapters[0].paragraphs == [BODY.strip()]
