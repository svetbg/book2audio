from pathlib import Path

from book2audio import Book, Chapter
from book2audio.exceptions import MissingDependency
from book2audio.utils import clean_text, fill_missing_titles

BLOCK_TAGS = ["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote",
              "div", "pre", "td", "dt", "dd"]


def _flatten_toc(toc):
    out = []
    for entry in toc:
        if isinstance(entry, (list, tuple)):
            sec = entry[0]
            children = entry[1] if len(entry) > 1 else []
            if getattr(sec, "href", None):
                out.append((sec.href, sec.title))
            out.extend(_flatten_toc(children))
        elif getattr(entry, "href", None):
            out.append((entry.href, entry.title))
    return out


def load_epub(path: Path, min_chars: int) -> Book:
    import warnings
    warnings.filterwarnings("ignore")
    try:
        import ebooklib
        from ebooklib import epub
        from bs4 import BeautifulSoup
    except ImportError:
        raise MissingDependency("Missing packages. Run: pip install ebooklib beautifulsoup4")

    book = epub.read_epub(str(path))

    def meta(name):
        try:
            v = book.get_metadata("DC", name)
            return clean_text(v[0][0]) if v else ""
        except Exception:
            return ""

    # Chapter titles from the table of contents (matched by file name)
    toc_titles = {}
    try:
        for href, title in _flatten_toc(book.toc):
            key = Path(href.split("#")[0]).name
            if key and title and key not in toc_titles:
                toc_titles[key] = clean_text(title)
    except Exception:
        pass

    chapters = []
    for idref, _linear in book.spine:
        item = book.get_item_with_id(idref)
        if item is None or item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue
        soup = BeautifulSoup(item.get_content(), "html.parser")
        for t in soup(["script", "style", "head"]):
            t.decompose()
        for br in soup.find_all("br"):
            br.replace_with("\n")
        body = soup.body or soup

        paragraphs = []
        for el in body.find_all(BLOCK_TAGS):
            if el.find(BLOCK_TAGS):        # only leaf blocks, avoids duplicates
                continue
            txt = clean_text(el.get_text())
            if txt:
                paragraphs.append(txt)
        if not paragraphs:
            txt = clean_text(body.get_text())
            if txt:
                paragraphs = [txt]

        ch = Chapter(title="", paragraphs=paragraphs)
        if ch.chars < min_chars:           # cover, copyright page, TOC, ...
            continue
        heading = body.find(["h1", "h2", "h3"])
        ch.title = (toc_titles.get(Path(item.get_name()).name)
                    or (clean_text(heading.get_text()) if heading else ""))
        chapters.append(ch)

    fill_missing_titles(chapters)

    cover, ext = None, "jpg"
    try:
        items = list(book.get_items_of_type(ebooklib.ITEM_COVER))
        if not items:
            cid = book.get_metadata("OPF", "cover")
            if cid:
                it = book.get_item_with_id(cid[0][1].get("content"))
                if it is not None:
                    items = [it]
        if not items:
            items = [i for i in book.get_items_of_type(ebooklib.ITEM_IMAGE)
                     if "cover" in i.get_name().lower()]
        if items:
            cover = items[0].get_content()
            ext = "png" if items[0].get_name().lower().endswith(".png") else "jpg"
    except Exception:
        pass

    return Book(title=meta("title") or path.stem, author=meta("creator"),
                chapters=chapters, cover=cover, cover_ext=ext)
