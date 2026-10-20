import sys
from pathlib import Path

if not __package__ and Path(sys.path[0]).resolve() == Path(__file__).resolve().parent:
    # run by path: swap the package dir for its parent on sys.path, so our logging.py
    # does not shadow the stdlib logging module
    sys.path[0] = str(Path(__file__).resolve().parent.parent)

from book2audio.app import Book2Audio
from book2audio.args import parse_arguments
from book2audio.exceptions import Book2AudioException
from book2audio.logging import error


def main():
    args = parse_arguments()
    try:
        Book2Audio(args).run()
    except Book2AudioException as e:
        error(f"\nERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
