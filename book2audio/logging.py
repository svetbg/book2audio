import sys

debug_mode = False


def debug(msg: str, flush: bool = False):
    """Print debug msg"""
    if debug_mode:
        print(f"DEBUG {msg}", flush=flush)


def log(msg: str = "", flush: bool = False):
    """Display msg in log"""
    print(msg, flush=flush)


def error(msg: str):
    """Display error msg on stderr"""
    print(msg, file=sys.stderr)
