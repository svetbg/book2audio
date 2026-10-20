import re

import numpy as np

from book2audio.config import CHARS_PER_SEC


def silence(secs, sr):
    return np.zeros(int(round(secs * sr)), dtype=np.float32)


def trim_silence(a, sr, rel_thresh=0.03, pad=0.06):
    if a.size == 0:
        return a
    peak = float(np.max(np.abs(a)))
    if peak <= 0:
        return a
    idx = np.where(np.abs(a) > rel_thresh * peak)[0]
    if idx.size == 0:
        return a
    s = max(0, idx[0] - int(pad * sr))
    e = min(len(a), idx[-1] + int(pad * sr))
    return a[s:e]


def normalize(a, target_dbfs=-21.0):
    if a.size == 0:
        return a
    rms = float(np.sqrt(np.mean(a ** 2)))
    if rms < 1e-6:
        return a
    a = a * (10 ** (target_dbfs / 20) / rms)
    peak = float(np.max(np.abs(a)))
    if peak > 0.97:
        a = a * (0.97 / peak)
    return a.astype(np.float32)


def expected_secs(text):
    return max(0.6, len(re.findall(r"[^\W_]", text)) / CHARS_PER_SEC)


def plausible(dur, text):
    e = expected_secs(text)
    return 0.4 * e - 0.3 <= dur <= 2.2 * e + 1.5


def fmt_time(secs):
    secs = int(max(0, secs))
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"
