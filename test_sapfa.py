"""Synthetic signal test: python test_sapfa.py"""
import numpy as np

from sapfa import PRECISION, analyze

# freq, t_on, t_off, expected note, expected cents
# notes and gaps longer than the longest window, so every precision level applies
SINES = [
    (110.0,  0.6, 8.4, "La2",  0),
    (261.63, 0.6, 8.4, "Do4",  0),
    (440.0,  1.5, 4.5, "La4",  0),
    (447.0,  6.0, 8.4, "La4",  27),
    (1000.0, 3.0, 8.4, "Si5",  21),
    (3000.0, 1.5, 6.0, "Fa#7", 23),
]


def synth(sr, dur=9.0):
    t = np.arange(int(dur * sr)) / sr
    x = np.zeros_like(t)
    for f, on, off, *_ in SINES:
        env = np.clip(np.minimum(t - on, off - t) / 0.02, 0, 1)  # 20 ms ramps
        x += 0.1 * env * np.sin(2 * np.pi * f * t)
    return x


def test_synthetic():
    for sr in (44100, 96000):
        for level, win_s in PRECISION.items():
            check(synth(sr), sr, level, win_s)


def check(x, sr, level, win_s):
    *_, df = analyze(x, sr, win_s)
    print(f"\n--- sr = {sr}, precision = {level}\n{df.to_string()}")
    assert len(df) == len(SINES), len(df)
    for f, on, off, note, cents in SINES:
        row = df.iloc[(df.freq_hz - f).abs().argmin()]
        err = 1200 * np.log2(row.freq_hz / f)
        assert abs(err) < 1, (f, err)
        assert (row.note, row.cents) == (note, cents), (f, row.note, row.cents)
        assert abs(row.t_start - on) < win_s / 2 and abs(row.t_end - off) < win_s / 2, (f, row.t_start, row.t_end)


if __name__ == "__main__":
    test_synthetic()
    print("\nok")
