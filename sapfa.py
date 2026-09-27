"""SAPFA - Spectral Audio Partials Frequency Annotator.

Usage: python sapfa.py input.wav|flac [-p low|mid|high|extra] [-o outdir] [--dpi N]
"""
import argparse
import math
from pathlib import Path

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from matplotlib.ticker import ScalarFormatter
from scipy.signal import find_peaks
from scipy.signal.windows import blackmanharris
from tqdm import tqdm

# analysis window length in seconds: longer = finer in frequency, blurrier in time
PRECISION = {"low": 0.17, "mid": 0.34, "high": 0.68, "extra": 1.36}
HOP = 512
FMIN, FMAX = 15, 5000
NOISE_DB = 15       # minimum peak height above the local noise floor
FLOOR_HZ = 300      # width of the median filter that estimates the noise floor
THRESHOLD_DB = -80  # relative to the file maximum: cuts the window sidelobes
TOL_CENTS = 30      # max jump of a partial between two frames
MAX_GAP = 2         # missing frames tolerated inside a track
SPLIT_CENTS = 40    # max deviation from the track mean before splitting (note change)
MIN_DUR = 0.1       # s; with long windows also at least half a window (shorter = artifacts)
LABEL_MIN_DUR = 0.5  # s, shorter tracks are not labeled
A4 = 440.0
ACCENT = "#1de9b6"  # aqua green for tracks and labels
DB_RANGE = 70       # dynamic range shown in the spectrogram
NOTE_NAMES = ["Do", "Do#", "Re", "Re#", "Mi", "Fa", "Fa#", "Sol", "Sol#", "La", "La#", "Si"]


def load_mono(path):
    x, sr = sf.read(path, always_2d=True)
    return x.mean(axis=1), sr


def spectrogram(x, sr, win_s):
    n_fft = 2 ** round(np.log2(sr * win_s))
    win = blackmanharris(n_fft)
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    keep = freqs <= FMAX
    frames = np.lib.stride_tricks.sliding_window_view(np.pad(x, n_fft // 2), n_fft)[::HOP]
    mag = np.concatenate([np.abs(np.fft.rfft(frames[i:i + 256] * win))[:, keep]
                          for i in range(0, len(frames), 256)]) * 2 / win.sum()
    db = 20 * np.log10(mag + 1e-12)
    times = np.arange(len(frames)) * HOP / sr  # frame centers
    return db, times, freqs[keep]


def pick_peaks(spectrum, bin_hz, threshold):
    """Local maxima above threshold, parabolic interpolation on dB."""
    k, _ = find_peaks(spectrum, height=threshold)
    a, b, c = spectrum[k - 1], spectrum[k], spectrum[k + 1]
    p = 0.5 * (a - c) / (a - 2 * b + c)
    f, amp = (k + p) * bin_hz, b - 0.25 * (a - c) * p
    return f[f >= FMIN], amp[f >= FMIN]


def track_partials(db, bin_hz):
    """Link peaks frame to frame. Each track is a list of (frame, freq, amp)."""
    active, done = [], []
    floor_bins = round(FLOOR_HZ / bin_hz)
    top = db.max()
    for i, spectrum in enumerate(tqdm(db, desc="analyzing", unit="frame")):
        floor = pd.Series(spectrum).rolling(floor_bins, center=True, min_periods=1).median().to_numpy()
        freqs, amps = pick_peaks(spectrum, bin_hz, np.maximum(floor + NOISE_DB, top + THRESHOLD_DB))
        done += [t for t in active if i - t[-1][0] > MAX_GAP + 1]
        active = [t for t in active if i - t[-1][0] <= MAX_GAP + 1]

        # greedy matching: closest peak-track pairs (in cents) first
        used_p, used_t = set(), set()
        if active and len(freqs):
            last = np.array([t[-1][1] for t in active])
            dist = np.abs(1200 * np.log2(freqs[:, None] / last[None, :]))
            mask = dist < TOL_CENTS
            for p, j in np.argwhere(mask)[np.argsort(dist[mask])]:
                if p in used_p or j in used_t:
                    continue
                used_p.add(p)
                used_t.add(j)
                active[j].append((i, freqs[p], amps[p]))
        active += [[(i, freqs[p], amps[p])] for p in range(len(freqs)) if p not in used_p]
    return done + active


def split_drift(track):
    """Split the track when its frequency drifts away from the mean of the current piece."""
    pieces, start, total = [], 0, 0.0
    for i, c in enumerate(1200 * np.log2(track[:, 1])):
        if i > start and abs(c - total / (i - start)) > SPLIT_CENTS:
            pieces.append(track[start:i])
            start, total = i, 0.0
        total += c
    return pieces + [track[start:]]


def note_name(freq):
    midi = 69 + 12 * math.log2(freq / A4)
    n = round(midi)
    return f"{NOTE_NAMES[n % 12]}{n // 12 - 1}", round(100 * (midi - n))


def analyze(x, sr, win_s=PRECISION["high"]):
    db, times, freqs = spectrogram(x, sr, win_s)
    tracks = [p for t in track_partials(db, freqs[1]) for p in split_drift(np.array(t))]
    min_dur = max(MIN_DUR, win_s / 2)
    tracks = [t for t in tracks if times[int(t[-1, 0])] - times[int(t[0, 0])] >= min_dur]
    tracks.sort(key=lambda t: (t[0, 0], t[0, 1]))

    rows = []
    for i, t in enumerate(tracks):
        f = np.median(t[:, 1])
        note, cents = note_name(f)
        rows.append(dict(track_id=i, t_start=times[int(t[0, 0])], t_end=times[int(t[-1, 0])],
                         freq_hz=f, amp_db=t[:, 2].max(), note=note, cents=cents))
    return times, tracks, pd.DataFrame(rows)


def plot(x, sr, times, tracks, df, path, dpi=500):
    # constant-Q background: uniform line thickness on the log axis (measurements stay on the STFT)
    bpo = 48
    cqt = librosa.cqt(x, sr=sr, hop_length=HOP, fmin=FMIN, n_bins=int(bpo * np.log2(FMAX / FMIN)),
                      bins_per_octave=bpo, scale=False)
    cqt_db = 20 * np.log10(2 * np.abs(cqt) + 1e-12)
    cqt_freqs = librosa.cqt_frequencies(len(cqt), fmin=FMIN, bins_per_octave=bpo)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(16, 18))
    vmax = cqt_db.max()
    ax.pcolormesh(times, cqt_freqs, cqt_db[:, :len(times)], cmap="gray", vmin=vmax - DB_RANGE, vmax=vmax)
    ax.set_yscale("log")
    ax.set_ylim(FMIN, FMAX)
    ax.set_yticks([20, 50, 100, 200, 500, 1000, 2000, 5000])
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")

    # labels: strongest tracks first; if the spot is taken, slide forward along the track
    renderer = fig.canvas.get_renderer()
    placed = []
    for i in df.amp_db.to_numpy().argsort()[::-1]:
        t, row = tracks[i], df.iloc[i]
        if row.t_end - row.t_start < LABEL_MIN_DUR:
            continue
        ax.plot(times[t[:, 0].astype(int)], t[:, 1], color=ACCENT, lw=0.3)
        label = ax.text(0, 0, f"{row.freq_hz:.1f} Hz {row.note} {row.cents:+d}c", color=ACCENT,
                        fontsize=4, va="bottom",
                        bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6, lw=0))
        for j in range(0, len(t), max(1, len(t) // 50)):
            label.set_position((times[int(t[j, 0])], t[j, 1]))
            box = label.get_window_extent(renderer).padded(6)  # include the label box
            if box.x1 < ax.bbox.x1 and not any(box.overlaps(b) for b in placed):
                placed.append(box)
                break
        else:
            label.remove()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser(description="Spectrogram with labeled partials + CSV.")
    ap.add_argument("audio", help="WAV, FLAC, AIFF or any format read by libsndfile")
    ap.add_argument("-o", "--outdir", help="output folder (default: next to the wav)")
    ap.add_argument("-p", "--precision", choices=PRECISION, default="high",
                    help="low: fast, sharp in time; high: slow, separates close partials (default: high)")
    ap.add_argument("--dpi", type=int, default=500, help="PNG resolution (default: 500)")
    args = ap.parse_args()

    audio = Path(args.audio)
    outdir = Path(args.outdir or audio.parent)
    outdir.mkdir(parents=True, exist_ok=True)
    base = outdir / f"{audio.stem}_{args.precision}"

    x, sr = load_mono(audio)
    times, tracks, df = analyze(x, sr, PRECISION[args.precision])
    df.to_csv(f"{base}.csv", index=False, float_format="%.3f")
    plot(x, sr, times, tracks, df, f"{base}.png", args.dpi)
    print(f"{base.name}.png saved in {outdir}/")


if __name__ == "__main__":
    main()
