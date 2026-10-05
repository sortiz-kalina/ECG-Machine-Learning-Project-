"""
Raw single-lead ECG segment -> per-beat predictions.

  1. validate, resample to 360 Hz (the rate the model was trained at)
  2. z-score the segment (training z-scored each whole record)
  3. detect R peaks (Pan-Tompkins-style, NumPy only)
  4. cut windows + timing features with the SAME functions training used
  5. score with the saved ensemble and apply the chosen threshold

Depends on NumPy plus ecg_data.py / ecg_model.py (no wfdb, scikit-learn or scipy).
This is an educational demo, not a medical device.
"""
import numpy as np

from ecg_data import FS, segment_windows, rr_features, standardize_record

PRE, POST = 90, 144          # window around the R peak, in 360 Hz samples
MIN_SECONDS, MAX_SECONDS = 5.0, 120.0
MIN_RATE, MAX_RATE = 100.0, 2000.0
PEAK_SHIFT_SAMPLES = 0       # calibrate with eval_pipeline.py (median offset vs annotations)

DISCLAIMER = ("Educational demonstration only. This is not a medical device and must "
              "not be used to diagnose, monitor, or make decisions about health. It was "
              "trained on the MIT-BIH Arrhythmia Database (lead MLII) and has not been "
              "validated on other devices, leads, or populations.")


# ----------------------------------------------------------------------------
# signal processing
# ----------------------------------------------------------------------------
def bandpass(x, fs, lo, hi):
    """Zero-phase FFT band-pass (lo=0 gives a low-pass). Edges are reflect-padded
    so the circular FFT does not leak one end of the segment into the other."""
    pad = min(len(x) - 1, int(fs))
    xp = np.pad(x, pad, mode="reflect") if pad > 0 else x
    spec = np.fft.rfft(xp)
    f = np.fft.rfftfreq(len(xp), 1.0 / fs)
    spec[(f < lo) | (f > hi)] = 0
    y = np.fft.irfft(spec, len(xp))
    return y[pad:pad + len(x)] if pad > 0 else y


def resample(x, fs_in, fs_out=FS):
    if abs(fs_in - fs_out) < 1e-9:
        return np.asarray(x, dtype=float).copy()
    if fs_in > fs_out:  # remove content above the new Nyquist before decimating
        x = bandpass(x, fs_in, 0.0, 0.45 * fs_out)
    n_out = int(round(len(x) * fs_out / fs_in))
    return np.interp(np.arange(n_out) / fs_out, np.arange(len(x)) / fs_in, x)


def _beat_threshold(h):
    """
    Energy level separating real QRS complexes from P/T waves and noise bumps.
    QRS energy is typically >10x that of the junk, so heights are bimodal in log
    space: split them in two (Otsu's method) and use the gap. If the two clusters
    are NOT far apart (< ~16x in energy) there is no junk to remove, and splitting
    would only separate tall premature beats from normal ones, so keep everything
    above a fraction of the median instead.
    """
    h = np.asarray(h, dtype=float)
    fallback = 0.25 * np.median(h)
    if len(h) < 6:
        return fallback
    s = np.sort(np.log10(h + 1e-30))
    n = len(s)
    csum = np.cumsum(s)
    i = np.arange(1, n)                      # split: s[:i] | s[i:]
    m0 = csum[:-1] / i
    m1 = (csum[-1] - csum[:-1]) / (n - i)
    between = (i / n) * ((n - i) / n) * (m0 - m1) ** 2
    k = int(np.argmax(between)) + 1
    if m1[k - 1] - m0[k - 1] < 1.2:          # clusters closer than ~16x in energy
        return fallback
    return float(10 ** ((s[k - 1] + s[k]) / 2))


def detect_r_peaks(x, fs=FS):
    """
    Pan-Tompkins-style detector: band-pass 5-15 Hz -> derivative -> square ->
    150 ms moving average, then non-maximum suppression with a 200 ms refractory
    period and a threshold relative to the typical peak height. Each peak is
    refined to the largest deflection of the 5-25 Hz signal within +/-60 ms
    (absolute value, so inverted QRS complexes work). Returns sample indices.
    """
    xb = bandpass(x, fs, 5.0, 15.0)
    if np.std(xb) < 1e-3 * max(np.std(x), 1e-12):
        return np.empty(0, dtype=int)  # essentially no energy in the QRS band
    d = np.gradient(xb)
    win = max(1, int(round(0.150 * fs)))
    integ = np.convolve(d * d, np.ones(win) / win, mode="same")

    cand = np.where((integ[1:-1] > integ[:-2]) & (integ[1:-1] >= integ[2:]))[0] + 1
    if len(cand) == 0:
        return np.empty(0, dtype=int)

    refractory = int(round(0.200 * fs))
    taken = np.zeros(len(x), dtype=bool)
    kept = []
    for k in np.argsort(-integ[cand]):  # strongest first
        p = cand[k]
        if taken[max(0, p - refractory):p + refractory + 1].any():
            continue
        taken[p] = True
        kept.append(p)
    kept = np.array(sorted(kept), dtype=int)
    kept = kept[integ[kept] >= _beat_threshold(integ[kept])]
    if len(kept) == 0:
        return kept

    xr = bandpass(x, fs, 5.0, 25.0)
    half = int(round(0.060 * fs))
    refined = []
    for p in kept:
        lo, hi = max(0, p - half), min(len(x), p + half + 1)
        refined.append(lo + int(np.argmax(np.abs(xr[lo:hi]))))
    refined = np.array(sorted(set(refined)), dtype=int)
    # refinement can pull two detections together: keep the larger deflection
    out = [refined[0]]
    for p in refined[1:]:
        if p - out[-1] < refractory:
            if abs(xr[p]) > abs(xr[out[-1]]):
                out[-1] = p
        else:
            out.append(p)
    return np.clip(np.array(out, dtype=int) + PEAK_SHIFT_SAMPLES, 0, len(x) - 1)


def build_inputs(sig, peaks, use_timing):
    """
    Model inputs for the given peaks, built with the same functions training used.
    sig must already be z-scored. Returns (X, kept) where kept indexes `peaks`
    (peaks too close to the segment edges cannot be windowed).
    """
    symbols = ["N"] * len(peaks)  # real symbols are unknown; only used for beat timing
    windows, _, _, kept = segment_windows(sig, peaks, symbols, PRE, POST)
    if len(windows) == 0:
        return windows, kept
    if use_timing:
        feats = rr_features(peaks, symbols)[kept]
        return np.hstack([windows, feats]), kept
    return windows, kept


# ----------------------------------------------------------------------------
# end to end
# ----------------------------------------------------------------------------
def analyze(signal, fs, model, operating_point="best_f1",
            min_seconds=MIN_SECONDS, max_seconds=MAX_SECONDS):
    x = np.asarray(signal, dtype=float)
    if x.ndim != 1:
        raise ValueError("signal must be a flat list of numbers (one lead)")
    if not np.all(np.isfinite(x)):
        raise ValueError("signal contains NaN or infinite values")
    if not (MIN_RATE <= fs <= MAX_RATE):
        raise ValueError(f"sampling_rate must be between {MIN_RATE:g} and {MAX_RATE:g} Hz")
    duration = len(x) / fs
    if duration < min_seconds:
        raise ValueError(f"signal is {duration:.1f} s long; at least {min_seconds:g} s is needed")
    if duration > max_seconds:
        raise ValueError(f"signal is {duration:.1f} s long; the maximum is {max_seconds:g} s")
    if np.std(x) < 1e-12:
        raise ValueError("signal is flat (no variation)")
    thresholds = model.meta["thresholds"]
    if operating_point not in thresholds:
        raise ValueError(f"unknown operating_point '{operating_point}'; "
                         f"choose one of {sorted(thresholds)}")

    sig = standardize_record(resample(x, fs, FS))
    peaks = detect_r_peaks(sig)
    if len(peaks) < 3:
        raise ValueError("fewer than 3 heartbeats detected; check that the signal is a "
                         "single-lead ECG and the sampling rate is correct")

    use_timing = bool(model.meta["feature_spec"]["timing_features"])
    X, kept = build_inputs(sig, peaks, use_timing)
    if len(X) == 0:
        raise ValueError("no heartbeat has enough surrounding signal to analyze")
    proba = model.predict_proba(X)
    thr = float(thresholds[operating_point])
    flagged = proba >= thr

    rr = np.diff(peaks) / FS
    warnings = []
    if abs(fs - FS) > 1e-9:
        warnings.append(f"Signal was resampled from {fs:g} Hz to {FS} Hz.")
    if len(kept) < len(peaks):
        warnings.append(f"{len(peaks) - len(kept)} beat(s) near the segment edges were "
                        "used for timing but not scored.")
    hr = 60.0 / float(np.mean(rr))
    if hr < 30 or hr > 220:
        warnings.append(f"Detected heart rate ({hr:.0f} bpm) is outside the usual range; "
                        "peak detection may be unreliable on this signal.")
    if np.std(rr) / np.mean(rr) > 0.5:
        warnings.append("R-R intervals are very irregular. This can be real, but it can "
                        "also mean noise is confusing the peak detector.")

    scale = fs / FS
    beats = [dict(index=int(i), sample=int(round(peaks[k] * scale)),
                  time_s=round(float(peaks[k] / FS), 4),
                  probability=round(float(p), 6), flagged=bool(f))
             for i, (k, p, f) in enumerate(zip(kept, proba, flagged))]
    return dict(
        sampling_rate=float(fs), duration_s=round(duration, 3),
        operating_point=operating_point, threshold=thr,
        summary=dict(beats_detected=int(len(peaks)), beats_scored=int(len(kept)),
                     beats_flagged=int(flagged.sum()),
                     flagged_fraction=round(float(flagged.mean()), 4),
                     mean_heart_rate_bpm=round(hr, 1)),
        beats=beats, warnings=warnings, disclaimer=DISCLAIMER,
    )
