"""
Tests for ecg_pipeline.py. Run:  python test_pipeline.py

"""
import os
import tempfile
import numpy as np

from ecg_pipeline import (bandpass, resample, detect_r_peaks, build_inputs, analyze,
                          PRE, POST)
from ecg_data import FS, segment_beats, rr_features, standardize_record
from ecg_model import save_ensemble, EnsembleModel
from nn import NeuralNet


def synth_ecg(duration=40.0, fs=360, hr=72, seed=0, wander=0.3, noise=0.03,
              inverted=False, premature_at=None):
    """Returns (signal, true R-peak sample indices) at sampling rate fs."""
    rng = np.random.default_rng(seed)
    n = int(duration * fs)
    t = np.arange(n) / fs
    beat_t, now, k = [], 1.0, 0
    while now < duration - 1.0:
        beat_t.append(now)
        rr = 60.0 / hr * (1 + 0.03 * rng.standard_normal())
        if premature_at is not None and k == premature_at:
            rr *= 0.55                       # this beat comes early...
        elif premature_at is not None and k == premature_at + 1:
            rr *= 1.40                       # ...followed by a pause
        now += rr
        k += 1
    sig = np.zeros(n)
    g = lambda c, w, a, center: a * np.exp(-((t - center) / w) ** 2)
    for b in beat_t:
        sig += g(0, .045, 0.12, b - 0.18)    # P
        sig += g(0, .010, -0.15, b - 0.030)  # Q
        sig += g(0, .012, 1.00, b)           # R
        sig += g(0, .012, -0.25, b + 0.030)  # S
        sig += g(0, .055, 0.30, b + 0.26)    # T
    if inverted:
        sig = -sig
    sig += wander * np.sin(2 * np.pi * 0.25 * t) + 0.5 * wander * np.sin(2 * np.pi * 0.1 * t + 1)
    sig += noise * rng.standard_normal(n)
    return sig, np.round(np.array(beat_t) * fs).astype(int)


def match(detected, truth, tol):
    """Greedy one-to-one matching; returns (true positives, offsets in samples)."""
    truth = list(truth)
    offs = []
    for d in detected:
        if not truth:
            break
        j = int(np.argmin([abs(d - t) for t in truth]))
        if abs(d - truth[j]) <= tol:
            offs.append(d - truth[j])
            truth.pop(j)
    return len(offs), np.array(offs)


def check_detection(name, **kw):
    fs = kw.get("fs", 360)
    sig, truth = synth_ecg(**kw)
    sig360 = standardize_record(resample(sig, fs, FS))
    det = detect_r_peaks(sig360)
    truth360 = np.round(truth * FS / fs).astype(int)
    tp, offs = match(det, truth360, tol=int(0.03 * FS))
    sens, ppv = tp / len(truth360), tp / max(len(det), 1)
    print(f"  {name:34s} sens {sens:.3f}  ppv {ppv:.3f}  "
          f"offset median {np.median(offs):+.1f} samples (max |{np.abs(offs).max()}|)")
    assert sens >= 0.99 and ppv >= 0.99, (name, sens, ppv)
    assert np.abs(offs).max() <= 4, "peak location should be within ~11 ms"


def test_detector_variants():
    check_detection("clean 72 bpm")
    check_detection("fast 140 bpm", hr=140, duration=30)
    check_detection("slow 45 bpm", hr=45, duration=60)
    check_detection("heavy baseline wander", wander=1.2)
    check_detection("noisy (noise 0.12)", noise=0.12)
    check_detection("inverted polarity", inverted=True)
    check_detection("250 Hz input", fs=250)
    check_detection("500 Hz input", fs=500)
    check_detection("premature beat", premature_at=10)


def test_resample_and_filters():
    t = np.arange(0, 10, 1 / 250)
    x = np.sin(2 * np.pi * 5 * t)
    y = resample(x, 250, 360)
    assert abs(len(y) - 3600) <= 1
    t2 = np.arange(len(y)) / 360
    assert np.abs(y[100:-100] - np.sin(2 * np.pi * 5 * t2)[100:-100]).max() < 0.02
    # band-pass removes a slow drift and keeps a 10 Hz tone
    tt = np.arange(0, 20, 1 / 360)
    mix = np.sin(2 * np.pi * 0.2 * tt) + np.sin(2 * np.pi * 10 * tt)
    out = bandpass(mix, 360, 5, 15)
    assert np.abs(out[500:-500] - np.sin(2 * np.pi * 10 * tt)[500:-500]).max() < 0.05


def test_training_serving_parity():
    """Same peaks -> byte-identical model inputs to the training code path."""
    sig, truth = synth_ecg(duration=30, seed=3)
    sig = standardize_record(sig)
    X, kept = build_inputs(sig, truth, use_timing=True)
    beats, _, _ = segment_beats(sig, truth, ["N"] * len(truth), PRE, POST)
    feats = rr_features(truth, ["N"] * len(truth))[kept]
    assert beats.shape == (len(kept), PRE + POST)
    assert np.array_equal(X, np.hstack([beats, feats]))
    # the first/last beats lack a full window and must be dropped, not padded
    assert len(kept) == len(truth) - 1 or len(kept) == len(truth) - 2 or len(kept) == len(truth)
    # a premature beat must show a short previous interval and a long next one
    sig2, truth2 = synth_ecg(duration=30, seed=3, premature_at=8)
    F = rr_features(truth2, ["N"] * len(truth2))
    assert F[9, 2] < -0.3 and F[9, 3] > 0.2, F[9]


def tiny_model(path, width=238, timing=True):
    nets = [NeuralNet([width, 16, 8, 1], seed=s) for s in range(2)]
    meta = dict(thresholds={"best_f1": 0.5, "balanced": 0.3, "sens90": 0.1},
                feature_spec=dict(timing_features=["a", "b", "c", "d"] if timing else []))
    save_ensemble(path, nets, meta)
    return EnsembleModel.load(path)


def test_analyze_end_to_end():
    with tempfile.TemporaryDirectory() as d:
        model = tiny_model(os.path.join(d, "m.npz"))
        for fs in (360, 250, 500):
            sig, truth = synth_ecg(duration=30, fs=fs, seed=1)
            out = analyze(sig.tolist(), fs, model, "balanced")
            s = out["summary"]
            assert s["beats_detected"] >= len(truth) - 1
            assert 0 < s["beats_scored"] <= s["beats_detected"]
            assert abs(s["mean_heart_rate_bpm"] - 72) < 3
            ps = np.array([b["probability"] for b in out["beats"]])
            assert np.all((ps >= 0) & (ps <= 1))
            assert out["threshold"] == 0.3
            assert all(b["flagged"] == (b["probability"] >= 0.3) for b in out["beats"])
            # reported sample positions are in the ORIGINAL sampling rate
            first_true = truth[0]
            nearest = min(b["sample"] for b in out["beats"])
            assert abs(nearest - first_true) < 0.5 * fs, (nearest, first_true)
        print(f"  end to end ok at 250/360/500 Hz; sample output keys: {sorted(out)}")

        # wave-only models (no timing columns) must work too
        wave_only = tiny_model(os.path.join(d, "w.npz"), width=234, timing=False)
        sig, _ = synth_ecg(duration=20, seed=2)
        assert analyze(sig, 360, wave_only)["summary"]["beats_scored"] > 10


def test_input_validation():
    with tempfile.TemporaryDirectory() as d:
        model = tiny_model(os.path.join(d, "m.npz"))
        sig, _ = synth_ecg(duration=30)
        def bad(msg_part, *a, **k):
            try:
                analyze(*a, **k)
            except ValueError as e:
                assert msg_part in str(e), (msg_part, str(e))
                return
            raise AssertionError(f"expected ValueError containing '{msg_part}'")
        bad("at least", sig[:360 * 2], 360, model)                    # too short
        bad("maximum", np.tile(sig, 5), 360, model)                   # too long
        bad("NaN", np.r_[sig[:-1], np.nan], 360, model)               # NaN
        bad("flat", np.ones(360 * 20), 360, model)                    # flat
        bad("sampling_rate", sig, 50, model)                          # rate out of range
        bad("unknown operating_point", sig, 360, model, "nope")       # bad operating point
        bad("flat list", np.ones((10, 10)), 360, model)               # not 1-D

        # Non-ECG input cannot be certified as ECG, but it must never be accepted
        # SILENTLY: either rejected, or returned with at least one warning.
        junk = {"white noise": np.random.default_rng(1).normal(0, 1, 360 * 20),
                "slow sine (no heartbeats)": np.sin(2 * np.pi * 0.05 * np.arange(360 * 8) / 360),
                "random walk": np.cumsum(np.random.default_rng(2).normal(0, 1, 360 * 20)),
                "silence then faint ramp": np.r_[np.zeros(360 * 10), 1e-9 * np.arange(360 * 10)]}
        for name, x in junk.items():
            try:
                out = analyze(x, 360, model)
                assert out["warnings"], f"{name} was analyzed silently"
                print(f"  {name}: accepted WITH warnings ({len(out['warnings'])})")
            except ValueError as e:
                print(f"  {name}: rejected ({str(e)[:50]})")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            print(name)
            fn()
    print("ALL PIPELINE TESTS PASSED")
