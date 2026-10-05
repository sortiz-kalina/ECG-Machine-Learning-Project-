"""
MIT-BIH Arrhythmia Database -> beat-level dataset, split by PATIENT.

Pipeline per record:
  1. read the MLII lead + beat annotations (wfdb)
  2. standardize the signal using only that record's own statistics
  3. cut a fixed window around every annotated beat (R peak)
  4. map the beat symbol to a label (normal = 0, abnormal = 1)

Splitting is done by record, never by beat: every beat from one person ends up
entirely in train or entirely in test, so the model can't "recognize the
patient" instead of learning the arrhythmia.
"""
import os
import numpy as np

FS = 360  # MIT-BIH sampling rate (Hz)

# Standard inter-patient split from de Chazal et al. (2004), used by much of the
# literature so results are comparable. Paced records 102, 104, 107, 217 are
# excluded. VERIFY these lists against the paper before quoting results.
DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124,
       201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212,
       213, 214, 219, 221, 222, 228, 231, 232, 233, 234]

# AAMI EC57 grouping of MIT-BIH beat symbols.
AAMI = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",   # normal-ish
    "A": "S", "a": "S", "J": "S", "S": "S",             # supraventricular
    "V": "V", "E": "V",                                 # ventricular
    "F": "F",                                           # fusion
    # "Q" class (paced / unclassifiable: '/', 'f', 'Q') is dropped on purpose.
}
BINARY = {"N": 0, "S": 1, "V": 1, "F": 1}  # 1 = abnormal beat


# ----------------------------------------------------------------------------
# Pure NumPy logic (testable without any data files)
# ----------------------------------------------------------------------------
# Annotation symbols that mark an actual heartbeat (everything else, like '+'
# or '~', marks rhythm changes or noise). Used to measure beat-to-beat timing.
BEAT_SYMBOLS = set("NLRBAaJSVrFejnE/fQ?")
RR_FEATURE_NAMES = ["rr_prev_s", "rr_next_s", "log_rr_prev_ratio", "log_rr_next_ratio"]


def rr_features(sample_idx, symbols, fs=FS, window=10):
    """
    Timing features for every annotation (rows for non-beat annotations are 0):
      rr_prev_s          seconds since the previous beat
      rr_next_s          seconds until the next beat
      log_rr_prev_ratio  log(rr_prev / mean of the last `window` intervals)
      log_rr_next_ratio  log(rr_next / that same mean)
    A premature beat has a short rr_prev (negative ratio) and usually a long
    rr_next. The first/last beat borrows its one available interval.
    """
    sample_idx = np.asarray(sample_idx)
    feats = np.zeros((len(sample_idx), 4))
    beat_pos = [i for i, sym in enumerate(symbols) if sym in BEAT_SYMBOLS]
    if len(beat_pos) < 2:
        return feats
    t = sample_idx[beat_pos] / fs
    rr = np.maximum(np.diff(t), 1e-3)  # rr[j] = time from beat j to beat j+1
    n = len(t)
    for j in range(n):
        prev = rr[j - 1] if j >= 1 else rr[0]
        nxt = rr[j] if j < n - 1 else rr[-1]
        recent = rr[max(0, j - window):j]  # the intervals up to and incl. rr_prev
        local = recent.mean() if len(recent) else prev
        feats[beat_pos[j]] = [prev, nxt, np.log(prev / local), np.log(nxt / local)]
    return feats


def _segment(signal, sample_idx, symbols, pre, post):
    """Core windowing. Also returns the indices of annotations that were kept."""
    beats, labels, classes, kept = [], [], [], []
    n = len(signal)
    for i, (r, sym) in enumerate(zip(sample_idx, symbols)):
        cls = AAMI.get(sym)
        if cls is None:
            continue  # rhythm markers, noise, paced/unclassifiable beats...
        if r - pre < 0 or r + post > n:
            continue
        beats.append(signal[r - pre: r + post])
        labels.append(BINARY[cls])
        classes.append(cls)
        kept.append(i)
    if not beats:
        return (np.empty((0, pre + post)), np.empty(0, dtype=int), [],
                np.empty(0, dtype=int))
    return (np.asarray(beats), np.asarray(labels, dtype=int), classes,
            np.asarray(kept, dtype=int))


def segment_beats(signal, sample_idx, symbols, pre=90, post=144):
    """
    Cut a window [r - pre, r + post) around each annotated beat.

    Returns (beats, labels, kept_symbols):
      beats   (n, pre+post) float array
      labels  (n,) int array, 0 normal / 1 abnormal
      kept_symbols  list of AAMI class letters ('N','S','V','F') per beat
    Beats near the signal edges, and non-beat or unmapped annotations, are skipped.
    """
    beats, labels, classes, _ = _segment(signal, sample_idx, symbols, pre, post)
    return beats, labels, classes


def standardize_record(signal):
    """Zero-mean / unit-variance using this record only (no cross-patient leak)."""
    return (signal - signal.mean()) / (signal.std() + 1e-8)


# ----------------------------------------------------------------------------
# File reading (needs wfdb and the downloaded data)
# ----------------------------------------------------------------------------
def load_record(data_dir, rec, pre=90, post=144):
    import wfdb  # imported here so the pure functions work without it

    path = os.path.join(data_dir, str(rec))
    record = wfdb.rdrecord(path)
    ann = wfdb.rdann(path, "atr")

    # Pick the MLII lead by NAME: its channel index differs between records.
    names = [n.strip() for n in record.sig_name]
    if "MLII" not in names:
        return None  # a few records lack it; skip rather than guess
    sig = record.p_signal[:, names.index("MLII")]

    sig = standardize_record(sig)
    beats, labels, classes, kept = _segment(sig, ann.sample, ann.symbol, pre, post)
    # timing features come from ALL beat annotations, then are matched to kept beats
    feats = rr_features(ann.sample, ann.symbol)[kept]
    return beats, labels, classes, feats


def load_records(data_dir, records, pre=90, post=144, verbose=True):
    """Returns X, y, groups (record id per beat), classes (AAMI letter per beat),
    F (timing features per beat)."""
    Xs, ys, gs, cs, fs_ = [], [], [], [], []
    for rec in records:
        out = load_record(data_dir, rec, pre, post)
        if out is None:
            if verbose:
                print(f"  record {rec}: no MLII lead, skipped")
            continue
        beats, labels, classes, feats = out
        Xs.append(beats)
        ys.append(labels)
        gs.append(np.full(len(labels), rec))
        cs.extend(classes)
        fs_.append(feats)
        if verbose:
            print(f"  record {rec}: {len(labels):5d} beats, "
                  f"{labels.mean():.1%} abnormal")
    return (np.concatenate(Xs), np.concatenate(ys),
            np.concatenate(gs), np.asarray(cs), np.concatenate(fs_))


def load_dataset(data_dir, cache="ecg_cache_v3.npz"):
    """
    Development set = all of DS1 (22 patients), test set = DS2 (22 patients).
    Keeps beat types (AAMI letters N/S/V/F) and timing features (F_*), one row
    per beat. Parsing is cached since it takes a while.
    """
    if cache and os.path.exists(cache):
        return dict(np.load(cache, allow_pickle=False))

    print("development records (DS1):")
    Xd, yd, gd, cd, Fd = load_records(data_dir, DS1)
    print("test records (DS2):")
    Xt, yt, gt, ct, Ft = load_records(data_dir, DS2)

    assert not (set(gd) & set(gt)), "patient leakage between dev and test!"

    out = dict(X_dev=Xd, y_dev=yd, g_dev=gd, c_dev=cd, F_dev=Fd,
               X_test=Xt, y_test=yt, g_test=gt, c_test=ct, F_test=Ft)
    if cache:
        np.savez_compressed(cache, **out)
    return out


def grouped_folds(y, groups, k=5, seed=0):
    """
    Split PATIENTS (not beats) into k folds, balanced by abnormal-beat count.

    Records vary wildly (0% to 77% abnormal), so random folds can end up with
    almost no abnormal beats. Greedy balancing: give each record, largest
    abnormal count first, to the fold that currently has the fewest.
    Returns a list of k arrays of record ids.
    """
    rng = np.random.default_rng(seed)
    ids = np.unique(groups)
    rng.shuffle(ids)  # random tie-breaking between records with equal counts
    counts = {r: int(y[groups == r].sum()) for r in ids}
    ids = sorted(ids, key=lambda r: -counts[r])
    folds = [[] for _ in range(k)]
    totals = [0] * k
    for r in ids:
        j = int(np.argmin(totals))
        folds[j].append(r)
        totals[j] += counts[r]
    return [np.asarray(f) for f in folds]