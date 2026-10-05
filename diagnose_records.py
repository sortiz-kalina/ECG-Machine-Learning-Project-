"""
Diagnostics for the false-alarm records (117, 214) on MIT-BIH.

  1. Symbol counts for all 44 records, with a table of bundle-branch-block
     patients (beats 'L' and 'R') and which split they fall in.
  2. A numeric check of how far each record's average normal beat is from the
     other records' (is 117 really an outlier?).
  3. A figure saved to ecg_diagnostics.png.


"""
import os
import sys
import collections
import numpy as np
import matplotlib
matplotlib.use("Agg")  # write a file; no window needed
import matplotlib.pyplot as plt

from ecg_data import DS1, DS2, FS, segment_beats, standardize_record

PRE, POST = 90, 144
FOCUS_NORMAL, FOCUS_BBB = 117, 214


# ----------------------------------------------------------------------------
# reading (the only part that needs wfdb)
# ----------------------------------------------------------------------------
def read_record(data_dir, rec):
    import wfdb
    path = os.path.join(data_dir, str(rec))
    record = wfdb.rdrecord(path)
    ann = wfdb.rdann(path, "atr")
    names = [n.strip() for n in record.sig_name]
    if "MLII" not in names:
        return None
    sig = standardize_record(record.p_signal[:, names.index("MLII")])
    return sig, np.asarray(ann.sample), list(ann.symbol)


def beats_for(rec_data, wanted):
    """Windows around beats whose raw symbol is in `wanted` (e.g. {'L'})."""
    sig, samples, syms = rec_data
    keep = [i for i, s in enumerate(syms) if s in wanted]
    if not keep:
        return np.empty((0, PRE + POST))
    beats, _, _ = segment_beats(sig, samples[keep], [syms[i] for i in keep],
                                PRE, POST)
    return beats


# ----------------------------------------------------------------------------
# 1. symbol counts / bundle branch table
# ----------------------------------------------------------------------------
def split_of(rec):
    return "dev (DS1)" if rec in DS1 else "test (DS2)"


def bundle_branch_table(counts):
    rows = [(r, split_of(r), c["L"], c["R"], sum(c.values()))
            for r, c in sorted(counts.items()) if c["L"] or c["R"]]
    print("\nrecords containing bundle branch block beats (L = left, R = right):")
    print(f"  {'record':>6s}  {'split':10s} {'L beats':>8s} {'R beats':>8s} "
          f"{'% of annotations':>17s}")
    for r, sp, l, rr, tot in rows:
        print(f"  {r:6d}  {sp:10s} {l:8d} {rr:8d} {(l + rr) / tot:17.1%}")
    for sp in ("dev (DS1)", "test (DS2)"):
        n = sum(1 for r in rows if r[1] == sp)
        beats = sum(r[2] + r[3] for r in rows if r[1] == sp)
        print(f"  {sp}: {n} record(s), {beats} bundle-branch beats")


# ----------------------------------------------------------------------------
# 2. outlier ranking of average normal beats
# ----------------------------------------------------------------------------
def mean_normal_beats(data, min_beats=50):
    means = {}
    for rec, rd in data.items():
        b = beats_for(rd, {"N"})
        if len(b) >= min_beats:
            means[rec] = b.mean(axis=0)
    return means


def outlier_ranking(means, top=6):
    """RMS distance from each record's mean normal beat to the average of the
    OTHER records' means. Larger = looks less like everyone else."""
    recs = list(means)
    stack = np.array([means[r] for r in recs])
    dist = {}
    for i, r in enumerate(recs):
        ref = np.delete(stack, i, axis=0).mean(axis=0)
        dist[r] = float(np.sqrt(np.mean((stack[i] - ref) ** 2)))
    ranked = sorted(dist.items(), key=lambda kv: -kv[1])
    print(f"\nrecords whose average NORMAL beat differs most from the others "
          f"(of {len(recs)} records with >= 50 'N' beats):")
    for rank, (r, d) in enumerate(ranked[:top], 1):
        flag = "  <-- focus record" if r == FOCUS_NORMAL else ""
        print(f"  #{rank}  record {r}: RMS distance {d:.3f}{flag}")
    if FOCUS_NORMAL in dist:
        pos = [r for r, _ in ranked].index(FOCUS_NORMAL) + 1
        print(f"  record {FOCUS_NORMAL} ranks #{pos} of {len(ranked)} "
              f"(median distance {np.median(list(dist.values())):.3f})")
    else:
        print(f"  record {FOCUS_NORMAL} was not ranked (too few 'N' beats)")
    return dist


# ----------------------------------------------------------------------------
# 3. figure
# ----------------------------------------------------------------------------
def make_figure(data, means, path="ecg_diagnostics.png", seed=0):
    rng = np.random.default_rng(seed)
    t_ms = (np.arange(PRE + POST) - PRE) / FS * 1000.0
    others = [r for r in means if r not in (FOCUS_NORMAL, FOCUS_BBB)]
    ref_mean = np.mean([means[r] for r in others], axis=0)

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6), sharey=True)

    # A: every record's average normal beat, focus record highlighted
    for r, m in means.items():
        if r == FOCUS_NORMAL:
            continue
        ax[0].plot(t_ms, m, color="0.75", lw=0.8)
    if FOCUS_NORMAL in means:
        ax[0].plot(t_ms, means[FOCUS_NORMAL], color="crimson", lw=2.2,
                   label=f"record {FOCUS_NORMAL}")
    ax[0].plot(t_ms, ref_mean, color="black", lw=1.6, ls="--",
               label="average of other records")
    ax[0].set_title("Average normal beat, one line per record")
    ax[0].legend()

    # B: record 117's individual normal beats vs the typical normal beat
    b117 = beats_for(data[FOCUS_NORMAL], {"N"}) if FOCUS_NORMAL in data else []
    if len(b117):
        pick = rng.choice(len(b117), size=min(60, len(b117)), replace=False)
        for i in pick:
            ax[1].plot(t_ms, b117[i], color="crimson", alpha=0.12, lw=0.8)
        ax[1].plot(t_ms, b117.mean(axis=0), color="crimson", lw=2,
                   label=f"record {FOCUS_NORMAL} mean")
    ax[1].plot(t_ms, ref_mean, color="black", lw=1.6, ls="--",
               label="typical normal beat")
    ax[1].set_title(f"Record {FOCUS_NORMAL}: individual normal beats")
    ax[1].legend()

    # C: record 214 - left bundle branch beats vs ventricular vs normal
    if FOCUS_BBB in data:
        for sym, color, name in [("L", "tab:orange", "L (left bundle branch)"),
                                 ("V", "tab:purple", "V (ventricular)")]:
            b = beats_for(data[FOCUS_BBB], {sym})
            if len(b):
                m, s = b.mean(axis=0), b.std(axis=0)
                ax[2].plot(t_ms, m, color=color, lw=2,
                           label=f"record {FOCUS_BBB}: {name}, n={len(b)}")
                ax[2].fill_between(t_ms, m - s, m + s, color=color, alpha=0.15)
    ax[2].plot(t_ms, ref_mean, color="black", lw=1.6, ls="--",
               label="typical normal beat")
    ax[2].set_title(f"Record {FOCUS_BBB}: 'normal' L beats vs V beats")
    ax[2].legend(fontsize=8)

    for a in ax:
        a.axvline(0, color="0.6", lw=0.6)
        a.set_xlabel("ms from R peak")
    ax[0].set_ylabel("standardized amplitude")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"\nsaved figure: {path}")



def main(data_dir="data/mitdb"):
    data, counts = {}, {}
    print("reading records...")
    for rec in DS1 + DS2:
        rd = read_record(data_dir, rec)
        if rd is None:
            print(f"  record {rec}: no MLII lead, skipped")
            continue
        data[rec] = rd
        counts[rec] = collections.Counter(rd[2])
    bundle_branch_table(counts)
    means = mean_normal_beats(data)
    outlier_ranking(means)
    make_figure(data, means)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/mitdb")