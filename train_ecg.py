
"""
ECG training (v4): ablation study, then a multi-seed ensemble.
 
Stage 1  Compare six model configurations with 5-fold patient-grouped
         cross-validation on the DEVELOPMENT patients only (DS1):
           wave only -> + timing features -> + dropout / augmentation / smaller net
         Each config trains every fold for one shared epoch count (picked from the
         averaged validation-AUC curve), giving comparable out-of-fold predictions.
Stage 2  The config with the best pooled out-of-fold average precision is trained
         with several seeds. Averaged out-of-fold predictions set the thresholds.
Stage 3  A final ensemble is trained on all dev patients and evaluated ONCE on the
         test patients (DS2). The ensemble is saved for the web app.
 
Usage:  python train_ecg.py [path/to/mitdb]      (default: data/mitdb)
"""
import sys
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
 
from nn import (NeuralNet, gradient_check, classification_report, roc_auc,
                average_precision)
from ecg_data import load_dataset, grouped_folds, RR_FEATURE_NAMES, FS
from ecg_model import save_ensemble
 
N_WAVE = 234  # 90 samples before + 144 after the R peak
TYPE_NAMES = {"N": "normal", "S": "supraventricular", "V": "ventricular",
              "F": "fusion"}
 
CONFIGS = [
    dict(name="wave only",                 rr=False, dropout=0.0, aug=False, hidden=(64, 32)),
    dict(name="+ timing features",         rr=True,  dropout=0.0, aug=False, hidden=(64, 32)),
    dict(name="+ dropout 0.3",             rr=True,  dropout=0.3, aug=False, hidden=(64, 32)),
    dict(name="+ augmentation",            rr=True,  dropout=0.0, aug=True,  hidden=(64, 32)),
    dict(name="+ dropout + augmentation",  rr=True,  dropout=0.3, aug=True,  hidden=(64, 32)),
    dict(name="smaller (32,16) + both",    rr=True,  dropout=0.3, aug=True,  hidden=(32, 16)),
]
 
 
# ----------------------------------------------------------------------------
# inputs, class weights, augmentation
# ----------------------------------------------------------------------------
def make_inputs(d, split, rr):
    X = d[f"X_{split}"]
    return np.hstack([X, d[f"F_{split}"]]) if rr else X
 
 
def make_class_weights(y, mode):
    ratio = (len(y) - y.sum()) / max(y.sum(), 1)
    return {"none": (1.0, 1.0),
            "sqrt": (1.0, float(np.sqrt(ratio))),
            "full": (1.0, float(ratio))}[mode]
 
 
def make_augment(n_wave=N_WAVE, noise_std=0.05, amp_range=(0.85, 1.15),
                 wander_amp=0.25, max_shift=4):
    """
    Random changes applied to the WAVEFORM columns of each training batch:
    small time shift, amplitude scaling, slow baseline wander, light noise.
    The timing-feature columns (if present) are left untouched.
    """
    t = np.linspace(0.0, 1.0, n_wave)
 
    def augment(xb, rng):
        xb = xb.copy()
        n = len(xb)
        w = xb[:, :n_wave]
        shifts = rng.integers(-max_shift, max_shift + 1, size=n)
        idx = np.clip(np.arange(n_wave)[None, :] + shifts[:, None], 0, n_wave - 1)
        w = np.take_along_axis(w, idx, axis=1)
        w = w * rng.uniform(*amp_range, size=(n, 1))
        freq = rng.uniform(0.5, 1.5, size=(n, 1))
        phase = rng.uniform(0, 2 * np.pi, size=(n, 1))
        amp = rng.uniform(0, wander_amp, size=(n, 1))
        w = w + amp * np.sin(2 * np.pi * freq * t[None, :] + phase)
        w = w + rng.normal(0, noise_std, size=w.shape)
        xb[:, :n_wave] = w
        return xb
 
    return augment
 
 
# ----------------------------------------------------------------------------
# operating points (vectorized, from quantiles of the scores)
# ----------------------------------------------------------------------------
def candidate_thresholds(scores, n=300):
    return np.unique(np.quantile(scores, np.linspace(0.05, 0.999, n)))
 
 
def operating_points(y, scores, thresholds):
    order = np.argsort(scores)
    s, yy = scores[order], y[order]
    pos_below = np.concatenate([[0], np.cumsum(yy)])
    total_pos = yy.sum()
    idx = np.searchsorted(s, thresholds, side="left")
    tp = total_pos - pos_below[idx]
    flagged = len(s) - idx
    return tp / max(total_pos, 1), tp / np.maximum(flagged, 1)
 
 
def best_f1_threshold(y, scores):
    ts = candidate_thresholds(scores)
    sens, prec = operating_points(y, scores, ts)
    f1 = np.where(sens + prec > 0, 2 * sens * prec / np.maximum(sens + prec, 1e-12), 0)
    return float(ts[int(np.argmax(f1))])
 
 
def threshold_for_sensitivity(y, scores, target=0.90):
    ts = candidate_thresholds(scores)
    sens, _ = operating_points(y, scores, ts)
    ok = np.where(sens >= target)[0]
    return float(ts[ok[-1]]) if len(ok) else float(ts[0])
 
 
# ----------------------------------------------------------------------------
# cross-validation machinery
# ----------------------------------------------------------------------------
def smooth(curve, w=3):
    pad = w // 2
    return np.convolve(np.pad(curve, (pad, pad), mode="edge"),
                       np.ones(w) / w, mode="valid")
 
 
def fold_masks(g, folds):
    for held in folds:
        va = np.isin(g, held)
        yield held, ~va, va
 
 
def fit_kwargs(cfg, y_tr, mode, lr, wd, batch):
    return dict(batch_size=batch, lr=lr, weight_decay=wd,
                class_weights=make_class_weights(y_tr, mode),
                augment=make_augment() if cfg["aug"] else None)
 
 
def cv_curves(X, y, g, folds, cfg, sizes, mode, seed, max_epochs, lr, wd, batch):
    """Validation AUC per epoch for every fold: array (k, max_epochs)."""
    curves = []
    for i, (_, tr, va) in enumerate(fold_masks(g, folds)):
        net = NeuralNet(sizes, seed=seed + i, dropout=cfg["dropout"])
        h = net.fit(X[tr], y[tr], epochs=max_epochs, X_val=X[va], y_val=y[va],
                    val_metric=roc_auc, verbose=False,
                    **fit_kwargs(cfg, y[tr], mode, lr, wd, batch))
        curves.append(h["val_metric"])
        print(".", end="", flush=True)
    return np.array(curves)
 
 
def oof_predictions(X, y, g, folds, cfg, sizes, mode, seed, epochs, lr, wd, batch):
    oof = np.zeros(len(y))
    for i, (_, tr, va) in enumerate(fold_masks(g, folds)):
        net = NeuralNet(sizes, seed=seed + i, dropout=cfg["dropout"])
        net.fit(X[tr], y[tr], epochs=epochs, verbose=False,
                **fit_kwargs(cfg, y[tr], mode, lr, wd, batch))
        oof[va] = net.predict_proba(X[va])
        print("+", end="", flush=True)
    return oof
 
 
# ----------------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------------
def type_aucs(classes, proba):
    """AUC of each abnormal type against normal beats (threshold-free)."""
    out = {}
    for c in "SVF":
        n_c = int((classes == c).sum())
        if n_c == 0:
            continue
        m = (classes == "N") | (classes == c)
        out[c] = roc_auc((classes[m] == c).astype(int), proba[m])
    return out
 
 
def show(name, report):
    print(f"\n{name}")
    for k, v in report.items():
        print(f"  {k:22s} {v if isinstance(v, dict) else round(v, 4)}")
 
 
def per_type_report(classes, proba, threshold):
    flagged = proba >= threshold
    print(f"  {'type':18s} {'beats':>7s}  result")
    for c in "NSVF":
        m = classes == c
        n = int(m.sum())
        if n == 0:
            continue
        if c == "N":
            print(f"  {TYPE_NAMES[c]:18s} {n:7d}  {1 - flagged[m].mean():6.1%} correctly passed")
        else:
            print(f"  {TYPE_NAMES[c]:18s} {n:7d}  {flagged[m].mean():6.1%} caught")
 
 
def worst_false_alarm_records(g, y, proba, t, top=6, min_normal=300):
    rows = []
    for r in np.unique(g):
        normal = (g == r) & (y == 0)
        if normal.sum() < min_normal:
            continue
        rows.append(((proba[normal] >= t).mean(), int(r), int(normal.sum()),
                     int(((g == r) & (y == 1)).sum())))
    rows.sort(reverse=True)
    if not rows:
        print(f"  (no records with at least {min_normal} normal beats)")
        return
    print(f"  {'record':>6s} {'normal beats':>13s} {'abnormal beats':>15s}  false alarm rate")
    for fa, r, n_norm, n_ab in rows[:top]:
        print(f"  {r:6d} {n_norm:13d} {n_ab:15d}  {fa:6.1%}")
 
 
# ----------------------------------------------------------------------------
# main experiment
# ----------------------------------------------------------------------------
def run(d, k=5, seed=0, lr=1e-3, wd=1e-4, batch=128, max_epochs=40,
        mode="sqrt", n_seeds=5, configs=CONFIGS):
    y, g, c_dev = d["y_dev"], d["g_dev"], d["c_dev"]
    yt, ct, gt = d["y_test"], d["c_test"], d["g_test"]
    print(f"\ndev  {d['X_dev'].shape[0]} beats ({y.mean():.1%} abnormal, "
          f"{len(np.unique(g))} patients)")
    print(f"test {d['X_test'].shape[0]} beats ({yt.mean():.1%} abnormal, "
          f"{len(np.unique(gt))} patients)")
    print(f"class weighting: '{mode}'   max epochs per CV run: {max_epochs}")
 
    folds = grouped_folds(y, g, k, seed)
 
    # gradient check on the real inputs, with dropout switched on
    Xchk = make_inputs(d, "dev", True)[:30]
    err = gradient_check(NeuralNet([Xchk.shape[1], 64, 32, 1], seed=seed, dropout=0.3),
                         Xchk, y[:30])
    print(f"gradient check (with dropout) max relative error: {err:.2e} "
          f"({'PASS' if err < 1e-6 else 'FAIL'})")
 
    # ---- Stage 1: ablation ------------------------------------------------
    print(f"\nSTAGE 1: ablation, {k}-fold patient-grouped CV on dev patients "
          f"(. = curve fold, + = out-of-fold)")
    results = []
    for cfg in configs:
        t0 = time.time()
        X = make_inputs(d, "dev", cfg["rr"])
        sizes = [X.shape[1], *cfg["hidden"], 1]
        print(f"  {cfg['name']:28s} ", end="", flush=True)
        curves = cv_curves(X, y, g, folds, cfg, sizes, mode, seed, max_epochs, lr, wd, batch)
        epochs = int(np.argmax(smooth(curves.mean(axis=0)))) + 1
        oof = oof_predictions(X, y, g, folds, cfg, sizes, mode, seed, epochs, lr, wd, batch)
        fold_aucs = curves[:, epochs - 1]
        results.append(dict(cfg=cfg, epochs=epochs, oof=oof, sizes=sizes,
                            auc=roc_auc(y, oof), ap=average_precision(y, oof),
                            fold_aucs=fold_aucs, types=type_aucs(c_dev, oof),
                            secs=time.time() - t0))
        print()
 
    print(f"\n  {'config':28s} {'ep':>3s} {'fold AUC (mean+/-sd, min)':>27s} "
          f"{'pooled AUC':>10s} {'AP':>6s} {'S-AUC':>6s} {'V-AUC':>6s} {'F-AUC':>6s} {'sec':>5s}")
    for r in results:
        fa, t = r["fold_aucs"], r["types"]
        print(f"  {r['cfg']['name']:28s} {r['epochs']:3d} "
              f"{fa.mean():8.3f} +/- {fa.std():.3f}, {fa.min():.3f} "
              f"{r['auc']:10.3f} {r['ap']:6.3f} "
              f"{t.get('S', float('nan')):6.3f} {t.get('V', float('nan')):6.3f} "
              f"{t.get('F', float('nan')):6.3f} {r['secs']:5.0f}")
    print("  (S/V/F-AUC = that beat type vs normal beats, out-of-fold on dev patients)")
 
    best = max(results, key=lambda r: r["ap"])
    cfg, E, sizes = best["cfg"], best["epochs"], best["sizes"]
    print(f"\nselected: '{cfg['name']}' ({E} epochs), highest pooled out-of-fold "
          f"average precision {best['ap']:.3f}")
 
    # ---- Stage 2: multi-seed out-of-fold predictions -> thresholds --------
    X = make_inputs(d, "dev", cfg["rr"])
    Xt = make_inputs(d, "test", cfg["rr"])
    print(f"\nSTAGE 2: {n_seeds} seeds of out-of-fold predictions "
          f"(seed 0 reused from stage 1)")
    oofs = [best["oof"]]
    for s in range(1, n_seeds):
        print(f"  seed {s} ", end="", flush=True)
        oofs.append(oof_predictions(X, y, g, folds, cfg, sizes, mode,
                                    seed + 1000 * s, E, lr, wd, batch))
        print()
    oof_ens = np.mean(oofs, axis=0)
    aps = [average_precision(y, o) for o in oofs]
    print(f"  out-of-fold AP per seed: {np.mean(aps):.3f} +/- {np.std(aps):.3f}; "
          f"ensemble AP {average_precision(y, oof_ens):.3f}, "
          f"AUC {roc_auc(y, oof_ens):.3f}")
    t_f1 = best_f1_threshold(y, oof_ens)
    t_sens = threshold_for_sensitivity(y, oof_ens, 0.90)
    print(f"  thresholds from ensemble out-of-fold scores: best-F1 {t_f1:.3f}, "
          f"90%-sensitivity {t_sens:.3f}")
    print("  dev patients with the most false alarms (out-of-fold, best-F1):")
    worst_false_alarm_records(g, y, oof_ens, t_f1)
 
    # ---- Stage 3: final ensemble, evaluated once on the test set ----------
    print(f"\nSTAGE 3: training {n_seeds} final models on all dev patients "
          f"({E} epochs each)")
    nets = []
    for s in range(n_seeds):
        net = NeuralNet(sizes, seed=1000 * s, dropout=cfg["dropout"])
        net.fit(X, y, epochs=E, verbose=False,
                **fit_kwargs(cfg, y, mode, lr, wd, batch))
        nets.append(net)
    probas = [n.predict_proba(Xt) for n in nets]
    proba = np.mean(probas, axis=0)
 
    print("\n=== TEST SET (DS2), evaluated once ===")
    seed_auc = [roc_auc(yt, p) for p in probas]
    seed_ap = [average_precision(yt, p) for p in probas]
    print(f"single models: AUC {np.mean(seed_auc):.4f} +/- {np.std(seed_auc):.4f}, "
          f"AP {np.mean(seed_ap):.4f} +/- {np.std(seed_ap):.4f}")
    print(f"ENSEMBLE:      AUC {roc_auc(yt, proba):.4f}, "
          f"AP {average_precision(yt, proba):.4f}   (chance AP = {yt.mean():.3f})")
    print("per-type AUC against normal beats:")
    for c, a in type_aucs(ct, proba).items():
        print(f"  {TYPE_NAMES[c]:18s} {int((ct == c).sum()):6d} beats   AUC {a:.3f}")
    for label, t in [("best-F1", t_f1), ("90%-sensitivity target", t_sens)]:
        show(f"Ensemble, {label} threshold {t:.3f}",
             classification_report(yt, proba, threshold=t))
        print("  per beat type:")
        per_type_report(ct, proba, t)
    print("\ntest patients with the most false alarms (best-F1 threshold):")
    worst_false_alarm_records(gt, yt, proba, t_f1)
 
    print("\nLogistic regression baselines (class_weight='balanced'):")
    for name, use_rr in [("wave only", False), ("wave + timing", True)]:
        Xb, Xbt = make_inputs(d, "dev", use_rr), make_inputs(d, "test", use_rr)
        lrm = LogisticRegression(max_iter=3000, class_weight="balanced").fit(Xb, y)
        bp = lrm.predict_proba(Xbt)[:, 1]
        print(f"  {name:14s} AUC {roc_auc(yt, bp):.4f}   AP {average_precision(yt, bp):.4f}")
 
    meta = dict(
        config=cfg["name"], hidden=list(cfg["hidden"]), dropout=cfg["dropout"],
        augmentation=cfg["aug"], epochs=E, class_weighting=mode, sizes=sizes,
        thresholds={"best_f1": t_f1, "sens90": t_sens},
        feature_spec=dict(n_wave=N_WAVE, pre=90, post=144, fs=FS,
                          standardization="per-record z-score of the MLII lead",
                          timing_features=RR_FEATURE_NAMES if cfg["rr"] else []),
        dev_out_of_fold=dict(auc=float(roc_auc(y, oof_ens)),
                             ap=float(average_precision(y, oof_ens))),
    )
    return nets, meta
 
 
if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/mitdb"
    d = load_dataset(data_dir)
    nets, meta = run(d)
    save_ensemble("ecg_model.npz", nets, meta)
    print("\nSaved ecg_model.npz")
