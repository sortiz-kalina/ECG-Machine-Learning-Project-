"""
Inference-only loader for the trained ECG ensemble. Depends on NumPy alone, so
a FastAPI service can import this file without any of the training code.

Input rows are what training used: 234 waveform samples (90 before / 144 after
the R peak, 360 Hz, standardized with the record's own mean/std) followed, if
the model uses them, by 4 timing features:
    [rr_prev_s, rr_next_s, log_rr_prev_ratio, log_rr_next_ratio]
meta["feature_spec"] records exactly what the model expects.
"""
import json
import numpy as np


def _sigmoid(z):
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def save_ensemble(path, nets, meta):
    """nets: list of trained NeuralNet; meta: JSON-serializable dict."""
    arrays = {}
    for m, net in enumerate(nets):
        for k, (W, b) in enumerate(net.get_weights()):
            arrays[f"m{m}_W{k}"] = W
            arrays[f"m{m}_b{k}"] = b
    meta = dict(meta, n_models=len(nets))
    arrays["meta"] = np.array(json.dumps(meta))
    np.savez(path, **arrays)


class EnsembleModel:
    def __init__(self, members, meta):
        self.members = members  # list of [(W, b), ...] per model
        self.meta = meta

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data["meta"]))
        members = []
        for m in range(meta["n_models"]):
            layers, k = [], 0
            while f"m{m}_W{k}" in data.files:
                layers.append((data[f"m{m}_W{k}"], data[f"m{m}_b{k}"]))
                k += 1
            members.append(layers)
        return cls(members, meta)

    @staticmethod
    def _forward(layers, X):
        h = X
        for i, (W, b) in enumerate(layers):
            h = h @ W + b
            if i < len(layers) - 1:
                h = np.maximum(h, 0.0)  # ReLU; dropout is inactive at inference
        return h

    def predict_proba(self, X):
        """Probability that each beat is abnormal (mean over ensemble members)."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        expected = self.members[0][0][0].shape[0]
        if X.shape[1] != expected:
            raise ValueError(f"expected {expected} features per beat, got {X.shape[1]}")
        probs = [_sigmoid(self._forward(layers, X)).ravel() for layers in self.members]
        return np.mean(probs, axis=0)

    def predict(self, X, operating_point="best_f1"):
        """Binary decision at one of the stored operating points."""
        t = self.meta["thresholds"][operating_point]
        return (self.predict_proba(X) >= t).astype(int)