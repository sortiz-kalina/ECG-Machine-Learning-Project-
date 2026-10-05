"""
A feedforward neural network built from scratch with NumPy.

Binary classifier: Dense -> ReLU -> ... -> Dense -> Sigmoid, trained with
binary cross-entropy and the Adam optimizer. No autograd: every gradient is
derived by hand and verified with numerical gradient checking.
"""
import numpy as np


# ----------------------------------------------------------------------------
# Layers
# ----------------------------------------------------------------------------
class Dense:
    """Fully connected layer: out = x @ W + b."""

    def __init__(self, n_in, n_out, rng, init="he"):
        scale = np.sqrt(2.0 / n_in) if init == "he" else np.sqrt(1.0 / n_in)
        self.W = rng.standard_normal((n_in, n_out)) * scale
        self.b = np.zeros(n_out)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._x = None

    def forward(self, x):
        self._x = x  # cache input for the backward pass
        return x @ self.W + self.b

    def backward(self, grad_out):
        # grad_out has shape (batch, n_out)
        self.dW = self._x.T @ grad_out
        self.db = grad_out.sum(axis=0)
        return grad_out @ self.W.T  # gradient w.r.t. this layer's input

    def params(self):
        return [(self.W, self.dW), (self.b, self.db)]


class ReLU:
    def __init__(self):
        self._mask = None
        self.freeze = False

    def forward(self, x):
        if self.freeze and self._mask is not None and self._mask.shape == x.shape:
            return x * self._mask
        self._mask = x > 0
        return x * self._mask

    def backward(self, grad_out):
        return grad_out if self._mask is None else grad_out * self._mask

    def params(self):
        return []


class Dropout:
    """
    Inverted dropout: during training, zero each unit with probability p and
    scale the survivors by 1/(1-p), so no rescaling is needed at test time.
    In eval mode (training=False) it is the identity.
    """

    def __init__(self, p, rng):
        self.p = p
        self.rng = rng
        self._mask = None
        self.freeze = False  # reuse one mask (used only by gradient_check)

    def forward(self, x, training=False):
        if not training or self.p <= 0.0:
            self._mask = None
            return x
        if not (self.freeze and self._mask is not None
                and self._mask.shape == x.shape):
            self._mask = self.rng.random(x.shape) >= self.p
        return x * self._mask / (1.0 - self.p)

    def backward(self, grad_out):
        if self._mask is None:
            return grad_out
        return grad_out * self._mask / (1.0 - self.p)

    def params(self):
        return []


# ----------------------------------------------------------------------------
# Loss: sigmoid + binary cross-entropy, computed together for stability
# ----------------------------------------------------------------------------
def sigmoid(z):
    # numerically stable: never exponentiates a large positive number
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def bce_with_logits(logits, y, class_weights=None):
    """
    Mean binary cross-entropy from raw logits (stable log-sum-exp form).
    Returns (loss, dLoss/dLogits).

    class_weights = (w_neg, w_pos) optionally reweights the two classes, which
    matters on imbalanced medical data.
    """
    y = y.reshape(-1, 1)
    z = logits
    w = np.ones_like(y, dtype=float)
    if class_weights is not None:
        w = np.where(y == 1, class_weights[1], class_weights[0])
    # loss_i = max(z,0) - z*y + log(1 + exp(-|z|))
    per_sample = np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))
    loss = np.mean(w * per_sample)
    # derivative of sigmoid+BCE w.r.t. the logit simplifies to (p - y)
    grad = w * (sigmoid(z) - y) / len(y)
    return loss, grad


# ----------------------------------------------------------------------------
# Optimizer
# ----------------------------------------------------------------------------
class Adam:
    def __init__(self, layers, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8,
                 weight_decay=0.0):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.wd = weight_decay
        self.t = 0
        self.slots = []  # (param, m, v) triples, grads are read off the layer
        for layer in layers:
            for p, _ in layer.params():
                self.slots.append((p, np.zeros_like(p), np.zeros_like(p)))
        self.layers = layers

    def step(self):
        self.t += 1
        i = 0
        for layer in self.layers:
            for p, g in layer.params():
                _, m, v = self.slots[i]
                if self.wd and p.ndim == 2:  # L2 on weights only, not biases
                    g = g + self.wd * p
                m *= self.b1
                m += (1 - self.b1) * g
                v *= self.b2
                v += (1 - self.b2) * g * g
                m_hat = m / (1 - self.b1 ** self.t)  # bias correction
                v_hat = v / (1 - self.b2 ** self.t)
                p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
                i += 1


# ----------------------------------------------------------------------------
# Network
# ----------------------------------------------------------------------------
class NeuralNet:
    def __init__(self, sizes, seed=0, dropout=0.0):
        """sizes = [n_features, hidden1, hidden2, ..., 1]; dropout applies
        after every hidden ReLU."""
        self.rng = np.random.default_rng(seed)
        self.layers = []
        for i in range(len(sizes) - 1):
            last = i == len(sizes) - 2
            self.layers.append(Dense(sizes[i], sizes[i + 1], self.rng,
                                     init="xavier" if last else "he"))
            if not last:
                self.layers.append(ReLU())
                if dropout > 0:
                    self.layers.append(Dropout(dropout, self.rng))

    def forward(self, x, training=False):
        for layer in self.layers:
            if isinstance(layer, Dropout):
                x = layer.forward(x, training)
            else:
                x = layer.forward(x)
        return x  # logits

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def predict_proba(self, x):
        return sigmoid(self.forward(x)).ravel()

    def get_weights(self):
        return [(l.W.copy(), l.b.copy()) for l in self.layers if isinstance(l, Dense)]

    def set_weights(self, weights):
        dense = [l for l in self.layers if isinstance(l, Dense)]
        for layer, (W, b) in zip(dense, weights):
            layer.W[...] = W  # in-place, so the optimizer's references stay valid
            layer.b[...] = b

    def fit(self, X, y, epochs=200, batch_size=32, lr=1e-3, weight_decay=0.0,
            class_weights=None, X_val=None, y_val=None, patience=None,
            val_metric=None, augment=None, verbose=True):
        """
        If patience is set (needs X_val/y_val), training stops once the
        validation score hasn't improved for `patience` epochs, and the best
        weights seen are restored.

        The score is -validation loss by default. Pass val_metric, a function
        (y_true, proba) -> float where HIGHER is better (e.g. roc_auc), to
        monitor that instead.

        augment, if given, is a function (X_batch, rng) -> X_batch applied to
        every training minibatch (never to validation data).
        """
        opt = Adam(self.layers, lr=lr, weight_decay=weight_decay)
        n = len(X)
        history = {"loss": [], "val_loss": [], "val_metric": [],
                   "best_epoch": None}
        best_score, best_weights, bad_epochs = -np.inf, None, 0
        for epoch in range(1, epochs + 1):
            order = self.rng.permutation(n)
            epoch_loss, batches = 0.0, 0
            for start in range(0, n, batch_size):
                idx = order[start:start + batch_size]
                xb = X[idx]
                if augment is not None:
                    xb = augment(xb, self.rng)
                logits = self.forward(xb, training=True)
                loss, grad = bce_with_logits(logits, y[idx], class_weights)
                self.backward(grad)
                opt.step()
                epoch_loss += loss
                batches += 1
            history["loss"].append(epoch_loss / batches)
            if X_val is not None:
                vl, _ = bce_with_logits(self.forward(X_val), y_val, class_weights)
                history["val_loss"].append(vl)
                if val_metric is not None:
                    score = val_metric(y_val, self.predict_proba(X_val))
                    history["val_metric"].append(score)
                else:
                    score = -vl
                if score > best_score + 1e-6:
                    best_score, best_weights, bad_epochs = score, self.get_weights(), 0
                    history["best_epoch"] = epoch
                else:
                    bad_epochs += 1
            if verbose and (epoch == 1 or epoch % max(1, epochs // 10) == 0):
                msg = f"epoch {epoch:4d}  loss {history['loss'][-1]:.4f}"
                if X_val is not None:
                    msg += f"  val_loss {history['val_loss'][-1]:.4f}"
                print(msg)
            if patience is not None and bad_epochs >= patience:
                if verbose:
                    print(f"early stop at epoch {epoch} "
                          f"(best epoch {history['best_epoch']})")
                break
        if patience is not None and best_weights is not None:
            self.set_weights(best_weights)
        return history

    # -- weight (de)serialization, so a FastAPI service can load the model --
    def save(self, path, mean=None, std=None):
        arrays = {}
        k = 0
        for layer in self.layers:
            if isinstance(layer, Dense):
                arrays[f"W{k}"], arrays[f"b{k}"] = layer.W, layer.b
                k += 1
        if mean is not None:
            arrays["mean"], arrays["std"] = mean, std
        np.savez(path, **arrays)


# ----------------------------------------------------------------------------
# Gradient checking: compare backprop against finite differences
# ----------------------------------------------------------------------------
def gradient_check(net, X, y, eps=1e-5, n_checks=10, class_weights=None, seed=0):
    """
    Returns the max relative error between analytic and numerical gradients
    over a random sample of parameters. ~1e-7 or smaller means backprop is right.
    """
    rng = np.random.default_rng(seed)
    relus = [l for l in net.layers if isinstance(l, ReLU)]
    dropouts = [l for l in net.layers if isinstance(l, Dropout)]
    for l in relus + dropouts:
        l.freeze = True  # one fixed activation/mask pattern, so the loss is deterministic
    logits = net.forward(X, training=True)
    _, grad = bce_with_logits(logits, y, class_weights)
    net.backward(grad)

    worst = 0.0
    for layer in net.layers:
        for p, g in layer.params():
            flat_p, flat_g = p.reshape(-1), g.reshape(-1)  # views into p, g
            for i in rng.choice(flat_p.size, size=min(n_checks, flat_p.size),
                                replace=False):
                orig = flat_p[i]
                flat_p[i] = orig + eps
                l_plus, _ = bce_with_logits(net.forward(X, training=True), y, class_weights)
                flat_p[i] = orig - eps
                l_minus, _ = bce_with_logits(net.forward(X, training=True), y, class_weights)
                flat_p[i] = orig
                numeric = (l_plus - l_minus) / (2 * eps)
                analytic = flat_g[i]
                # The finite-difference probe is only meaningful when the gradient is
                # not tiny: in saturated ReLU / dropout cases, both sides can be
                # effectively zero and the relative error becomes numerically unstable.
                # With an eps of 1e-5, anything below about 1e-4 is below the
                # practical resolution of the check and should be skipped.
                if max(abs(numeric), abs(analytic)) < 1e-4:
                    continue
                denom = max(abs(numeric) + abs(analytic), 1e-12)
                worst = max(worst, abs(numeric - analytic) / denom)
    for l in relus + dropouts:
        l.freeze = False
        l._mask = None
    return worst


# ----------------------------------------------------------------------------
# Metrics (implemented by hand, no sklearn)
# ----------------------------------------------------------------------------
def roc_auc(y_true, scores):
    """AUC via the rank-sum (Mann-Whitney U) formulation, with tie handling."""
    y_true = np.asarray(y_true)
    order = np.argsort(scores, kind="mergesort")
    s = np.asarray(scores)[order]
    ranks = np.empty(len(s))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1  # average rank for ties
        i = j + 1
    r = np.empty(len(s))
    r[order] = ranks
    n_pos = int((y_true == 1).sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5  # undefined with one class present; treat as chance
    return (r[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def average_precision(y_true, scores):
    """
    Area under the precision-recall curve (step-wise form): the mean of the
    precision values at the rank of each true positive. More informative than
    ROC-AUC when positives are rare (ties are broken arbitrarily).
    """
    y_true = np.asarray(y_true).ravel()
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return 0.0
    order = np.argsort(-np.asarray(scores), kind="mergesort")
    y_sorted = y_true[order]
    precision_at_k = np.cumsum(y_sorted) / np.arange(1, len(y_sorted) + 1)
    return float((precision_at_k * y_sorted).sum() / n_pos)


def classification_report(y_true, proba, threshold=0.5):
    y_true = np.asarray(y_true).ravel()
    pred = (proba >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    div = lambda a, b: a / b if b else 0.0
    return {
        "accuracy": div(tp + tn, len(y_true)),
        "sensitivity (recall)": div(tp, tp + fn),
        "specificity": div(tn, tn + fp),
        "precision": div(tp, tp + fp),
        "roc_auc": roc_auc(y_true, proba),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }