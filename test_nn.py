"""
Tests for the from-scratch network. Run:  python test_nn.py

The key idea: a gradient check is only trustworthy if it (a) passes on correct
code in the awkward cases (tiny gradients, confident predictions, dropout) and
(b) FAILS when backprop is deliberately broken.
"""
import numpy as np
from nn import (NeuralNet, Dense, ReLU, Dropout, gradient_check, bce_with_logits,
                roc_auc, average_precision)

TOL = 1e-6


def make_batch(seed, balanced=True):
    r = np.random.default_rng(seed)
    X = np.hstack([r.normal(0, 1, (30, 234)), r.normal(.85, .05, (30, 4))])
    y = np.zeros(30, dtype=int)
    if balanced:
        y[:15] = 1
    return X, y


def make_net(seed, bias=0.0, dropout=0.3):
    net = NeuralNet([238, 64, 32, 1], seed=seed, dropout=dropout)
    [l for l in net.layers if isinstance(l, Dense)][-1].b[...] = bias
    return net


# --- (a) correct code passes, even in the cases that used to look like failures --
def test_correct_backprop_passes():
    worst = 0.0
    for bias in (0.0, -8.0, -14.0):          # increasingly confident "normal"
        for balanced in (False, True):
            for seed in range(12):
                X, y = make_batch(seed, balanced)
                worst = max(worst, gradient_check(make_net(seed, bias), X, y))
    print(f"  correct code, 72 nets: worst error {worst:.1e}")
    assert worst < TOL, worst


# --- (b) planted bugs must be caught ---------------------------------------------
def _expect_failure(name, patch):
    caught = []
    for seed in range(6):
        X, y = make_batch(seed, True)
        net = make_net(seed, 0.0)
        patch(net)
        caught.append(gradient_check(net, X, y))
    print(f"  planted bug '{name}': min error over 6 nets {min(caught):.1e}")
    assert min(caught) > 1e-3, f"bug '{name}' was NOT caught ({min(caught):.1e})"


def test_planted_bugs_are_caught():
    def dense_scale(net):  # weight gradient 2% too large
        for l in net.layers:
            if isinstance(l, Dense):
                orig = l.backward
                def bw(g, l=l, orig=orig):
                    out = orig(g); l.dW = l.dW * 1.02; return out
                l.backward = bw

    def dense_bias_missing(net):  # forgets the bias gradient
        for l in net.layers:
            if isinstance(l, Dense):
                orig = l.backward
                def bw(g, l=l, orig=orig):
                    out = orig(g); l.db = l.db * 0.0; return out
                l.backward = bw

    def relu_ignores_mask(net):  # passes gradient through dead units
        for l in net.layers:
            if isinstance(l, ReLU):
                l.backward = lambda g: g

    def dropout_ignores_mask(net):  # forgets to apply the dropout mask
        for l in net.layers:
            if isinstance(l, Dropout):
                l.backward = lambda g: g

    def wrong_input_gradient(net):  # gradient flowing back through layer 2 is 2% off
        first = [l for l in net.layers if isinstance(l, Dense)][1]
        orig = first.backward
        def bw(g, orig=orig, first=first):
            out = orig(g); return out * 0.98
        first.backward = bw

    _expect_failure("weight gradient 2% too large", dense_scale)
    _expect_failure("bias gradient missing", dense_bias_missing)
    _expect_failure("ReLU ignores its mask", relu_ignores_mask)
    _expect_failure("dropout ignores its mask", dropout_ignores_mask)
    _expect_failure("input gradient 2% off", wrong_input_gradient)


# --- layer behaviour ---------------------------------------------------------------
def test_dropout_modes():
    rng = np.random.default_rng(0)
    d = Dropout(0.3, rng)
    x = np.ones((100000, 1))
    assert np.array_equal(d.forward(x, training=False), x)        # identity in eval
    assert abs(d.forward(x, training=True).mean() - 1.0) < 0.01   # mean preserved
    net = NeuralNet([10, 8, 1], seed=0, dropout=0.5)
    X = rng.normal(size=(5, 10))
    assert np.array_equal(net.forward(X), net.forward(X))         # eval deterministic


def test_metrics_match_known_values():
    y = np.array([0, 0, 1, 1]); s = np.array([0.1, 0.4, 0.35, 0.8])
    assert abs(roc_auc(y, s) - 0.75) < 1e-12
    # ranked by score: pos(.8), neg(.4), pos(.35), neg(.1) -> precision at the two
    # positives is 1/1 and 2/3, so AP = (1 + 2/3) / 2
    assert abs(average_precision(y, s) - (1 + 2 / 3) / 2) < 1e-12
    assert roc_auc(np.zeros(5, dtype=int), np.arange(5.0)) == 0.5  # single class guard


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            print(name)
            fn()
    print("ALL TESTS PASSED")
