"""Evidentiary-discipline helpers: Wilson CIs, TOST equivalence,
Holm-Bonferroni / BH-FDR multiplicity control, and failure-budget sizing.

Ported unchanged from the validated research statistics module.
"""
import math


def wilson_ci(f, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = f / n
    d = 1 + z*z/n
    c = p + z*z/(2*n)
    h = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return ((c - h)/d, (c + h)/d)


def _two_prop(f1, n1, f2, n2):
    p1, p2 = f1/n1, f2/n2
    se = math.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2) or 1e-12
    return p1, p2, se


def tost_equivalent(f1, n1, f2, n2, margin_rel=0.10, alpha=0.05):
    """Two one-sided tests: equivalent iff the difference is significantly within +-margin
    (margin relative to the mean rate). Returns bool."""
    from statistics import NormalDist
    p1, p2, se = _two_prop(f1, n1, f2, n2)
    margin = margin_rel * ((p1 + p2) / 2 or 1e-12)
    nd = NormalDist()
    z_lower = ((p1 - p2) - (-margin)) / se   # H0: diff <= -margin  (reject if z_lower large +)
    z_upper = ((p1 - p2) - (margin)) / se    # H0: diff >= +margin  (reject if z_upper large -)
    return bool((1 - nd.cdf(z_lower)) < alpha and nd.cdf(z_upper) < alpha)


def holm_bonferroni(pvals, alpha=0.05):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    rej = [False] * len(pvals)
    for rank, i in enumerate(order):
        if pvals[i] <= alpha / (len(pvals) - rank):
            rej[i] = True
        else:
            break
    return rej


def bh_fdr(pvals, q=0.05):
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    rej = [False] * m
    kmax = 0
    for rank, i in enumerate(order, start=1):
        if pvals[i] <= q * rank / m:
            kmax = rank
    for rank, i in enumerate(order, start=1):
        if rank <= kmax:
            rej[i] = True
    return rej


def failures_to_shots(target_failures, ler):
    return int(math.ceil(target_failures / max(ler, 1e-12)))
