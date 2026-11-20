#!/usr/bin/env python3
"""
failure_diagnostics.py -- per-case failure diagnostics for the expert-tree method.

Computes, on a set of evaluation cases (use VALIDATION or TEST cases that were
NOT used to build the flagged training set):

  1. recall@k of the anchor's candidate regions  (is the lesion findable?)
       failure type per case:  "delineation" (rank-1 candidate covers the GT but Dice is low),
                               "wrong-choice" (GT covered only by a lower-ranked candidate),
                               "missed"       (no candidate covers the GT).
  2. gradient alignment with the flagged training failures (Prop. C1 in routing_theory.tex):
       rho_max   = max_i cos(g_x, g_i)
       rho_G     = cos(g_x, G),  G = sum_i w_i g_i
       A         = <g_x, G>                   (first-order gain per unit step; > 0 needed)
       gain_x_M  = A^2 / (2 ||G||^2)          (guaranteed gain at best step = gain_x_M / M)
       coverage  = sum_i w_i ||g_i|| cos(g_x, g_i)
     with an optional control: the same statistics against random non-flagged cases.
  3. AUROC (with bootstrap CI) of each flag feature for "true failure" (Dice < delta),
     TPR / FPR / precision at a threshold tau, and, if shape features are named, the
     detection ceiling nu and the within-V AUROCs (Prop. C4).

Two layers:
  * pure-numpy functions (this file) + CLI reading an .npz file
  * an optional PyTorch helper `extract_grads` to produce the gradient matrices.

NPZ format (all optional except probs and gts):
    probs         (N,H,W) float   anchor probability maps for evaluation cases
    gts           (N,H,W) {0,1}   ground-truth masks (same size as probs; resize first)
    preds         (N,H,W) {0,1}   binary predictions actually used (default: probs > 0.5)
    features      (N,F)   float   flag scores Psi_j (higher = riskier)
    feature_names (F,)    str
    grads_eval    (N,P)   float   per-case gradients of the evaluation cases
    grads_flag    (M,P)   float   per-case gradients of the flagged training failures
    weights_flag  (M,)    float   weights w_i (default 1)
    grads_ctrl    (K,P)   float   gradients of random non-flagged training cases (control)

Usage:
    python failure_diagnostics.py --npz cases.npz --out-dir out/ --delta 0.5 \
        --tau convex=0.3 --tau shadow=0.25 --shape-features convex genus
    python failure_diagnostics.py --selftest
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
from scipy import ndimage
from scipy.stats import rankdata


# ----------------------------------------------------------------------------
# basic metrics
# ----------------------------------------------------------------------------
def dice(a, b, eps=1e-8):
    a = np.asarray(a).astype(bool)
    b = np.asarray(b).astype(bool)
    return float(2.0 * (a & b).sum() / (a.sum() + b.sum() + eps))


# ----------------------------------------------------------------------------
# 1. candidates and recall@k
# ----------------------------------------------------------------------------
def gt_rank(prob, gt, cand_thr=0.2, score="max", hit_frac=0.3):
    """1-based rank of the first candidate region that covers >= hit_frac of the GT area.

    Candidates are connected components of (prob > cand_thr), ranked by `score`
    ('max', 'sum' or 'mean' of the probability inside the component).
    Returns np.inf if there is no such candidate (lesion invisible at cand_thr).
    """
    gt = np.asarray(gt).astype(bool)
    if gt.sum() == 0:
        return np.inf
    lab, n = ndimage.label(prob > cand_thr)
    if n == 0:
        return np.inf
    idx = np.arange(1, n + 1)
    fn = {"max": ndimage.maximum, "sum": ndimage.sum, "mean": ndimage.mean}[score]
    scores = np.asarray(fn(prob, lab, idx), dtype=float)
    cover = np.asarray(ndimage.sum(gt.astype(float), lab, idx)) / gt.sum()
    for r, j in enumerate(np.argsort(-scores, kind="stable"), start=1):
        if cover[j] >= hit_frac:
            return float(r)
    return np.inf


def recall_at_k(ranks, ks=(1, 2, 3, 5)):
    ranks = np.asarray(ranks, dtype=float)
    return {int(k): float(np.mean(ranks <= k)) if len(ranks) else float("nan") for k in ks}


def failure_type(rank, is_failure):
    if not is_failure:
        return "ok"
    if np.isinf(rank):
        return "missed"
    return "delineation" if rank == 1 else "wrong-choice"


# ----------------------------------------------------------------------------
# 2. gradient alignment (Prop. C1)
# ----------------------------------------------------------------------------
def alignment(g_eval, g_flag, w=None):
    """Alignment of evaluation-case gradients with flagged-training-failure gradients.

    g_eval: (m,P), g_flag: (n,P), w: (n,) non-negative weights.
    Evaluate on cases NOT in the flagged set (otherwise the self-term dominates).
    """
    g_eval = np.asarray(g_eval, float)
    g_flag = np.asarray(g_flag, float)
    w = np.ones(len(g_flag)) if w is None else np.asarray(w, float)
    ne = np.linalg.norm(g_eval, axis=1) + 1e-12
    nf = np.linalg.norm(g_flag, axis=1) + 1e-12
    cos = (g_eval / ne[:, None]) @ (g_flag / nf[:, None]).T          # (m,n)
    G = (w[:, None] * g_flag).sum(0)
    nG = np.linalg.norm(G) + 1e-12
    A = g_eval @ G
    return dict(
        rho_max=cos.max(1),
        nn_index=cos.argmax(1),
        rho_G=A / (ne * nG),
        A=A,
        gain_x_M=A ** 2 / (2.0 * nG ** 2) * (A > 0),   # guaranteed gain = this / M, only if A>0
        coverage=(cos * (w * nf)[None, :]).sum(1),
    )


# ----------------------------------------------------------------------------
# 3. flag quality (AUROC, threshold stats, detection ceiling)
# ----------------------------------------------------------------------------
def auroc(score, label):
    """Mann-Whitney AUROC with mid-ranks. Higher score = more likely failure."""
    s = np.asarray(score, float)
    y = np.asarray(label).astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def auroc_ci(score, label, B=1000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    s = np.asarray(score, float)
    y = np.asarray(label).astype(bool)
    pos, neg = np.where(y)[0], np.where(~y)[0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan"), float("nan")
    vals = []
    for _ in range(B):
        ii = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        vals.append(auroc(s[ii], y[ii]))
    return tuple(float(v) for v in np.quantile(vals, [alpha / 2, 1 - alpha / 2]))


def flag_stats(score, label, tau):
    s = np.asarray(score, float)
    y = np.asarray(label).astype(bool)
    f = s > tau
    tp, fp = int((f & y).sum()), int((f & ~y).sum())
    fn, tn = int((~f & y).sum()), int((~f & ~y).sum())
    return dict(tau=float(tau), tpr=tp / max(tp + fn, 1), fpr=fp / max(fp + tn, 1),
                precision=tp / max(tp + fp, 1), n_flagged=int(f.sum()))


def youden_tau(score, label):
    """Threshold maximizing TPR-FPR on THIS data (optimistic; use validation data)."""
    s = np.asarray(score, float)
    best, bt = -np.inf, float(np.min(s)) - 1.0
    for t in np.concatenate([[np.min(s) - 1.0], np.unique(s)]):
        st = flag_stats(s, label, t)
        if st["tpr"] - st["fpr"] > best:
            best, bt = st["tpr"] - st["fpr"], float(t)
    return bt


# ----------------------------------------------------------------------------
# optional PyTorch helper for per-case gradients (needs torch; not used by the CLI selftest)
# ----------------------------------------------------------------------------
def extract_grads(model, loss_fn, loader, device="cuda", param_filter=None, last_n=2,
                  proj_dim=None, seed=0):
    """Per-case flattened gradients of the loss w.r.t. a subset of parameters.

    loader yields (x, y) with batch size 1 (or any size; processed case by case).
    param_filter: callable(name)->bool selecting parameters (e.g. lambda n: 'decoder' in n);
                  if None, the last `last_n` trainable parameter tensors are used.
    proj_dim: optional Gaussian random projection to this dimension (preserves inner
              products in expectation) to save memory. Use the SAME seed for all calls
              so flagged / evaluation / control gradients live in the same space.
    loss_fn(logits, y) must return a scalar (use the same loss as training).
    Returns a float32 numpy array (num_cases, P).
    """
    import torch
    model.eval()
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    params = [p for n, p in named if param_filter(n)] if param_filter else [p for _, p in named][-last_n:]
    P = sum(p.numel() for p in params)
    proj = None
    if proj_dim is not None:
        gen = torch.Generator(device="cpu").manual_seed(seed)
        proj = (torch.randn(P, proj_dim, generator=gen) / proj_dim ** 0.5).to(device)
    out = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        for k in range(x.shape[0]):
            model.zero_grad(set_to_none=True)
            loss = loss_fn(model(x[k:k + 1]), y[k:k + 1])
            grads = torch.autograd.grad(loss, params)
            flat = torch.cat([g.reshape(-1) for g in grads])
            if proj is not None:
                flat = flat @ proj
            out.append(flat.detach().float().cpu().numpy())
    return np.stack(out)


# ----------------------------------------------------------------------------
# main analysis
# ----------------------------------------------------------------------------
def analyse(d, delta=0.5, ks=(1, 2, 3, 5), cand_thr=0.2, score="max", hit_frac=0.3,
            taus=None, shape_features=None, gamma_over_delta=None, seed=0):
    taus = taus or {}
    probs, gts = np.asarray(d["probs"]), np.asarray(d["gts"])
    preds = np.asarray(d["preds"]) if "preds" in d else (probs > 0.5)
    N = len(probs)
    dsc = np.array([dice(preds[i], gts[i]) for i in range(N)])
    is_fail = dsc < delta
    ranks = np.array([gt_rank(probs[i], gts[i], cand_thr, score, hit_frac) for i in range(N)])
    ftype = [failure_type(ranks[i], is_fail[i]) for i in range(N)]
    gt_maxp = np.array([probs[i][gts[i].astype(bool)].max() if gts[i].any() else np.nan for i in range(N)])

    rows = {"case": np.arange(N), "dice": dsc, "is_failure": is_fail.astype(int),
            "gt_rank": ranks, "failure_type": ftype, "gt_max_prob": gt_maxp}
    summary = {"n_cases": int(N), "n_failures": int(is_fail.sum()), "delta": delta}

    # 1. recall@k
    summary["failure_types"] = {t: int(sum(1 for x in ftype if x == t))
                                for t in ("delineation", "wrong-choice", "missed")}
    summary["recall_at_k_on_failures"] = recall_at_k(ranks[is_fail], ks)
    summary["recall_at_k_on_all"] = recall_at_k(ranks, ks)

    # 2. gradient alignment
    if "grads_eval" in d and "grads_flag" in d:
        al = alignment(d["grads_eval"], d["grads_flag"], d.get("weights_flag"))
        for k, v in al.items():
            rows[k] = v
        summary["alignment_on_failures"] = {
            k: float(np.mean(al[k][is_fail])) if is_fail.any() else float("nan")
            for k in ("rho_max", "rho_G", "A", "gain_x_M", "coverage")}
        summary["frac_failures_with_A_positive"] = float(np.mean(al["A"][is_fail] > 0)) if is_fail.any() else float("nan")
        summary["frac_failures_with_rho_max_positive"] = float(np.mean(al["rho_max"][is_fail] > 0)) if is_fail.any() else float("nan")
        if "grads_ctrl" in d:
            # control: random non-flagged cases as the 'training set' (same size as flagged set)
            rng = np.random.default_rng(seed)
            gc = np.asarray(d["grads_ctrl"])
            sel = rng.choice(len(gc), size=min(len(gc), len(d["grads_flag"])), replace=False)
            ac = alignment(d["grads_eval"], gc[sel])
            summary["control_alignment_on_failures"] = {
                k: float(np.mean(ac[k][is_fail])) if is_fail.any() else float("nan")
                for k in ("rho_max", "rho_G", "A")}

    # 3. flag quality
    if "features" in d:
        F = np.asarray(d["features"], float)
        names = [str(x) for x in d.get("feature_names", [f"f{j}" for j in range(F.shape[1])])]
        feat = {}
        for j, nm in enumerate(names):
            lo, hi = auroc_ci(F[:, j], is_fail, seed=seed)
            tau = taus.get(nm)
            used_youden = tau is None
            if used_youden:
                tau = youden_tau(F[:, j], is_fail)
            st = flag_stats(F[:, j], is_fail, tau)
            st["tau_source"] = "youden (in-sample, optimistic)" if used_youden else "given"
            e = dict(auroc=auroc(F[:, j], is_fail), auroc_ci95=[lo, hi], **st)
            if gamma_over_delta is not None and st["precision"] < 1:
                odds = st["precision"] / (1 - st["precision"])
                e["breakeven_ok"] = bool(odds > gamma_over_delta)
            # which failure types does this flag catch?
            fl = F[:, j] > tau
            e["tpr_by_failure_type"] = {t: (float(fl[np.array(ftype) == t].mean()) if any(x == t for x in ftype) else None)
                                        for t in ("delineation", "wrong-choice", "missed")}
            feat[nm] = e
            rows["flag_" + nm] = fl.astype(int)
        summary["features"] = feat

        # detection ceiling (Prop. C4)
        if shape_features:
            idx = [names.index(n) for n in shape_features]
            miss = [n for n in shape_features if n not in taus]
            if miss:
                raise ValueError(f"--shape-features need --tau for: {miss}")
            flagged_any = np.any(np.stack([F[:, j] > taus[names[j]] for j in idx], 1), 1)
            inV = ~flagged_any
            ceil = {"nu_TPR_of_shape_flags": float(flagged_any[is_fail].mean()) if is_fail.any() else float("nan"),
                    "FPR_of_shape_flags": float(flagged_any[~is_fail].mean()) if (~is_fail).any() else float("nan"),
                    "n_failures_inside_V": int((is_fail & inV).sum()),
                    "n_nonfailures_inside_V": int((~is_fail & inV).sum()),
                    "auroc_within_V": {}}
            for j, nm in enumerate(names):
                ceil["auroc_within_V"][nm] = auroc(F[inV, j], is_fail[inV]) if inV.sum() else float("nan")
            summary["detection_ceiling"] = ceil
    return rows, summary


def _write_outputs(rows, summary, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    keys = list(rows.keys())
    with open(os.path.join(out_dir, "per_case.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for i in range(len(rows["case"])):
            w.writerow([f"{rows[k][i]:.6g}" if isinstance(rows[k][i], (float, np.floating)) else rows[k][i] for k in keys])
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=lambda o: None if (isinstance(o, float) and np.isnan(o)) else float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    print(json.dumps(summary, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o)))
    print(f"\nwrote {out_dir}/per_case.csv and {out_dir}/summary.json")


# ----------------------------------------------------------------------------
# self-test on synthetic data with known answers
# ----------------------------------------------------------------------------
def _blob(shape, c, r):
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return ((yy - c[0]) ** 2 + (xx - c[1]) ** 2) <= r * r


def _synthetic(seed=0, n=240, hw=64, p=300, n_flag=30, sig=6.0):
    from scipy.stats import norm
    rng = np.random.default_rng(seed)
    slots = [(16, 16), (16, 48), (48, 16), (48, 48)]
    probs = np.zeros((n, hw, hw)); gts = np.zeros((n, hw, hw), bool)
    planned = []
    for i in range(n):
        order = rng.permutation(4)
        gt_c, others = slots[order[0]], [slots[j] for j in order[1:]]
        gts[i] = _blob((hw, hw), gt_c, 6)
        if rng.random() < 0.4:                       # failure
            r = rng.choice([2, 3, np.inf], p=[0.4, 0.3, 0.3])
            if np.isinf(r):
                probs[i][gts[i]] = 0.1               # below cand_thr -> missed
                probs[i][_blob((hw, hw), others[0], 6)] = 0.9
            else:
                probs[i][gts[i]] = 0.45
                for k in range(int(r) - 1):
                    probs[i][_blob((hw, hw), others[k], 6)] = 0.9 - 0.1 * k
            planned.append(r)
        else:                                        # success (+ one weak distractor)
            probs[i][gts[i]] = 0.95
            probs[i][_blob((hw, hw), others[0], 6)] = 0.3
            planned.append(0)
    is_fail = np.array(planned) != 0
    # flag features with known AUROC: A informative (shift 1), B uninformative
    fa = rng.normal(0, 1, n) + 1.0 * is_fail
    fb = rng.normal(0, 1, n)
    feats = np.stack([fa, fb], 1)
    # gradients: failures of 'aligned' type share direction u; other failures use orthogonal v
    u = rng.normal(size=p); u /= np.linalg.norm(u)
    v = rng.normal(size=p); v -= v @ u * u; v /= np.linalg.norm(v)
    g_flag = sig * u + rng.normal(size=(n_flag, p))
    aligned = rng.random(n) < 0.5
    g_eval = np.where(aligned[:, None], sig * u, sig * v) + rng.normal(size=(n, p))
    g_ctrl = rng.normal(size=(100, p))
    exp_cos = (sig * sig * n_flag) / (np.sqrt(sig ** 2 + p) * np.sqrt(n_flag ** 2 * sig ** 2 + n_flag * p))
    d = dict(probs=probs, gts=gts, features=feats, feature_names=np.array(["informative", "noise"]),
             grads_eval=g_eval, grads_flag=g_flag, grads_ctrl=g_ctrl)
    truth = dict(planned=np.array(planned), aligned=aligned, exp_cos=exp_cos,
                 exp_auc=float(norm.cdf(1.0 / np.sqrt(2))))
    return d, truth


def selftest():
    d, t = _synthetic()
    rows, s = analyse(d, delta=0.5, taus={"informative": 0.5, "noise": 0.5})
    ok = True

    def check(name, cond, info=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name} {info}")

    fail = t["planned"] != 0
    exp_r = {k: float(np.mean(np.isin(t["planned"][fail], [r for r in (1, 2, 3, 5) if r <= k]))) for k in (1, 2, 3, 5)}
    got = s["recall_at_k_on_failures"]
    check("recall@k matches planned ranks", all(abs(got[k] - exp_r[k]) < 1e-9 for k in got), f"(got {got}, expected {exp_r})")
    check("failure count", s["n_failures"] == int(fail.sum()))
    ft = s["failure_types"]
    check("failure types (wrong-choice / missed)",
          ft["wrong-choice"] == int(np.isin(t["planned"], [2, 3]).sum()) and ft["missed"] == int(np.isinf(t["planned"]).sum()),
          f"{ft}")
    a = s["features"]["informative"]; b = s["features"]["noise"]
    check("AUROC informative ~ Phi(1/sqrt2)", abs(a["auroc"] - t["exp_auc"]) < 0.08 and a["auroc_ci95"][0] <= t["exp_auc"] <= a["auroc_ci95"][1],
          f"({a['auroc']:.3f} vs {t['exp_auc']:.3f}, CI {a['auroc_ci95'][0]:.3f}-{a['auroc_ci95'][1]:.3f})")
    check("AUROC noise ~ 0.5", b["auroc_ci95"][0] <= 0.5 <= b["auroc_ci95"][1], f"({b['auroc']:.3f})")
    al = rows["rho_G"]; al_aligned = al[t["aligned"]].mean(); al_orth = al[~t["aligned"]].mean()
    check("rho_G aligned ~ analytic cosine", abs(al_aligned - t["exp_cos"]) < 0.05, f"({al_aligned:.3f} vs {t['exp_cos']:.3f})")
    check("rho_G orthogonal ~ 0", abs(al_orth) < 0.05, f"({al_orth:.3f})")
    check("A>0 only for aligned cases (mostly)", np.mean(rows["A"][t["aligned"]] > 0) > 0.95 and 0.3 < np.mean(rows["A"][~t["aligned"]] > 0) < 0.7)
    check("control rho_max below flagged rho_max", s["control_alignment_on_failures"]["rho_G"] < s["alignment_on_failures"]["rho_G"],
          f"({s['control_alignment_on_failures']['rho_G']:.3f} < {s['alignment_on_failures']['rho_G']:.3f})")
    # detection ceiling: shape feature = 'noise' (uninformative) -> within-V AUROC about 0.5
    rows2, s2 = analyse(d, delta=0.5, taus={"informative": 0.5, "noise": 1.0}, shape_features=["noise"])
    c = s2["detection_ceiling"]
    check("detection ceiling reported", 0 <= c["nu_TPR_of_shape_flags"] <= 1 and c["n_failures_inside_V"] > 0,
          f"(nu={c['nu_TPR_of_shape_flags']:.2f}, within-V AUROC informative={c['auroc_within_V']['informative']:.2f})")
    print("\nSELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz")
    ap.add_argument("--out-dir", default="diag_out")
    ap.add_argument("--delta", type=float, default=0.5, help="Dice below this = true failure")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3, 5])
    ap.add_argument("--cand-thr", type=float, default=0.2, help="probability threshold for candidate regions")
    ap.add_argument("--score", default="max", choices=["max", "sum", "mean"], help="candidate ranking score")
    ap.add_argument("--hit-frac", type=float, default=0.3, help="fraction of GT area a candidate must cover to count as a hit")
    ap.add_argument("--tau", action="append", default=[], help="feature=threshold (repeatable)")
    ap.add_argument("--shape-features", nargs="*", default=None, help="names of features that are functions of the mask only")
    ap.add_argument("--gamma-over-delta", type=float, default=None, help="Gamma/Delta for the break-even precision test (Part A)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if not a.npz:
        ap.error("--npz is required (or use --selftest)")
    d = dict(np.load(a.npz, allow_pickle=True))
    taus = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in a.tau}
    rows, summary = analyse(d, a.delta, tuple(a.ks), a.cand_thr, a.score, a.hit_frac, taus,
                            a.shape_features, a.gamma_over_delta, a.seed)
    _write_outputs(rows, summary, a.out_dir)


if __name__ == "__main__":
    main()