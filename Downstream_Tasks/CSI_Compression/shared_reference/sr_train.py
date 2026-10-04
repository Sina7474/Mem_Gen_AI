"""
sr_train.py — CRNet trainer for the shared-reference experiment
===============================================================
A thin, self-contained training loop that reuses the vetted CRNet architecture
and NMSE metric from ``src/crnet_core.py`` but adds the convergence controls the
implementation guide (section 8) asks for and that the legacy trainer lacks:

  * validation-based early stopping with patience,
  * restore-best-validation weights,
  * rich run-level record: best_epoch, max_epoch, early-stopping status,
    final learning rate, best validation and test NMSE (dB).

Nothing here modifies the legacy Task-09 code.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from crnet_core import CRNet, nmse_db  # noqa: E402


def _loader(x, batch_size, shuffle, gen=None):
    return DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=shuffle,
                      pin_memory=True, num_workers=2, generator=gen)


def _nmse_over(model, loader, device):
    preds, tgts = [], []
    with torch.no_grad():
        for (x,) in loader:
            preds.append(model(x.to(device)).cpu())
            tgts.append(x)
    return nmse_db(torch.cat(preds), torch.cat(tgts))


def train_crnet_convergence(
    X_train, X_val, X_test, *, tag, log_path: Path, ckpt_path: Path,
    nr=4, nt=32, reduction=4, max_epochs=500, patience=None,
    batch_size=512, lr=1e-3, seed=0, device=None, skip_done=True,
    extra_meta=None,
):
    """Train one CRNet and cache the run record to ``log_path``.

    Matches the legacy Task-09 trainer: full cosine schedule over ``max_epochs``,
    restore best-validation weights. ``patience`` is OFF by default (``None``);
    when set to a positive int it enables validation-based early stopping (mainly
    for auditing — the model reaches its floor only near the end of the schedule,
    so early stopping typically leaves it under-trained).
    """
    if skip_done and log_path.exists():
        d = json.loads(log_path.read_text())
        print(f"  [CACHED] {tag}: test={d['test_nmse_db']:.2f} dB "
              f"(best_ep={d['best_epoch']}/{d['max_epoch']})")
        return d

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    g = torch.Generator();  g.manual_seed(seed)

    tr_ld = _loader(X_train, batch_size, True, g)
    vl_ld = _loader(X_val, batch_size, False)
    te_ld = _loader(X_test, batch_size, False)

    model = CRNet(nr=nr, nt=nt, reduction=reduction).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=1e-5)
    crit = nn.MSELoss()

    best_val, best_state, best_epoch = float("inf"), None, 0
    epochs_since_best = 0
    early_stopped = False
    t0 = time.time()
    last_epoch = 0

    for ep in range(1, max_epochs + 1):
        last_epoch = ep
        model.train()
        for (x,) in tr_ld:
            x = x.to(device)
            opt.zero_grad()
            crit(model(x), x).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            vl = sum(crit(model(x.to(device)), x.to(device)).item()
                     for (x,) in vl_ld) / len(vl_ld)

        if vl < best_val - 1e-7:
            best_val, best_epoch = vl, ep
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_since_best = 0
        else:
            epochs_since_best += 1
        sch.step()

        if ep % 50 == 0:
            print(f"    [{tag}] ep {ep:4d}/{max_epochs}  val_mse={vl:.6f}  "
                  f"best_ep={best_epoch}  elapsed={time.time()-t0:.0f}s")

        if patience is not None and epochs_since_best >= patience:
            early_stopped = True
            print(f"    [{tag}] early stop at ep {ep} (no val gain for "
                  f"{patience} epochs; best_ep={best_epoch})")
            break

    model.load_state_dict(best_state)
    model.eval()
    test_nmse = _nmse_over(model, te_ld, device)
    val_nmse = _nmse_over(model, vl_ld, device)
    final_lr = float(opt.param_groups[0]["lr"])
    elapsed = time.time() - t0

    # Convergence heuristic (guide section 8): a full run that keeps improving to
    # the very last epoch may still be under-trained → flag it for inspection.
    near_max = best_epoch >= int(0.95 * max_epochs)
    converged = early_stopped or not near_max

    print(f"  [{tag}] test={test_nmse:.2f} dB  val={val_nmse:.2f} dB  "
          f"best_ep={best_epoch}/{last_epoch}  early={early_stopped}  "
          f"time={elapsed:.0f}s")

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, str(ckpt_path))
    rec = {
        "tag": tag, "n_train": int(len(X_train)), "n_val": int(len(X_val)),
        "n_test": int(len(X_test)), "reduction": reduction,
        "max_epoch": int(last_epoch), "max_epochs_budget": int(max_epochs),
        "patience": (int(patience) if patience is not None else None),
        "best_epoch": int(best_epoch),
        "early_stopped": bool(early_stopped), "converged_flag": bool(converged),
        "best_epoch_near_max": bool(near_max),
        "test_nmse_db": float(test_nmse), "validation_nmse_db": float(val_nmse),
        "final_lr": final_lr, "seed": int(seed), "batch_size": int(batch_size),
        "train_time_s": float(elapsed),
    }
    if extra_meta:
        rec.update(extra_meta)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(rec, indent=2))
    return rec
