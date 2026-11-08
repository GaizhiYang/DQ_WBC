"""Fail before nonfinite tensors cross the real simulator boundary."""

from typing import Mapping, Optional

import torch


def require_finite_tensors(
    tensors: Mapping[str, torch.Tensor], *, phase: str, step=None,
    env_ids: Optional[Mapping[str, torch.Tensor]] = None,
) -> None:
    """Check a group on its device, with one host decision on the normal path.

    Tensor rows denote environments. For indexed uploads, ``env_ids[name]``
    maps those rows back to actual environment ids (possibly with repeats).
    Only failure diagnostics transfer row ids to the host. Values, gradients,
    and the random number generator are untouched; this never repairs data.
    """
    if not tensors:
        return
    values = list(tensors.items())
    device = values[0][1].device
    for name, value in values:
        if value.device != device:
            raise ValueError("finite-check tensors must be on the same device")
        if env_ids is not None and name in env_ids:
            rows = value.shape[0] if value.ndim else 1
            if env_ids[name].ndim != 1 or env_ids[name].numel() != rows:
                raise ValueError("env_ids must map every row of {}".format(name))

    # No per-tensor .item()/bool()/CPU copies on the successful GPU path.
    finite = torch.stack([torch.isfinite(value.detach()).all()
                          for _, value in values]).all()
    if bool(finite):
        return

    failures = []
    for name, value in values:
        invalid = ~torch.isfinite(value.detach())
        if invalid.ndim > 1:
            invalid = invalid.flatten(start_dim=1).any(dim=1)
        bad_rows = invalid.reshape(-1).nonzero(as_tuple=False).flatten()
        if not bad_rows.numel():
            continue
        if env_ids is not None and name in env_ids:
            mapping = env_ids[name]
            bad_ids = mapping[bad_rows.to(mapping.device)].unique()
        else:
            bad_ids = bad_rows
        shown = bad_ids[:16].cpu().tolist()
        failures.append("{}: env_ids={} ({} affected)".format(
            name, shown, bad_ids.numel()))
    raise FloatingPointError(
        "Nonfinite simulator input; phase={}; step={}; {}".format(
            phase, step, "; ".join(failures)))
