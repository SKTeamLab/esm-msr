"""Unified pLDDT conditioning helpers for ESM3 forward passes.

Supported ``--plddt_mode`` values:
  - ``zero``  (default): pass ``per_res_plddt=None`` so ESM3 uses its built-in zero-filled default
    (preserves exact legacy / manuscript ``FINAL_p0`` behavior).
  - ``one``: pass ``per_res_plddt=1.0`` wherever backbone coordinates are finite and ``0.0`` elsewhere
    (matches official ``ESM3.logits()`` conditioning).
  - ``infer``: infer real per-residue pLDDTs from the structure/cache and normalize to ESM3's ``[0, 1]``
    RBF domain, along with the per-protein ``average_plddt``. Structures without valid predicted pLDDTs
    (e.g. experimental crystal/NMR B-factors or all-zero Rosetta/ideal backbones) automatically fall back
    to ``1.0`` on finite-coordinate residues.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Tuple, Union

import torch

PLDDT_MODES: Tuple[str, ...] = ('zero', 'one', 'infer')
_EXPERIMENTAL_KEYWORDS = (
    'X-RAY',
    'NMR',
    'ELECTRON MICROSCOPY',
    'CRYO-EM',
    'NEUTRON',
    'DIFFRACTION',
    'SOLUTION SCATTERING',
)


def resolve_plddt_mode(
    plddt_mode: Optional[str] = None,
    default_plddt_one: Optional[bool] = None,
    use_plddt: Optional[bool] = None,
    default: str = 'zero',
) -> str:
    """Resolve effective pLDDT mode from explicit flag or legacy boolean aliases."""
    if plddt_mode is not None:
        mode = str(plddt_mode).strip().lower()
        if mode not in PLDDT_MODES:
            raise ValueError(f"Invalid plddt_mode {plddt_mode!r}; expected one of {PLDDT_MODES}.")
        return mode
    if use_plddt:
        return 'infer'
    if default_plddt_one:
        return 'one'
    if default not in PLDDT_MODES:
        raise ValueError(f"Invalid default plddt_mode {default!r}; expected one of {PLDDT_MODES}.")
    return default


def plddt_mode_from_hparams(hparams: Mapping) -> str:
    """Read effective pLDDT mode from a checkpoint's ``hparams.yaml`` (backward-compatible)."""
    raw_mode = hparams.get('plddt_mode', None)
    if raw_mode is not None:
        return resolve_plddt_mode(plddt_mode=raw_mode)
    return resolve_plddt_mode(
        default_plddt_one=bool(hparams.get('default_plddt_one', False)),
        use_plddt=bool(hparams.get('use_plddt', False)),
        default='zero',
    )


def is_experimental_pdb(pdb_path: Union[str, Path]) -> bool:
    """Return True if PDB headers indicate an experimental crystal/NMR/EM structure.

    AlphaFold2 / Biopython-exported predicted PDBs often contain a dummy unit-cell line
    ``CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1``; we ignore
    that placeholder and only treat non-trivial unit cells or experimental ``EXPDTA``
    records as experimental structures.
    """
    try:
        with open(pdb_path, 'r', errors='ignore') as fh:
            for line in fh:
                rec = line[:6].strip()
                if rec in ('ATOM', 'HETATM'):
                    break
                if rec == 'EXPDTA':
                    upper = line.upper()
                    if any(kw in upper for kw in _EXPERIMENTAL_KEYWORDS):
                        return True
                elif rec == 'CRYST1':
                    try:
                        a = float(line[6:15])
                        b = float(line[15:24])
                        c = float(line[24:33])
                    except ValueError:
                        continue
                    is_dummy_unit_cell = (
                        abs(a - 1.0) < 1e-2 and abs(b - 1.0) < 1e-2 and abs(c - 1.0) < 1e-2
                    )
                    if not is_dummy_unit_cell and (a > 2.0 or b > 2.0 or c > 2.0):
                        return True
    except OSError:
        return False
    return False


def resolve_plddt_tensors(
    plddt_mode: str,
    coords: Optional[torch.Tensor],
    per_res_plddt: Optional[torch.Tensor],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Compute ``(per_res_plddt, average_plddt)`` in ESM3's ``[0, 1]`` float32 RBF domain.

    Args:
        plddt_mode: One of ``'zero'``, ``'one'``, ``'infer'``.
        coords: Prepared 4D structure coordinates ``[B, L, A, 3]`` (or ``None``).
        per_res_plddt: Prepared 2D per-residue confidence tensor ``[B, L]`` (or ``None``).

    Returns:
        ``(per_res, avg)`` tensors of shape ``[B, L]`` (float32) or ``None`` to let
        ``ESM3.forward`` apply its built-in defaults.
    """
    if coords is None or plddt_mode == 'zero':
        return None, None

    finite_mask = coords.isfinite().all(dim=-1).any(dim=-1)  # [B, L]
    ones = finite_mask.to(dtype=torch.float32)

    if plddt_mode == 'one':
        return ones, None

    if plddt_mode != 'infer':
        raise ValueError(f"Invalid plddt_mode {plddt_mode!r}; expected one of {PLDDT_MODES}.")

    if per_res_plddt is None:
        per_res = ones
    else:
        raw = per_res_plddt.to(device=coords.device, dtype=torch.float32)
        if raw.shape != finite_mask.shape:
            # Broadcast single-structure pLDDT [1, L] across batch [B, L] if needed
            if raw.shape[0] == 1 and raw.shape[1] == finite_mask.shape[1]:
                raw = raw.expand_as(finite_mask)
            else:
                raise ValueError(f"per_res_plddt shape {tuple(raw.shape)} does not match coords mask {tuple(finite_mask.shape)}")

        per_res = torch.zeros_like(ones)
        B = finite_mask.shape[0]
        for b in range(B):
            m = finite_mask[b]
            vals = raw[b][m]
            valid_vals = vals[vals.isfinite()]
            if valid_vals.numel() == 0:
                per_res[b] = ones[b]
                continue
            vmin = float(valid_vals.min().item())
            vmax = float(valid_vals.max().item())
            # Crystal Debye-Waller B-factors (> 100), negative values, or all-zero placeholders
            # (e.g. Rosetta de novo designs / stripped PDBs) fall back to 1.0 on finite residues.
            if vmin < -1e-3 or vmax > 100.0 + 1e-3 or vmax <= 0.0:
                per_res[b] = ones[b]
            elif vmax > 1.0 + 1e-3:
                # AlphaFold2 / ColabFold / AFDB [0, 100] scale -> [0, 1]
                clean = torch.where(raw[b].isfinite(), raw[b] / 100.0, torch.zeros_like(raw[b]))
                per_res[b] = clean.clamp(0.0, 1.0) * ones[b]
            else:
                # Already on [0, 1] scale (ESMFold, pre-normalized, or crystal ones)
                clean = torch.where(raw[b].isfinite(), raw[b], torch.zeros_like(raw[b]))
                per_res[b] = clean.clamp(0.0, 1.0) * ones[b]

    denom = ones.sum(dim=-1, keepdim=True).clamp(min=1.0)
    avg = (per_res.sum(dim=-1, keepdim=True) / denom).expand_as(per_res)
    return per_res, avg
