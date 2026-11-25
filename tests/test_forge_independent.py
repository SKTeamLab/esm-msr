"""Unit tests for Forge ESM3 independent-masking runner (no network calls required)."""
import asyncio
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "inference_scripts" / "esm3_zs_testing.py"
spec = importlib.util.spec_from_file_location("esm3_zs_testing", SCRIPT_PATH)
forge_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(forge_mod)


def test_no_hardcoded_forge_token_in_source():
    src = SCRIPT_PATH.read_text()
    # Ensure no hard-coded Bearer/Forge token literal remains in the script
    import re
    assert not re.search(r'token\s*=\s*["\'][A-Za-z0-9_-]{16,}["\']', src)
    assert "ESM3ForgeInferenceClient(" not in src


def test_resolve_forge_token_requires_secret(monkeypatch):
    for k in ("ESM_API_KEY", "FORGE_API_KEY", "ESM_FORGE_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(RuntimeError, match="No Forge API token"):
        forge_mod.resolve_forge_token()
    monkeypatch.setenv("ESM_API_KEY", "test_secret_123")
    assert forge_mod.resolve_forge_token() == "test_secret_123"


def test_adaptive_token_limiter_parses_429_cap():
    lim = forge_mod.AdaptiveTokenLimiter()
    asyncio.run(
        lim.note_rate_limit(
            'Failure in logits: {"status":"error","message":"You have exceeded your usage cap of 200000 tokens during the last minute"}'
        )
    )
    assert lim.tokens_per_min == int(200000 * 0.94)


def test_independent_masking_matches_singles_doubles_and_multis(monkeypatch):
    """Verify exact algebra for singles (K=1), doubles (K=2 == 2-path), and triples (K=3)."""
    pred = forge_mod.ESM3ForgePredictor(token="dummy", model_name="esm3-small-2024-08")
    # Synthetic deterministic 6-aa protein: A C D E F G
    wt_seq = "ACDEFG"
    vocab = pred.vocab
    mask_id = pred.mask_token_id
    seq_ids = [0] + [vocab[a] for a in wt_seq] + [2]

    calls = []

    async def fake_encode(http_client, sem, pdb_path, chain_id, backbone_mutation=None):
        return {
            "wt_seq": wt_seq,
            "sequence": seq_ids,
            "structure": [0] * len(seq_ids),
            "coordinates": [],
            "n_tokens": len(seq_ids),
        }

    def fake_logits_for_state(overrides):
        """Deterministic non-additive logit table depending on overrides."""
        ov_dict = dict(overrides)
        out = {}
        for pos in range(1, len(wt_seq) + 1):
            vec = torch.zeros(64, dtype=torch.float32)
            # Context shift: sum of non-mask mutated token ids in overrides
            ctx_sum = sum(v for p, v in ov_dict.items() if v != mask_id)
            for aa, tid in vocab.items():
                if isinstance(tid, int) and 0 <= tid < 64:
                    vec[tid] = float(pos * 0.5 + tid * 0.1 + 0.01 * ctx_sum * pos)
            out[pos] = vec
        return out

    async def fake_fetch(http_client, sem, encoded_wt, overrides, positions_needed):
        calls.append(overrides)
        tbl = fake_logits_for_state(overrides)
        return {p: tbl[p] for p in positions_needed}

    monkeypatch.setattr(pred, "_encode_wt", fake_encode)
    monkeypatch.setattr(pred, "_fetch_state_positions", fake_fetch)

    df = pd.DataFrame({
        "mut_type": ["A1V", "C2W", "D3Y", "A1V:C2W", "A1V:C2W:D3Y"],
    })
    res = pred.infer_mutants_independent("dummy.pdb", df, chain="A", mut_col="mut_type")

    def delta(ov, pos, wt, mt):
        vec = fake_logits_for_state(ov)[pos]
        return (vec[vocab[mt]] - vec[vocab[wt]]).item()

    d1 = delta(((1, mask_id),), 1, "A", "V")
    d2 = delta(((2, mask_id),), 2, "C", "W")
    d3 = delta(((3, mask_id),), 3, "D", "Y")

    # Singles
    assert np.isclose(res.loc[0, "esm3_score"], d1)
    assert np.isclose(res.loc[0, "wt_lora_pred"], d1)
    assert np.isclose(res.loc[0, "mt_lora_pred"], d1)
    assert np.isclose(res.loc[0, "esm3_epistasis_score"], 0.0)

    # Double: verify exact identity with 2-path symmetric average!
    d2_given_1 = delta(((1, vocab["V"]), (2, mask_id)), 2, "C", "W")
    d1_given_2 = delta(((1, mask_id), (2, vocab["W"])), 1, "A", "V")
    two_path = 0.5 * ((d1 + d2_given_1) + (d2 + d1_given_2))
    ind_double = 0.5 * ((d1 + d2) + (d1_given_2 + d2_given_1))
    assert np.isclose(two_path, ind_double)
    assert np.isclose(res.loc[3, "pred_additive"], d1 + d2)
    assert np.isclose(res.loc[3, "mt_lora_pred"], d1_given_2 + d2_given_1)
    assert np.isclose(res.loc[3, "esm3_score"], two_path)
    assert np.isclose(res.loc[3, "esm3_epistasis_score"], two_path - (d1 + d2))

    # Triple: independent masking (each site masked with the other 2 mutations present)
    d1_23 = delta(((1, mask_id), (2, vocab["W"]), (3, vocab["Y"])), 1, "A", "V")
    d2_13 = delta(((1, vocab["V"]), (2, mask_id), (3, vocab["Y"])), 2, "C", "W")
    d3_12 = delta(((1, vocab["V"]), (2, vocab["W"]), (3, mask_id)), 3, "D", "Y")
    wt3 = d1 + d2 + d3
    mt3 = d1_23 + d2_13 + d3_12
    assert np.isclose(res.loc[4, "pred_additive"], wt3)
    assert np.isclose(res.loc[4, "mt_lora_pred"], mt3)
    assert np.isclose(res.loc[4, "esm3_score"], 0.5 * (wt3 + mt3))
    assert np.isclose(res.loc[4, "esm3_epistasis_score"], 0.5 * (mt3 - wt3))

    # Deduplication: 3 WT states + 2 double MT states + 3 triple MT states = 8 unique calls total!
    assert len(calls) == 8
