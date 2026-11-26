"""Unit tests for ProSST unmasked and independent-masking dual-view scoring (`prosst_scoring.py`)."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

BASELINES_DIR = Path(__file__).resolve().parents[1] / 'inference_scripts' / 'baselines'
sys.path.insert(0, str(BASELINES_DIR))

from prosst_scoring import parse_mutation, score_protein_variants, structure_tokens  # noqa: E402


class _MockTokenizer:
    """Matches ProSST's exact 25-token vocabulary."""

    TOKENS = [
        '<pad>', '<cls>', '<eos>',
        'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
        'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
        '<unk>', '<mask>',
    ]

    def __init__(self):
        self._vocab = {t: i for i, t in enumerate(self.TOKENS)}
        self.mask_token_id = self._vocab['<mask>']

    def get_vocab(self):
        return dict(self._vocab)

    def __call__(self, seqs, return_tensors='pt'):
        rows = [[1, *[self._vocab[ch] for ch in s], 2] for s in seqs]
        t = torch.tensor(rows, dtype=torch.long)
        return {'input_ids': t, 'attention_mask': torch.ones_like(t)}


class _CoupledMockProSST(torch.nn.Module):
    """Lightweight differentiable mock with explicit pairwise sequence-context + structure coupling."""

    def __init__(self, vocab_size: int = 25, ss_vocab_size: int = 2051):
        super().__init__()
        self.calls = 0
        self.total_examples = 0
        self.vocab_size = vocab_size
        g = torch.Generator().manual_seed(42)
        self.aa_embed = torch.nn.Parameter(torch.randn(vocab_size, 8, generator=g), requires_grad=False)
        self.ss_embed = torch.nn.Parameter(torch.randn(ss_vocab_size, 8, generator=g), requires_grad=False)
        self.pair_proj = torch.nn.Parameter(torch.randn(8, vocab_size, generator=g), requires_grad=False)

    def forward(self, input_ids, attention_mask=None, ss_input_ids=None):
        self.calls += 1
        self.total_examples += int(input_ids.shape[0])
        # Context-dependent logits: each position attends to all neighboring tokens (including mutants!) + structure
        h = self.aa_embed[input_ids]  # [B, L+2, 8]
        h = torch.where((input_ids == 24).unsqueeze(-1), torch.zeros_like(h), h)
        ss = self.ss_embed[ss_input_ids]
        # Mean over entire sequence creates non-zero pairwise epistasis between any two sites
        ctx = h.mean(dim=1, keepdim=True)
        logits = (h + 0.5 * ss + 1.5 * ctx) @ self.pair_proj
        return SimpleNamespace(logits=logits)


def test_structure_tokens_and_parse():
    st = structure_tokens([0, 5, 2047])
    assert st.tolist() == [[1, 3, 8, 2050, 2]]
    assert parse_mutation('A1G:C3W') == [('A', 0, 'G'), ('C', 2, 'W')]


def test_wt_unmasked_matches_upstream_and_is_additive():
    tok = _MockTokenizer()
    model = _CoupledMockProSST()
    seq = 'ACDEFGHIK'
    sst = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    muts = ['A1G', 'D3W', 'A1G:D3W']

    res = score_protein_variants(model, tok, seq, sst, muts, mode='wt_unmasked')
    assert res.n_wt_passes == 1
    assert res.n_mt_passes == 0
    assert np.allclose(res.epi_pred, 0.0)
    assert np.isclose(res.comb_pred[2], res.comb_pred[0] + res.comb_pred[1], atol=1e-6)

    # Exact parity with upstream proteingym_benchmark.py formula
    enc = tok([seq])
    lg = model(input_ids=enc['input_ids'], attention_mask=enc['attention_mask'], ss_input_ids=structure_tokens(sst)).logits
    lp = torch.log_softmax(lg[:, 1:-1, :], dim=-1)[0]
    v = tok.get_vocab()
    expected_single = float(lp[0, v['G']] - lp[0, v['A']])
    assert np.isclose(res.comb_pred[0], expected_single, atol=1e-6)


def test_dual_independent_singles_zero_epistasis_and_exact_cycle():
    tok = _MockTokenizer()
    model = _CoupledMockProSST()
    seq = 'ACDEFGHIK'
    sst = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    muts = ['A1G', 'A1W', 'D3V', 'A1G:D3V', 'A1W:D3V', 'A1G:D3V:H7Y']

    res = score_protein_variants(model, tok, seq, sst, muts, mode='dual_independent', batch_size=4)

    # 1. Singles: masking pos i in WT == masking pos i in MT -> mt_pred == wt_pred, epi == 0 identically!
    for i in (0, 1, 2):
        assert np.isclose(res.wt_pred[i], res.mt_pred[i], atol=1e-7)
        assert np.isclose(res.epi_pred[i], 0.0, atol=1e-7)

    # 2. Doubles: WT pass is strictly additive, MT pass sees partner mutant -> non-zero epistasis!
    assert np.isclose(res.wt_pred[3], res.wt_pred[0] + res.wt_pred[2], atol=1e-6)
    assert abs(res.epi_pred[3]) > 1e-4

    # 3. Exact thermodynamic cycle identity:
    #    comb(AB) - (comb(A) + comb(B)) == 0.5 * (mt(AB) - wt(AB)) == epi_pred(AB)
    cycle_epi = res.comb_pred[3] - (res.comb_pred[0] + res.comb_pred[2])
    assert np.isclose(cycle_epi, res.epi_pred[3], atol=1e-6)

    # 4. Higher-order (k=3) runs cleanly and yields non-zero epistasis
    assert np.isfinite(res.comb_pred[5]) and abs(res.epi_pred[5]) > 1e-4

    # 5. State deduplication: A1G and A1W share the same WT masked state at pos 0 (only 3 WT masked positions: 0, 2, 6)
    assert res.n_wt_passes == 3
    # MT states:
    #   singles A1G, A1W, D3V -> 0 new states (identical to WT masked states!)
    #   A1G:D3V -> (mask 0 w/ D3V), (mask 2 w/ A1G) = 2
    #   A1W:D3V -> (mask 0 w/ D3V -> DEDUPLICATED!), (mask 2 w/ A1W) = 1
    #   A1G:D3V:H7Y -> 3 new states
    #   Total MT unique states = 2 + 1 + 3 = 6
    assert res.n_mt_passes == 6


def test_dual_unmasked_computes_nonzero_epistasis():
    tok = _MockTokenizer()
    model = _CoupledMockProSST()
    seq = 'ACDEFGHIK'
    sst = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    muts = ['A1G', 'D3V', 'A1G:D3V']

    res = score_protein_variants(model, tok, seq, sst, muts, mode='dual_unmasked')
    assert res.n_wt_passes == 1
    assert res.n_mt_passes == 3  # 3 distinct unmasked mutant sequences
    cycle_epi = res.comb_pred[2] - (res.comb_pred[0] + res.comb_pred[1])
    assert np.isfinite(cycle_epi) and abs(cycle_epi) > 1e-4


def test_wt_mismatch_raises():
    tok = _MockTokenizer()
    model = _CoupledMockProSST()
    with pytest.raises(AssertionError, match='WT mismatch'):
        score_protein_variants(model, tok, 'ACDEF', [1, 2, 3, 4, 5], ['G1A'], mode='dual_independent')
