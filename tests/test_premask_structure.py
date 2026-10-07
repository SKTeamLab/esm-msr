"""Masking the mutated sites' geometry BEFORE the structure encoder (data.premask_structure) and the shared residue-blanking helper."""
import unittest

import torch
from torch import nn

from esm.utils.constants import esm3 as C

from esm_msr.data import blank_residues, encode_structure, premask_structure

L, A = 12, 37


class StubEncoder(nn.Module):
    """Records what it is asked to encode; tokens are the residue's index (so tokens can be traced), and a residue with no coordinates gets -1."""
    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.zeros(1))
        self.seen = []

    def encode(self, coords, residue_index=None):
        self.seen.append(coords.clone())
        n = coords.shape[-3]
        tok = torch.arange(n).unsqueeze(0)
        absent = torch.isnan(coords[..., 1, :]).all(dim=-1)        # the CA atom missing for the residue
        return None, torch.where(absent, torch.full_like(tok, -1), tok)


def chain():
    c = torch.randn(1, L, A, 3)
    c[..., 5:, :] = float('nan')                                   # side-chain atoms are absent in the real input too
    return c, torch.full((1, L), 90.0), torch.arange(L).unsqueeze(0)


class TestBlankResidues(unittest.TestCase):
    def test_masks_the_requested_residues_on_the_residue_axis(self):
        c, p, _ = chain(); before = c.clone()
        blank_residues(c, p, [4, 9])                                # 1-based
        for i in range(L):
            if i in (3, 8):
                self.assertTrue(torch.isnan(c[0, i]).all())
                self.assertEqual(float(p[0, i]), 0.0)
            else:
                self.assertTrue(torch.equal(torch.nan_to_num(c[0, i], nan=-1.0), torch.nan_to_num(before[0, i], nan=-1.0)))
                self.assertEqual(float(p[0, i]), 90.0)

    def test_the_first_residue_does_not_blank_the_whole_chain(self):
        c, p, _ = chain()
        blank_residues(c, p, [1])
        self.assertTrue(torch.isnan(c[0, 0]).all())
        self.assertFalse(torch.isnan(c[0, 1, :5]).any())            # the backbone atoms of residue 2 are untouched
        self.assertEqual(int((p == 0).sum()), 1)

    def test_positions_outside_the_chain_are_ignored(self):
        c, p, _ = chain(); before = c.clone()
        blank_residues(c, p, [0, L + 1, 99])
        self.assertTrue(torch.equal(torch.nan_to_num(c, nan=-1.0), torch.nan_to_num(before, nan=-1.0)))

    def test_works_without_the_leading_batch_axis(self):
        c, p = torch.randn(L, A, 3), torch.full((L,), 90.0)
        blank_residues(c, p, [3])
        self.assertTrue(torch.isnan(c[2]).all()); self.assertFalse(torch.isnan(c[3]).all()); self.assertEqual(float(p[2]), 0.0)


class TestPremask(unittest.TestCase):
    def test_encode_structure_adds_bos_and_eos(self):
        c, _, r = chain()
        tok = encode_structure(StubEncoder(), c, r)
        self.assertEqual(tuple(tok.shape), (L + 2,))
        self.assertEqual((int(tok[0]), int(tok[-1])), (C.STRUCTURE_BOS_TOKEN, C.STRUCTURE_EOS_TOKEN))

    def test_the_encoder_sees_the_blanked_chain_and_the_output_is_padded_like_an_item(self):
        enc = StubEncoder()
        c, p, r = chain()
        cp = torch.nn.functional.pad(c, (0, 0, 0, 0, 1, 1), value=float('inf'))
        pp = torch.nn.functional.pad(p, (1, 1), value=0)
        mc, mt = premask_structure(enc, cp, pp, r, [4, 9])
        seen = enc.seen[0]
        self.assertEqual(tuple(seen.shape), (1, L, A, 3))           # the padding is stripped before encoding
        self.assertTrue(torch.isnan(seen[0, 3]).all() and torch.isnan(seen[0, 8]).all())
        self.assertFalse(torch.isnan(seen[0, 0, :5]).any())
        self.assertEqual(tuple(mc.shape), (1, L + 2, A, 3)); self.assertEqual(tuple(mt.shape), (L + 2,))
        self.assertTrue(torch.isinf(mc[0, 0]).all() and torch.isinf(mc[0, -1]).all())
        self.assertEqual([int(mt[i + 1]) for i in (3, 8)], [-1, -1])           # the hidden residues come back as 'absent' tokens
        self.assertEqual(int(mt[1]), 0)                                         # the others are encoded from their own coordinates

    def test_the_stored_structure_is_not_modified(self):
        c, p, r = chain()
        cp = torch.nn.functional.pad(c, (0, 0, 0, 0, 1, 1), value=float('inf')); before = cp.clone()
        premask_structure(StubEncoder(), cp, torch.nn.functional.pad(p, (1, 1)), r, [2])
        self.assertTrue(torch.equal(torch.nan_to_num(cp, nan=-1.0, posinf=7.0), torch.nan_to_num(before, nan=-1.0, posinf=7.0)))


if __name__ == '__main__':
    unittest.main()
