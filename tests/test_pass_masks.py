"""
Per-pass sequence masking (--mask_strategy_wt / --mask_strategy_mt) and the adapter override the pass probe reads with.

The real MSRModel.forward_partitioned runs here against a stand-in backbone whose logits depend on exactly what each site is shown, so the
structural claims the probe rests on are checked on the real masking code:
  * a double's pair-masked input is the same whichever sequence it starts from;
  * 'independent' on the wild type is exactly the sum of the two singles' masked readouts (the reason the probe needs no forward for it);
  * 'independent' on the mutated sequence sees the partner's residue, so it is not that sum.
"""
import types

import pytest
import torch

from esm_msr.models import MSRModel, pass_mask_strategy, _per_pass

V = 33          # vocabulary; 32 is ESM3's sequence mask token
SCALE = {'wt': 1.0, 'mt': 2.0}


class Head(torch.nn.Module):
    def __init__(self, s, b):
        super().__init__()
        self.s, self.b = s, b

    def forward(self, x):
        return self.s * x + self.b


class FakeModel:
    """What forward_partitioned reads from self: a backbone whose logit for token v at site p is SCALE[adapter] if v is the token shown at p,
    plus 0.01 * v * (the sum of every OTHER site's token), so a site's readout depends on what its partner shows."""
    adapter_mode = 'dual'
    mask_structure = False

    def __init__(self):
        self.calibration_head_wt = Head(0.5, 0.1)
        self.calibration_head_mt = Head(0.25, -0.2)
        self.models_used = []

    def _active_model(self, adapter):
        return adapter

    def _backbone_logits(self, seq, coords, struct_tokens, plddt, active_model):
        self.models_used.append(active_model)
        B, L = seq.shape
        onehot = torch.nn.functional.one_hot(seq, V).float() * SCALE[active_model]
        ctx = seq.float().sum(1, keepdim=True) - seq.float()                       # [B, L]: the other sites' tokens
        logits = onehot + 0.01 * ctx.unsqueeze(-1) * torch.arange(V).float()
        return logits, torch.arange(B)


WT_SEQ = [4, 5, 6, 7, 8, 9]


def item(muts):
    """One batch row for mutations [(pos, to)], as the dataset builds a single or a double on the real wild type."""
    mt = list(WT_SEQ)
    for p, t in muts:
        mt[p] = t
    pos = [p for p, _ in muts] + [0] * (2 - len(muts))
    return {'wt_sequence_tokens': torch.tensor([WT_SEQ]), 'mt_sequence_tokens': torch.tensor([mt]),
            'mut_pos': torch.tensor([pos]), 'wt_id': torch.tensor([[WT_SEQ[p] for p in pos]]),
            'mt_id': torch.tensor([[t for _, t in muts] + [0] * (2 - len(muts))]),
            'mut_mask': torch.tensor([[True] * len(muts) + [False] * (2 - len(muts))]),
            'coords': None, 'structure_tokens': None, 'plddt': None}


def run(batch, pass_type, mask=None, adapter=None, model=None):
    m = model or FakeModel()
    return MSRModel.forward_partitioned(m, batch, pass_type=pass_type, mask_strategy=mask, adapter=adapter)['pred_calibrated']


A, B_ = (1, 12), (3, 15)


def test_a_pair_masked_double_is_the_same_input_from_either_sequence():
    d = item([A, B_])
    torch.testing.assert_close(run(d, 'wt', 'marginal'), run(d, 'mt', 'marginal', adapter='wt'), atol=0, rtol=0)


def test_independent_on_the_wild_type_is_the_sum_of_the_singles_masked_readouts():
    d, a, b = item([A, B_]), item([A]), item([B_])
    head = FakeModel().calibration_head_wt
    raw = lambda bt: MSRModel.forward_partitioned(FakeModel(), bt, pass_type='wt', mask_strategy='independent')['pred_raw']
    torch.testing.assert_close(raw(d), raw(a) + raw(b), atol=1e-5, rtol=0)
    # and on the calibrated scale up to the one bias a summed item carries once
    torch.testing.assert_close(run(d, 'wt', 'independent') - head.b, (run(a, 'wt', 'marginal') - head.b) + (run(b, 'wt', 'marginal') - head.b),
                               atol=1e-5, rtol=0)


def test_independent_on_the_mutated_sequence_sees_the_partner():
    d, a, b = item([A, B_]), item([A]), item([B_])
    raw = lambda bt, pt: MSRModel.forward_partitioned(FakeModel(), bt, pass_type=pt, mask_strategy='independent')['pred_raw']
    assert not torch.allclose(raw(d, 'mt'), raw(a, 'mt') + raw(b, 'mt'), atol=1e-3)
    # but on a single there is no partner, so the conditional readout is the plain masked one
    torch.testing.assert_close(raw(a, 'mt'), raw(a, 'wt'), atol=1e-6, rtol=0)


def test_the_adapter_override_picks_the_lora_and_head_while_the_pass_picks_the_sequence():
    d = item([A])
    m = FakeModel()
    got = MSRModel.forward_partitioned(m, d, pass_type='mt', adapter='wt')
    assert m.models_used == ['wt']
    # mutant visible at A with the WT adapter's scale: logit(mt) - logit(wt) = +1 plus the context term
    ctx = sum(WT_SEQ) - WT_SEQ[A[0]]
    raw = 1.0 + 0.01 * ctx * (A[1] - WT_SEQ[A[0]])
    torch.testing.assert_close(got['pred_raw'], torch.tensor([raw]), atol=1e-5, rtol=0)
    torch.testing.assert_close(got['pred_calibrated'], torch.tensor([0.5 * raw + 0.1]), atol=1e-5, rtol=0)


def test_without_an_override_the_adapter_follows_the_pass():
    m = FakeModel()
    MSRModel.forward_partitioned(m, item([A]), pass_type='mt')
    MSRModel.forward_partitioned(m, item([A]), pass_type='wt', mask_strategy='marginal')
    assert m.models_used == ['mt', 'wt']


def test_a_wild_type_cache_cannot_serve_the_other_adapter():
    with pytest.raises(AssertionError, match='cached_wt_esm3'):
        MSRModel.forward_partitioned(FakeModel(), item([A]), pass_type='wt', adapter='mt', cached_wt_esm3={'seq': None})


# ---------------- resolution ----------------

@pytest.mark.parametrize('hp, want', [
    ({}, (None, None)),
    ({'mask_strategy': 'marginal'}, ('marginal', 'marginal')),                                   # the old single flag, both passes
    ({'mask_strategy': 'marginal', 'mask_strategy_mt': 'independent'}, ('marginal', 'independent')),
    ({'mask_strategy': 'marginal', 'mask_strategy_wt': 'none'}, (None, 'marginal')),              # 'none' unmasks one pass
    ({'mask_strategy_wt': 'marginal', 'mask_strategy_mt': 'independent'}, ('marginal', 'independent')),
    ({'mask_strategy_wt': None, 'mask_strategy': None}, (None, None)),                           # hparams with the keys unset
])
def test_each_pass_resolves_its_own_strategy(hp, want):
    assert (pass_mask_strategy(hp, 'wt'), pass_mask_strategy(hp, 'mt')) == want


def test_forward_batch_hands_each_pass_its_own_strategy():
    calls = []

    def fp(batch, pass_type, cached_wt_esm3=None, mask_strategy=None):
        calls.append((pass_type, mask_strategy))
        return {'pred_calibrated': torch.zeros(1), 'pred_raw': torch.zeros(1)}
    m = types.SimpleNamespace(training=False, adapter_mode='dual', forward_partitioned=fp)
    MSRModel.forward_batch(m, item([A]), mask_strategy={'wt': 'marginal', 'mt': 'independent'})
    MSRModel.forward_batch(m, item([A]), mask_strategy='marginal')
    assert calls == [('wt', 'marginal'), ('mt', 'independent'), ('wt', 'marginal'), ('mt', 'marginal')]
    assert _per_pass(None, 'wt') is None


# ---------------- training ----------------

def test_training_runs_each_pass_with_its_own_strategy():
    from test_compose_censoring import StubModel, make_batch, run as compose
    seen = []
    orig = StubModel.forward_partitioned

    def recording(self, micro, pass_type, mask_strategy=None, cached_wt_esm3=None):
        seen.append((pass_type, mask_strategy))
        return orig(self, micro, pass_type, mask_strategy=mask_strategy)
    StubModel.forward_partitioned = recording
    try:
        compose(make_batch(with_cens=False), mask_strategy_wt='marginal', mask_strategy_mt='independent')
    finally:
        StubModel.forward_partitioned = orig
    assert set(seen) == {('wt', 'marginal'), ('mt', 'independent')}, seen


def test_a_masked_wt_pass_is_not_packed_into_one_unit():
    """Unmasked, every single of a library is one backbone input, so the WT block runs as one unit; masked, each site is its own input."""
    from esm_msr import training
    from test_compose_censoring import make_hp
    cls = training.ESM3EpistasisLightningModule
    n = 40
    batch = {'wt_sequence_tokens': torch.tensor([WT_SEQ] * n), 'coords': torch.zeros(n, 6, 3, 3), 'structure_tokens': torch.zeros(n, 6)}

    class M:
        dedup_backbone = True
        _unique_rows = staticmethod(lambda seq, c, s: (torch.tensor([0]), torch.zeros(len(seq), dtype=torch.long)))
    for hp_kw, n_units in (({}, 1), ({'mask_strategy_wt': 'marginal'}, 3)):
        stub = types.SimpleNamespace(model=M(), hparams=make_hp(lambda_mt_colrank=0.0, **hp_kw))
        units = cls._plan_units(stub, batch, torch.arange(n), torch.arange(0), 16)
        assert len(units) == n_units, (hp_kw, len(units))


# ---------------- flags ----------------

def _parse(*extra):
    import sys
    from esm_msr import config
    argv = sys.argv
    try:
        sys.argv = ['training.py', '--raw_data_file', 'x.csv', '--af_model_folder', 'x', '--experiment_name', 'x', *extra]
        return config.parse_arguments()
    finally:
        sys.argv = argv


def test_the_defaults_change_nothing():
    a = _parse()
    assert a.mask_strategy_wt is None and a.mask_strategy_mt is None and a.val_pass_probe == 0
    assert pass_mask_strategy(vars(a), 'wt') is None and pass_mask_strategy(vars(a), 'mt') is None


def test_the_ladder_parses():
    a = _parse('--mask_strategy_wt', 'marginal', '--mask_strategy_mt', 'independent', '--val_pass_probe', '2')
    assert (pass_mask_strategy(vars(a), 'wt'), pass_mask_strategy(vars(a), 'mt'), a.val_pass_probe) == ('marginal', 'independent', 2)


def test_a_negative_probe_interval_is_refused():
    with pytest.raises(SystemExit):
        _parse('--val_pass_probe', '-1')


# ---------------- inference ----------------

def test_inference_reports_the_masks_a_checkpoint_trained_with(tmp_path):
    from esm_msr.inference import parse_hparams_to_lora_config
    p = tmp_path / 'hparams.yaml'
    p.write_text("adapter_mode: dual\nmask_strategy: null\nmask_strategy_wt: marginal\nmask_strategy_mt: independent\n")
    assert parse_hparams_to_lora_config(str(p))['pass_masks'] == {'wt': 'marginal', 'mt': 'independent'}
    p.write_text("adapter_mode: dual\nmask_strategy: marginal\n")                       # a checkpoint from before the per-pass flags
    assert parse_hparams_to_lora_config(str(p))['pass_masks'] == {'wt': 'marginal', 'mt': 'marginal'}


def test_inference_refuses_a_checkpoint_with_different_per_pass_masks():
    src = open('src/esm_msr/inference.py').read()
    assert "if trained['wt'] != trained['mt']:" in src and 'raise NotImplementedError' in src
