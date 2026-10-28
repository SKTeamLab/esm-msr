"""
Train / evaluate ThermoMPNN-D on the ESM-MSR-matched oct06 tables (outputs/baselines_oct06/data/items_*.csv).

Only the data path differs from upstream: OctItemsDataset subclasses MegaScaleDatasetv2 and replaces its __init__
(the raw-CSV read + split pickle) with our item tables; __getitem__, _augment_double_mutants, _sort_dataset, the
models, trainers, losses and configs are upstream's, untouched.

  * One "WT" per ESM-MSR library code. A mutant-background library ('2K52_L42A') gets data/pdbs/2K52_L42A.pdb: the WT
    AlphaFold backbone with the background residue renamed, so the wild-type sequence ThermoMPNN-D reads matches the
    sequence the measurements are stated against (ESM-MSR uses this same backbone + sequence).
  * mut_types=[single] -> single + native_cond items; [double, double-aug] -> regressable doubles + upstream's
    augmentation. Augmentation needs the Rosetta model of each single (rosetta_data/<wt>/pdb_models), which exists
    only for WT-backbone singles, so mutant-background singles are left out of the augmentation pool.

Usage:
  train:  python thermompnn_oct06.py train --model {single,epistatic} --seed 1 --out DIR
  test:   python thermompnn_oct06.py test --single_ckpt C1 --epistatic_ckpt C2 --out DIR
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dms_source  # noqa: E402

# the shared resource tracker lives in the ESM-MSR repo's inference_scripts (it is also what pgym_runner uses)
INFERENCE_SCRIPTS = os.environ.get('MSR_INFERENCE_SCRIPTS', '/home/sreeves/esm-msr-devel/inference_scripts')
sys.path.insert(0, INFERENCE_SCRIPTS)
try:
    import resource_tracker  # noqa: E402
except ImportError:      # train mode needs no tracker; the inference modes say so below
    resource_tracker = None

TMPNN = '/home/sreeves/software/ThermoMPNN-D'
sys.path.insert(0, TMPNN)
from omegaconf import OmegaConf  # noqa: E402
from thermompnn.datasets.v2_datasets import MegaScaleDatasetv2, tied_featurize_mut  # noqa: E402
from thermompnn.protein_mpnn_utils import parse_PDB  # noqa: E402
from thermompnn.train_thermompnn import parse_cfg  # noqa: E402

# Which prepared split to train/evaluate on. Set MSR_BASELINE_DATA to point at another split's
# prepare_oct06_data.py output (e.g. .../outputs/baselines_capped/data).
DATA = os.environ.get('MSR_BASELINE_DATA', '/home/sreeves/esm-msr-devel/outputs/baselines_oct06/data')
ROSETTA = '/home/sreeves/software/esm-msr/data/tsuboyama/FINAL_results/'
CONFIGS = {'single': 'train_single.yaml', 'epistatic': 'train_epistatic.yaml'}


class OctItemsDataset(MegaScaleDatasetv2):
    def __init__(self, cfg, split, items=None):
        self.cfg = cfg
        self.split = split
        self.side_chains = self.cfg.data.get('side_chains', False)
        items = pd.read_csv(os.path.join(DATA, f'items_{split}.csv'), low_memory=False) if items is None else items
        items = items.copy()
        items['WT_name'] = items['code'] + '.pdb'
        items['aa_seq'] = items['wt_seq']
        is_double = items['mut_type'].str.contains(':')

        mut_list = []
        if 'single' in cfg.data.mut_types:
            mut_list.append(items.loc[~is_double])
        if 'double' in cfg.data.mut_types:
            mut_list.append(items.loc[is_double])
        df = pd.concat(mut_list, axis=0).reset_index(drop=True)
        df['DIRECT'] = True
        df['wt_orig'] = df['mut_type'].str[0]

        self.wt_names = sorted(items['WT_name'].unique())
        self.pdb_data = {}
        for wt_name in self.wt_names:
            code = wt_name.removesuffix('.pdb')
            self.pdb_data[code] = parse_PDB(os.path.join(DATA, 'pdbs', f'{code}.pdb'), side_chains=self.side_chains)[0]
            assert self.pdb_data[code]['seq'] == items.loc[items.code == code, 'wt_seq'].iloc[0], code

        df_list = [df]
        if 'double-aug' in cfg.data.mut_types:
            tmp = items.loc[(items['kind'] == 'single')].reset_index(drop=True)
            has_model = tmp.apply(lambda r: os.path.isfile(os.path.join(
                ROSETTA, r['code'], 'pdb_models', f"A[{r['mut_type']}].pdb")), axis=1)
            print(f'[{split}] augmentation pool: {int(has_model.sum())} of {len(tmp)} WT-backbone singles have a Rosetta model')
            tmp = tmp.loc[has_model].reset_index(drop=True)
            double_aug = self._augment_double_mutants(tmp, c=1)
            print('Generated %s augmented double mutations' % str(double_aug.shape[0]))
            double_aug['DIRECT'] = False
            self.tmp = tmp
            df_list.append(double_aug)
        self.df = pd.concat(df_list, axis=0).sort_values(by='WT_name').reset_index(drop=True)
        print(f'[{split}] {len(self.df)} rows ({int(self.df.DIRECT.sum())} direct) over {len(self.wt_names)} libraries')
        self._sort_dataset()


def load_cfg(model, seed):
    cfg = OmegaConf.merge(
        OmegaConf.create({'platform': {'accel': 'gpu', 'thermompnn_dir': TMPNN},
                          'data_loc': {'rosetta_data': ROSETTA}}),
        OmegaConf.load(os.path.join(TMPNN, 'examples/configs', CONFIGS[model])))
    cfg.name = f'{model}_oct06_seed{seed}'
    cfg.data.seed = seed  # upstream option: seeds the double-aug pairing
    return parse_cfg(cfg)


def train(args):
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from torch.utils.data import DataLoader
    from thermompnn.trainer.v2_trainer import TransferModelPLv2, TransferModelPLv2Siamese

    pl.seed_everything(args.seed, workers=True)
    cfg = load_cfg(args.model, args.seed)
    if args.num_workers is not None:
        cfg.training.num_workers = args.num_workers
    if args.epochs is not None:  # smoke tests only; the real runs use the config's 100
        cfg.training.epochs = args.epochs
    if args.batch_fraction is not None:
        cfg.training.batch_fraction = args.batch_fraction
    print('Configuration:\n', OmegaConf.to_yaml(cfg))
    # ThermoMPNN loads the vanilla ProteinMPNN weights relative to its repo dir
    os.chdir(os.path.join(TMPNN, 'thermompnn'))

    train_ds, val_ds = OctItemsDataset(cfg, 'train'), OctItemsDataset(cfg, 'val')
    collate = lambda b: tied_featurize_mut(b, side_chains=cfg.data.side_chains)  # noqa: E731
    train_loader = DataLoader(train_ds, collate_fn=collate, shuffle=cfg.training.shuffle,
                              num_workers=cfg.training.num_workers, batch_size=cfg.training.batch_size)
    val_loader = DataLoader(val_ds, collate_fn=collate, shuffle=False,
                            num_workers=cfg.training.num_workers, batch_size=cfg.training.batch_size)
    model_pl = TransferModelPLv2Siamese(cfg) if cfg.model.aggregation == 'siamese' else TransferModelPLv2(cfg)
    os.makedirs(args.out, exist_ok=True)
    ckpt = ModelCheckpoint(monitor='val_ddG_spearman', mode='max', dirpath=args.out,
                           filename=cfg.name + '_{epoch:02d}_{val_ddG_spearman:.02}', save_last=True)
    trainer = pl.Trainer(callbacks=[ckpt], logger=pl.loggers.CSVLogger(args.out, name='csv'), log_every_n_steps=100,
                         max_epochs=cfg.training.epochs, accelerator=cfg.platform.accel, devices=1,
                         limit_train_batches=cfg.training.batch_fraction)
    trainer.fit(model_pl, train_loader, val_loader)
    print('best checkpoint:', ckpt.best_model_path, ckpt.best_model_score)
    with open(os.path.join(args.out, 'best_checkpoint.txt'), 'w') as f:
        f.write(f'{ckpt.best_model_path}\n')


def _item_lengths(ds):
    """Length of the protein each item sits on, without materialising the items."""
    if isinstance(ds, list):
        return np.array([len(it['seq']) for it in ds])
    seqlen = {code: len(p['seq']) for code, p in ds.pdb_data.items()}
    return ds.df['WT_name'].str.replace('.pdb', '', regex=False).map(seqlen).to_numpy()


def _batches(lengths, batch_size, tokens):
    """Indices grouped so that (rows x padded length) stays under `tokens`: the external benchmark proteins are
    several times longer than the ~70-residue Tsuboyama libraries, and a fixed 256 row batch OOMs on them.
    Sorted by length so padding is minimal and each batch is near-homogeneous."""
    out, cur = [], []
    for j in np.argsort(lengths, kind='stable'):
        trial_max = max([lengths[j]] + [lengths[k] for k in cur])
        if cur and ((len(cur) + 1) * trial_max > tokens or len(cur) >= batch_size):
            out.append(cur)
            cur = []
        cur.append(int(j))
    return out + ([cur] if cur else [])


def _item_groups(ds):
    """The protein (library code) each item of a dataset sits on, in item order."""
    if isinstance(ds, list):
        return np.array([it['mutation'].pdb for it in ds])
    return ds.df['WT_name'].str.replace('.pdb', '', regex=False).to_numpy()


def _grouped_batches(lengths, groups, batch_size, tokens):
    """_batches within each group, so no batch spans two proteins. That makes a batch's cost belong to one protein
    (the per-protein resource rows need that) and is also how a user scores one structure at a time. The model reads
    every item independently (padding is masked), so a prediction does not depend on its batch-mates."""
    out = []
    groups = np.asarray(groups)
    for g in dict.fromkeys(groups.tolist()):
        gi = np.flatnonzero(groups == g)
        out.extend((g, [int(gi[k]) for k in b]) for b in _batches(lengths[gi], batch_size, tokens))
    return out


@torch.no_grad()
def predict(model_pl, ds, batch_size=256, siamese=False, tokens=16000, tracker=None, groups=None, tag=None,
            mixed_protein_batches=False):
    """Predicted ddG (ThermoMPNN convention, positive = destabilizing) for every item of ds, in order.

    Batches never span proteins unless mixed_protein_batches (the original behaviour: all items of a dataset sorted
    by length together). Each protein's batches are one `protein` scope of `tracker` (inference only: featurizing,
    the forward pass and the copy back; building the items and parsing structures happen outside), with the three
    steps as phases; `tag` adds columns (model=, part=) to those rows."""
    tracker = tracker or resource_tracker.NullTracker()    # callers always pass one; the tracker module is needed here
    model_pl.eval().cuda()
    lengths = _item_lengths(ds)
    groups = _item_groups(ds) if groups is None else np.asarray(groups)
    preds = np.full(len(lengths), np.nan, dtype=np.float64)
    if mixed_protein_batches:
        plan = [(None, idx) for idx in _batches(lengths, batch_size, tokens // (2 if siamese else 1))]
    else:
        plan = _grouped_batches(lengths, groups, batch_size, tokens // (2 if siamese else 1))
    by_group = {}
    for g, idx in plan:
        by_group.setdefault(g, []).append(idx)
    for g, batches in by_group.items():
        n_rows = sum(len(b) for b in batches)
        name = str(g) if g is not None else '(mixed)'
        with tracker.scope('protein', name, n_rows=n_rows, seq_len=int(lengths[batches[0][0]]), n_batches=len(batches),
                           batch_size=batch_size, tokens=tokens, **(tag or {})) as sc:
            for idx in batches:
                with sc.phase('featurize'):
                    batch = tied_featurize_mut([ds[j] for j in idx], side_chains=False)
                    X, S, mask, lengths_b, chain_M, chain_enc, res_idx, mut_pos, mut_wt, mut_mt, mut_ddG, atom_mask = batch
                with sc.phase('forward'):
                    args = [t.cuda() if torch.is_tensor(t) else t for t in
                            (X, S, mask, chain_M, res_idx, chain_enc, mut_pos, mut_wt, mut_mt, mut_ddG, atom_mask)]
                    if siamese:
                        a, b = model_pl(*args)
                        p = (a + b) / 2.
                    else:
                        p, _ = model_pl(*args)
                with sc.phase('d2h'):
                    preds[idx] = p.float().reshape(-1).cpu().numpy()
                del args, batch, p
    assert not np.isnan(preds).any(), 'some items were never scored'
    return preds


def test(args):
    from thermompnn.trainer.v2_trainer import TransferModelPLv2, TransferModelPLv2Siamese
    os.chdir(os.path.join(TMPNN, 'thermompnn'))
    tracker = _tracker(args, 'test')
    items = pd.read_csv(os.path.join(DATA, f'items_{args.split}.csv'), low_memory=False)
    _ds_scope = tracker.scope('dataset', f'tsuboyama_{args.split}', n_rows=len(items))
    _ds_scope.__enter__()
    items['row'] = np.arange(len(items))
    out = items.copy()
    out['single_pred'] = np.nan
    out['epistatic_pred'] = np.nan
    for model, ckpt in [('single', args.single_ckpt), ('epistatic', args.epistatic_ckpt)]:
        cfg = load_cfg(model, args.seed)
        cfg.data.mut_types = ['single', 'double']
        cls = TransferModelPLv2Siamese if cfg.model.aggregation == 'siamese' else TransferModelPLv2
        with tracker.stage(f'model_load_{model}'):
            model_pl = cls.load_from_checkpoint(ckpt, cfg=cfg, map_location='cpu')
            model_pl.cuda()
        ds = OctItemsDataset(cfg, args.split, items=items)
        # the single model scores singles; doubles get its additive (sum of the two singles) prediction
        if model == 'single':
            s = ds.df.loc[~ds.df.mut_type.str.contains(':')].copy()
            ds.df = s.reset_index(drop=True)
            out.loc[ds.df['row'].values, 'single_pred'] = predict(
                model_pl, ds, tracker=tracker, tag={'model': 'single', 'part': 'singles'}, mixed_protein_batches=args.mixed_protein_batches)
            d = items.loc[items.mut_type.str.contains(':')]
            add = np.zeros(len(d))
            for k in (0, 1):
                part = d.copy()
                part['mut_type'] = part['mut_type'].str.split(':').str[k]
                ds.df = part.assign(WT_name=part.code + '.pdb', DIRECT=True).reset_index(drop=True)
                add += predict(model_pl, ds, tracker=tracker, tag={'model': 'single', 'part': f'additive_{k}'},
                               mixed_protein_batches=args.mixed_protein_batches)
            out.loc[d['row'].values, 'single_pred'] = add
        else:
            d = ds.df.loc[ds.df.mut_type.str.contains(':')].copy()
            ds.df = d.reset_index(drop=True)
            out.loc[ds.df['row'].values, 'epistatic_pred'] = predict(
                model_pl, ds, siamese=True, tracker=tracker, tag={'model': 'epistatic', 'part': 'doubles'},
                mixed_protein_batches=args.mixed_protein_batches)
    # ESM-MSR convention (ddG_ML: positive = stabilizing)
    out['pred_single_additive'] = -out['single_pred']
    out['pred_combined'] = np.where(out.mut_type.str.contains(':'), -out['epistatic_pred'], -out['single_pred'])
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f'predictions_{args.split}.csv')
    out.drop(columns=['row']).to_csv(path, index=False)
    print('wrote', path)
    _ds_scope.__exit__(None, None, None)
    tracker.close()


def _tracker(args, mode):
    """The resource tracker for this invocation (a no-op with --no_resource_tracking)."""
    if resource_tracker is None:
        sys.exit(f'resource_tracker.py not found in {INFERENCE_SCRIPTS} (set MSR_INFERENCE_SCRIPTS to the repo\'s inference_scripts)')
    return resource_tracker.Tracker.from_args(
        args, label='thermompnn', default_dir=os.path.join(args.out, 'resources'),
        meta={'model': 'ThermoMPNN-D', 'mode': mode, 'seed': args.seed, 'single_ckpt': os.path.basename(args.single_ckpt or ''),
              'epistatic_ckpt': os.path.basename(args.epistatic_ckpt or ''), 'dtype': 'fp32',
              'mixed_protein_batches': bool(args.mixed_protein_batches)})


EXTERNAL = ['s571', 's783', 's2648', 's8754', 's669', 's461', 'ssym', 'q3421', 'k3822', 'k2369', 'ptmul', 'ptmuld']
SEQPOS_SETS = {'s669', 's461', 'ssym', 'q3421', 'k3822', 'k2369', 's571', 's783', 's2648', 's8754'}


def external_rows(name):
    """The rows and mutation strings (structure-sequence numbering) esm_msr_testing.py scores for one external set."""
    df = pd.read_csv(os.path.join('/home/sreeves/esm-msr-devel/data/preprocessed', f'{name}_mapped.csv'))
    if name in SEQPOS_SETS:
        df['mut_type'] = df['wild_type'] + df['seq_pos'].astype(int).astype(str) + df['mutation']
        if 'dTm' in df.columns:
            df['ddG'] = df['dTm']
    else:
        df['mut_type'] = df['mut_info_seq_pos'].map(lambda s: ':'.join(sorted(s.split(':'), key=lambda m: int(m[1:-1]))))
    df['id'] = df['code'] + df['chain'] + '_' + df['mut_type']
    return df[['id', 'code', 'chain', 'pdb_file', 'mut_type', 'ddG']]


DMS = ['DLG4_HUMAN_Faure_2021_abundance_domain', 'DLG4_HUMAN_Faure_2021_binding_domain',
       'GRB2_HUMAN_Faure_2021_abundance_domain', 'GRB2_HUMAN_Faure_2021_binding_domain',
       'MYO_HUMAN_Kung_2025_display', 'ESTA_BACSU_Nutschel_2020_dTm', 'GB1_Wu_2016_binding_domain']


def dms_rows(prot):
    """The rows esm_msr_testing.py's DMS block scores for one assay, keyed and scored the same way (id = code_mut_info,
    deduplicated, target ddG_ML). Audited: for all seven assays every mutation's position and wild-type residue match
    the parsed chain exactly, and the chain sequence equals wt_seq wherever the table carries one, so the positions are
    already in structure numbering and need no renumbering."""
    # dms_source.load: the complete tables, with the completeness guard (the old software/esm-msr copy of GB1 was
    # missing V39A/V39C/V39D, which silently cost 152 of its 2,091 scorable doubles).
    df = dms_source.load(prot)
    df['mut_type'] = df['mut_info'].map(lambda s: ':'.join(sorted(str(s).split(':'), key=lambda m: int(m[1:-1]))))
    df['id'] = df['code'] + '_' + df['mut_info']
    df = df[~df['id'].duplicated(keep='first')]
    df['ddG'] = df['ddG_ML']
    return df[['id', 'code', 'chain', 'pdb_file', 'mut_type', 'ddG']].reset_index(drop=True)


def domainome_rows():
    """ESM-MSR's domainome rows (domainome_mapped_2026.csv, code = domain_ID with '/'->'_', target scaled_fitness),
    keyed and deduplicated as esm_msr_testing.py's domainome block does. All 536k rows are single mutants; audited:
    every row's position and wild-type residue match its spurs_structures chain A."""
    df = pd.read_csv('/home/sreeves/software/esm-msr/data/domainome1/domainome_mapped_2026.csv',
                     usecols=['domain_ID', 'mut_type', 'scaled_fitness', 'pdb_file', 'position', 'wild_type',
                              'mutation'], low_memory=False)
    df['code'] = df['domain_ID'].str.replace('/', '_', regex=False)
    df = df.dropna(subset=['pdb_file', 'position'])
    df['pdb_file'] = df['pdb_file'].str.replace('/home/sareeves', '/home/sreeves', regex=False)
    df['chain'] = 'A'
    df['ddG'] = df['scaled_fitness']
    df['id'] = df['code'] + '_' + df['mut_type']
    df = df[~df['id'].duplicated(keep='first')]
    return df[['id', 'code', 'chain', 'pdb_file', 'mut_type', 'ddG']].reset_index(drop=True)


def domainome(args):
    """Score the domainome assay domain by domain.

    Kept separate from external() because that one materialises a deepcopy of the parsed structure for every mutation
    before predicting anything; at 536k rows over 522 domains that would not fit in memory. Here each domain is
    parsed once, predicted, and released. Only the single model is used: every row is a single mutant and
    ThermoMPNN-D takes exactly two mutations, so it makes no prediction on this benchmark at all.
    """
    from copy import deepcopy
    from thermompnn.datasets.dataset_utils import Mutation
    from thermompnn.protein_mpnn_utils import alt_parse_PDB
    from thermompnn.trainer.v2_trainer import TransferModelPLv2
    os.chdir(os.path.join(TMPNN, 'thermompnn'))
    os.makedirs(args.out, exist_ok=True)
    tracker = _tracker(args, 'domainome')
    cfg = load_cfg('single', args.seed)
    with tracker.stage('model_load_single'):
        model_pl = TransferModelPLv2.load_from_checkpoint(args.single_ckpt, cfg=cfg, map_location='cpu')
        model_pl.cuda()
    df = domainome_rows()
    preds = np.full(len(df), np.nan)
    bad = 0
    _ds_scope = tracker.scope('dataset', 'domainome', n_rows=len(df))
    _ds_scope.__enter__()
    for n, (code, g) in enumerate(df.groupby('code', sort=False), 1):
        p = alt_parse_PDB(g['pdb_file'].iloc[0], input_chain_list=['A'])[0]
        nongap = [k for k, a in enumerate(p['seq']) if a != '-']
        rows, items = [], []
        for i, mt in zip(g.index, g.mut_type):
            w, pos, t = mt[0], int(mt[1:-1]), mt[-1]
            k = nongap[pos - 1] if pos - 1 < len(nongap) else None
            if k is None or p['seq'][k] != w:
                bad += 1
                continue
            it = deepcopy(p); it['mutation'] = Mutation([k], [w], [t], 0.0, code)
            rows.append(i); items.append(it)
        if items:
            preds[rows] = predict(model_pl, items, tracker=tracker, groups=[code] * len(items), tag={'model': 'single'},
                                  mixed_protein_batches=args.mixed_protein_batches)
        if n % 50 == 0:
            print(f'  {n}/522 domains', flush=True)
    _ds_scope.__exit__(None, None, None)
    tracker.close()
    df['pred_single_additive'] = -preds
    df['pred_combined'] = df['pred_single_additive']   # identical: no row has two mutations
    df.to_csv(os.path.join(args.out, 'domainome.csv'), index=False)
    print(f'domainome: {len(df)} rows, {bad} not mappable onto the parsed structure, '
          f'{int(df.pred_single_additive.isna().sum())} without a prediction', flush=True)


def external(args):
    from copy import deepcopy
    from thermompnn.datasets.dataset_utils import Mutation
    from thermompnn.protein_mpnn_utils import alt_parse_PDB
    from thermompnn.trainer.v2_trainer import TransferModelPLv2, TransferModelPLv2Siamese
    os.chdir(os.path.join(TMPNN, 'thermompnn'))
    os.makedirs(args.out, exist_ok=True)
    tracker = _tracker(args, args.mode)
    models = {}
    for model, ckpt in [('single', args.single_ckpt), ('epistatic', args.epistatic_ckpt)]:
        cfg = load_cfg(model, args.seed)
        cls = TransferModelPLv2Siamese if cfg.model.aggregation == 'siamese' else TransferModelPLv2
        with tracker.stage(f'model_load_{model}'):
            models[model] = cls.load_from_checkpoint(ckpt, cfg=cfg, map_location='cpu')
            models[model].cuda()
    names, rows_fn, prefix = (EXTERNAL, external_rows, 'external') if args.mode == 'external' else (DMS, dms_rows, 'dms')
    for name in names:
        df = rows_fn(name)
        _ds_scope = tracker.scope('dataset', f'{prefix}_{name}', n_rows=len(df))
        _ds_scope.__enter__()
        pdbs, items = {}, {'single': [], 'epistatic': []}
        bad = 0
        for i, r in enumerate(df.itertuples()):
            key = (r.pdb_file, r.chain)
            if key not in pdbs:
                p = alt_parse_PDB(r.pdb_file, input_chain_list=[r.chain])[0]
                pdbs[key] = (p, [k for k, a in enumerate(p['seq']) if a != '-'])
            p, nongap = pdbs[key]
            muts = [(m[0], int(m[1:-1]), m[-1]) for m in r.mut_type.split(':')]
            idx = [nongap[pos - 1] if pos - 1 < len(nongap) else None for _, pos, _ in muts]
            if any(k is None or p['seq'][k] != w for k, (w, _, _) in zip(idx, muts)):
                bad += 1
                continue
            # singles of every row (the additive prediction of a multi-mutant sums them); doubles also go to the epistatic model
            for k, (w, _, t) in zip(idx, muts):
                it = deepcopy(p); it['mutation'] = Mutation([k], [w], [t], 0.0, r.code); items['single'].append((i, it))
            if len(muts) == 2:
                it = deepcopy(p)
                it['mutation'] = Mutation(idx, [m[0] for m in muts], [m[2] for m in muts], 0.0, r.code)
                items['epistatic'].append((i, it))
        add = np.full(len(df), np.nan); epi = np.full(len(df), np.nan)
        if items['single']:
            rows, its = zip(*items['single'])
            s = predict(models['single'], list(its), tracker=tracker, groups=df['code'].to_numpy()[list(rows)],
                        tag={'model': 'single'}, mixed_protein_batches=args.mixed_protein_batches)
            add_s = pd.Series(s).groupby(np.array(rows)).sum()
            add[add_s.index.values] = add_s.values
        if items['epistatic']:
            rows, its = zip(*items['epistatic'])
            epi[list(rows)] = predict(models['epistatic'], list(its), siamese=True, tracker=tracker,
                                      groups=df['code'].to_numpy()[list(rows)], tag={'model': 'epistatic'},
                                      mixed_protein_batches=args.mixed_protein_batches)
        n_mut = df.mut_type.str.count(':') + 1
        df['pred_single_additive'] = -add
        df['pred_combined'] = np.where(n_mut == 2, -epi, -add)
        df.loc[df.pred_single_additive.isna(), 'pred_combined'] = np.nan
        df.to_csv(os.path.join(args.out, f'{prefix}_{name}.csv'), index=False)
        _ds_scope.__exit__(None, None, None)
        n2 = int((n_mut == 2).sum()); nhi = int((n_mut > 2).sum())
        print(f'{name}: {len(df)} rows ({int((n_mut == 1).sum())} single, {n2} double, {nhi} order>2), '
              f'{bad} not mappable onto the parsed structure. ThermoMPNN-D scores the {n2} doubles; '
              f'order>2 rows get only the additive prediction', flush=True)
    tracker.close()


def pgym(args):
    """ProteinGym (all DMS, pgym_inputs from preprocessing/pgym_preprocess.py) through the shared pgym_runner loop, so
    the outputs (per-DMS CSVs + summary.csv, resumable) match ESM-MSR's. Same readouts as external(): the single
    model's additive prediction for every row, ThermoMPNN-D on rows with exactly two mutations, additive for order>2.

    Each distinct single of a DMS is scored once (the single model is deterministic per structure and mutation, and
    the large multi-mutant DMS reuse the same singles across hundreds of thousands of rows); items are built and
    predicted in chunks so a DMS never holds more than `chunk` structure copies at once."""
    from copy import deepcopy
    from thermompnn.datasets.dataset_utils import Mutation
    from thermompnn.protein_mpnn_utils import alt_parse_PDB
    from thermompnn.trainer.v2_trainer import TransferModelPLv2, TransferModelPLv2Siamese
    import pgym_runner
    os.chdir(os.path.join(TMPNN, 'thermompnn'))
    os.makedirs(args.out, exist_ok=True)
    tracker = _tracker(args, 'pgym')
    models = {}
    for model, ckpt in [('single', args.single_ckpt), ('epistatic', args.epistatic_ckpt)]:
        cfg = load_cfg(model, args.seed)
        cls = TransferModelPLv2Siamese if cfg.model.aggregation == 'siamese' else TransferModelPLv2
        with tracker.stage(f'model_load_{model}'):
            models[model] = cls.load_from_checkpoint(ckpt, cfg=cfg, map_location='cpu')
            models[model].cuda()

    def chunked_predict(model_pl, p, code, muts_list, siamese):
        out = np.full(len(muts_list), np.nan)
        for s in range(0, len(muts_list), args.chunk):
            items = []
            for muts in muts_list[s:s + args.chunk]:
                it = deepcopy(p)
                it['mutation'] = Mutation([k for k, _, _ in muts], [w for _, w, _ in muts], [t for _, _, t in muts], 0.0, code)
                items.append(it)
            out[s:s + len(items)] = predict(model_pl, items, siamese=siamese, tracker=tracker, groups=[code] * len(items),
                                            tag={'model': 'epistatic' if siamese else 'single'},
                                            mixed_protein_batches=args.mixed_protein_batches)
        return out

    def score_fn(df):
        code = df['code'].iloc[0]
        p = alt_parse_PDB(df['pdb_file'].iloc[0], input_chain_list=[df['chain'].iloc[0]])[0]
        nongap = [k for k, a in enumerate(p['seq']) if a != '-']
        muts = [[(nongap[int(m[1:-1]) - 1], m[0], m[-1]) for m in s.split(':')] for s in df['mut_type_renumbered']]
        bad = sum(p['seq'][k] != w for row in muts for k, w, _ in row)
        assert bad == 0, f'{bad} mutations disagree with the parsed structure'
        singles = sorted({m for row in muts for m in row})
        s_pred = dict(zip(singles, chunked_predict(models['single'], p, code, [[m] for m in singles], False)))
        add = np.array([sum(s_pred[m] for m in row) for row in muts])
        n_mut = np.array([len(row) for row in muts])
        epi = np.full(len(df), np.nan)
        dbl = np.where(n_mut == 2)[0]
        if len(dbl):
            epi[dbl] = chunked_predict(models['epistatic'], p, code, [muts[i] for i in dbl], True)
        torch.cuda.empty_cache()
        # ESM-MSR convention (positive = stabilizing)
        return pd.DataFrame({'pred_single_additive': -add, 'pred_combined': np.where(n_mut == 2, -epi, -add),
                             'n_mut': n_mut})

    pgym_runner.run_protein_gym(args.pgym_dir, args.out, score_fn, pred_col='pred_combined', dms=args.pgym_dms,
                                tag=f'ThermoMPNN single={os.path.basename(args.single_ckpt)} '
                                    f'epistatic={os.path.basename(args.epistatic_ckpt)}',
                                extra_cols=('pred_single_additive',), tracker=tracker)
    tracker.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['train', 'test', 'external', 'dms', 'domainome', 'pgym'])
    ap.add_argument('--model', choices=list(CONFIGS))
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--out', required=True)
    ap.add_argument('--num_workers', type=int, default=None)
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--batch_fraction', type=float, default=None)
    ap.add_argument('--split', default='test')
    ap.add_argument('--single_ckpt')
    ap.add_argument('--epistatic_ckpt')
    ap.add_argument('--pgym_dir', default='/home/sreeves/esm-msr-devel/pgym_inputs')
    ap.add_argument('--pgym_dms', default='all', help="comma-separated DMS ids, or 'all'")
    ap.add_argument('--chunk', type=int, default=20000, help='pgym: structure copies built per predict() call')
    ap.add_argument('--mixed_protein_batches', action='store_true',
                    help='Batch items of different proteins together, sorted by length (the behaviour before per-protein '
                         'resource tracking). Default: batches never span proteins.')
    if resource_tracker is not None:
        resource_tracker.add_tracking_args(ap)
    else:
        ap.add_argument('--no_resource_tracking', action='store_true')
    a = ap.parse_args()
    {'train': train, 'test': test, 'external': external, 'dms': external,
     'domainome': domainome, 'pgym': pgym}[a.mode](a)
