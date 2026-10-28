"""ThermoMPNN / ThermoMPNN-D: train on the ESM-MSR-matched tables, and run inference on every benchmark with resource tracking.

Only the data path differs from upstream (Kuhlman-Lab/ThermoMPNN-D at the commit in third_party.lock, unmodified):
OctItemsDataset subclasses MegaScaleDatasetv2 and replaces its __init__ (the raw-CSV read + split pickle) with our item
tables; __getitem__, _augment_double_mutants, _sort_dataset, the models, trainers, losses and configs are upstream's.

  * One "WT" per ESM-MSR library code. A mutant-background library ('2K52_L42A') gets data/pdbs/2K52_L42A.pdb: the WT
    AlphaFold backbone with the background residue renamed, so the wild-type sequence ThermoMPNN-D reads matches the
    sequence the measurements are stated against (ESM-MSR uses this same backbone + sequence).
  * mut_types=[single] -> single + native_cond items; [double, double-aug] -> regressable doubles + upstream's
    augmentation. Augmentation needs the Rosetta model of each single (MSR_ROSETTA_DIR/<wt>/pdb_models), which exists
    only for WT-backbone singles, so mutant-background singles are left out of the augmentation pool.

Two networks, either or both: the `single` network scores one mutation (a multi-mutant gets the sum of its singles, its
additive prediction) and the `epistatic` (siamese) network scores exactly two. Give the checkpoints you have:

  train:      python thermompnn_inference.py train --model {single,epistatic} --seed 1 --out DIR
  everything: python thermompnn_inference.py all --single_ckpt S --epistatic_ckpt E --out DIR
              (Tsuboyama val + test, the 12 external sets, the 7 DMS, Domainome, ProteinGym)
  one block:  python thermompnn_inference.py {test,external,dms,domainome,pgym} ...
  --single_ckpt only     every row gets its additive prediction (pred_combined == pred_single_additive)
  --epistatic_ckpt only  only rows with exactly two mutations are predicted
  --mutants {single,multi,both}   which rows are scored (default both); the others get NaN and cost nothing
  --batch_mode {auto,fixed}       dynamic or fixed batch size per protein (default auto; batching.py)

Per-dataset and per-protein wall time, CPU, RAM, VRAM and I/O are recorded for the inference calls by default
(inference_scripts/resource_tracker.py; --no_resource_tracking to skip). Run it in the fixed slot with
inference_scripts/sbatch_fixed_slot.sh.

Outputs under --out: predictions_{val,test}.csv, external_<set>.csv, dms_<assay>.csv, domainome.csv, ProteinGym's per-DMS
CSVs and summary.csv, and resources/<run_id>/.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                       # paths, dms_source
sys.path.insert(0, str(HERE.parent))                # resource_tracker, pgym_runner
import paths  # noqa: E402
from benchmark_rows import EXTERNAL, DMS, external_rows, dms_rows, domainome_rows  # noqa: E402
import resource_tracker  # noqa: E402
import batching  # noqa: E402

TMPNN = str(paths.THERMOMPNN_DIR)
sys.path.insert(0, TMPNN)
from omegaconf import OmegaConf  # noqa: E402
from thermompnn.datasets.v2_datasets import MegaScaleDatasetv2, tied_featurize_mut  # noqa: E402
from thermompnn.protein_mpnn_utils import parse_PDB  # noqa: E402
from thermompnn.train_thermompnn import parse_cfg  # noqa: E402

# The prepared item tables a baseline was trained on (prepare/prepare_oct06_data.py): MSR_BASELINE_DATA.
DATA = str(paths.BASELINE_DATA)
ROSETTA = str(paths.ROSETTA_DIR) + '/'
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


def _item_groups(ds):
    """The protein (library code) each item of a dataset sits on, in item order."""
    if isinstance(ds, list):
        return np.array([it['mutation'].pdb for it in ds])
    return ds.df['WT_name'].str.replace('.pdb', '', regex=False).to_numpy()


@torch.no_grad()
def predict(model_pl, ds, batching, siamese=False, tracker=None, groups=None, tag=None, tokens=16000):
    """Predicted ddG (ThermoMPNN convention, positive = destabilizing) for every item of ds, in order.

    Items are scored one protein at a time, in the batches `batching` decides (batching.py: dynamic against the GPU, or fixed;
    the fixed default is the historical rule, as many items as fit `tokens` residues, at most 256, halved for the siamese
    network). Each protein is one `protein` scope of `tracker` (inference only: featurizing, the forward pass and the copy
    back; building the items and parsing structures happen outside), with the three steps as phases; `tag` adds columns
    (model=, part=) to those rows. The model reads items independently (padding is masked), so a prediction does not depend on
    its batch-mates."""
    tracker = tracker or resource_tracker.NullTracker()    # callers always pass one; the tracker module is needed here
    model_pl.eval().cuda()
    dev = next(model_pl.parameters()).device
    lengths = _item_lengths(ds)
    groups = _item_groups(ds) if groups is None else np.asarray(groups)
    preds = np.full(len(lengths), np.nan, dtype=np.float64)
    for g in dict.fromkeys(groups.tolist()):
        idx = np.flatnonzero(groups == g)
        L = int(lengths[idx[0]])                              # one structure: every item of a protein has its length
        with tracker.scope('protein', str(g), n_rows=len(idx), seq_len=L, **(tag or {})) as sc:
            def chunk_fn(a, b, idx=idx, sc=sc):
                sub = idx[a:b]
                with sc.phase('featurize'):
                    batch = tied_featurize_mut([ds[j] for j in sub], side_chains=False)
                    X, S, mask, lengths_b, chain_M, chain_enc, res_idx, mut_pos, mut_wt, mut_mt, mut_ddG, atom_mask = batch
                with sc.phase('forward'):
                    args = [t.to(dev) if torch.is_tensor(t) else t for t in
                            (X, S, mask, chain_M, res_idx, chain_enc, mut_pos, mut_wt, mut_mt, mut_ddG, atom_mask)]
                    if siamese:
                        a_, b_ = model_pl(*args)
                        p = (a_ + b_) / 2.
                    else:
                        p, _ = model_pl(*args)
                with sc.phase('d2h'):
                    preds[sub] = p.float().reshape(-1).cpu().numpy()
                del args, batch, p
            info = batching.run(len(idx), chunk_fn, label=f'{g}', default_chunk=lambda: max(1, min(256, (tokens // (2 if siamese else 1)) // L)))
            sc.note(**info)
    assert not np.isnan(preds).any(), 'some items were never scored'
    return preds


def _n_mut(mut_types):
    return np.asarray(pd.Series(mut_types).astype(str).str.count(':') + 1)


def _limit(df, n):
    """--max_proteins: only the rows of the first n proteins (by `code`) of a frame; everything with n == 0."""
    if not n:
        return df
    keep = list(dict.fromkeys(df['code']))[:n]
    return df[df['code'].isin(keep)].reset_index(drop=True)


def _wanted(n_mut, mutants):
    """Which rows --mutants asks for."""
    n = np.asarray(n_mut)
    return {'single': n == 1, 'multi': n >= 2, 'both': np.ones(len(n), dtype=bool)}[mutants]


def _tracker(args, mode):
    """The resource tracker for this invocation (a no-op with --no_resource_tracking)."""
    return resource_tracker.Tracker.from_args(
        args, label='thermompnn', default_dir=os.path.join(args.out, 'resources'),
        meta={'model': 'ThermoMPNN-D', 'mode': mode, 'seed': args.seed, 'single_ckpt': os.path.basename(args.single_ckpt or ''),
              'epistatic_ckpt': os.path.basename(args.epistatic_ckpt or ''), 'dtype': 'fp32', 'mutants': args.mutants,
              **args.batching.meta()})


def load_models(args, tracker):
    """The networks whose checkpoints were given, loaded onto the GPU (a `stage` row each, not inference cost)."""
    from thermompnn.trainer.v2_trainer import TransferModelPLv2, TransferModelPLv2Siamese
    os.chdir(os.path.join(TMPNN, 'thermompnn'))
    models, cfgs = {}, {}
    for name, ckpt in (('single', args.single_ckpt), ('epistatic', args.epistatic_ckpt)):
        if not ckpt:
            continue
        cfg = load_cfg(name, args.seed)
        cls = TransferModelPLv2Siamese if cfg.model.aggregation == 'siamese' else TransferModelPLv2
        with tracker.stage(f'model_load_{name}'):
            models[name] = cls.load_from_checkpoint(ckpt, cfg=cfg, map_location='cpu')
            models[name].cuda()
        cfgs[name] = cfg
    if not models:
        sys.exit('give --single_ckpt and/or --epistatic_ckpt')
    return models, cfgs


def _combine(add, epi, n_mut, models):
    """ESM-MSR convention (positive = stabilizing) from the two networks' outputs (ThermoMPNN: positive = destabilizing).
    A double takes the epistatic prediction when that network ran, every other row its additive one."""
    pred_add = -add
    comb = np.where(n_mut == 2, -epi, -add) if 'epistatic' in models else -add
    if 'single' in models:
        comb = np.where(np.isnan(add), np.nan, comb)    # a row the structure could not be mapped for gets no prediction at all
    return pred_add, comb


def test(args, tracker, models, cfgs, split):
    """The prepared Tsuboyama item table of one split (val or test)."""
    os.chdir(os.path.join(TMPNN, 'thermompnn'))
    items = _limit(pd.read_csv(os.path.join(DATA, f'items_{split}.csv'), low_memory=False), args.max_proteins)
    items['row'] = np.arange(len(items))
    is_double = items.mut_type.str.contains(':').to_numpy()
    wanted = _wanted(_n_mut(items.mut_type), args.mutants)
    out = items.copy()
    out['single_pred'] = np.nan
    out['epistatic_pred'] = np.nan
    with tracker.scope('dataset', f'tsuboyama_{split}', n_rows=int(wanted.sum())):
        for model in ('single', 'epistatic'):
            if model not in models:
                continue
            cfg = cfgs[model]
            cfg.data.mut_types = ['single', 'double']
            ds = OctItemsDataset(cfg, split, items=items)
            if model == 'single':
                # the single network scores singles; doubles get its additive (sum of the two singles) prediction
                rows = items.loc[~is_double & wanted, 'row'].to_numpy()
                if len(rows):
                    ds.df = ds.df.loc[ds.df['row'].isin(rows)].reset_index(drop=True)
                    out.loc[ds.df['row'].values, 'single_pred'] = predict(
                        models[model], ds, tracker=tracker, tag={'model': 'single', 'part': 'singles'}, batching=args.batching)
                d = items.loc[is_double & wanted]
                add = np.zeros(len(d))
                for k in ((0, 1) if len(d) else ()):
                    part = d.copy()
                    part['mut_type'] = part['mut_type'].str.split(':').str[k]
                    ds.df = part.assign(WT_name=part.code + '.pdb', DIRECT=True).reset_index(drop=True)
                    add += predict(models[model], ds, tracker=tracker, tag={'model': 'single', 'part': f'additive_{k}'}, batching=args.batching)
                if len(d):
                    out.loc[d['row'].values, 'single_pred'] = add
            else:
                rows = items.loc[is_double & wanted, 'row'].to_numpy()
                if len(rows):
                    ds.df = ds.df.loc[ds.df.mut_type.str.contains(':') & ds.df['row'].isin(rows)].reset_index(drop=True)
                    out.loc[ds.df['row'].values, 'epistatic_pred'] = predict(
                        models[model], ds, siamese=True, tracker=tracker, tag={'model': 'epistatic', 'part': 'doubles'}, batching=args.batching)
    n_mut = np.where(is_double, 2, 1)
    out['pred_single_additive'], out['pred_combined'] = _combine(
        out['single_pred'].to_numpy(), out['epistatic_pred'].to_numpy(), n_mut, models)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f'predictions_{split}.csv')
    out.drop(columns=['row']).to_csv(path, index=False)
    print('wrote', path)


def domainome(args, tracker, models):
    """Score the domainome assay domain by domain.

    Kept separate from external() because that one materialises a deepcopy of the parsed structure for every mutation
    before predicting anything; at 536k rows over 522 domains that would not fit in memory. Here each domain is
    parsed once, predicted, and released. Only the single network is used: every row is a single mutant and
    ThermoMPNN-D takes exactly two mutations, so it makes no prediction on this benchmark at all.
    """
    from copy import deepcopy
    from thermompnn.datasets.dataset_utils import Mutation
    from thermompnn.protein_mpnn_utils import alt_parse_PDB
    if 'single' not in models or args.mutants == 'multi':
        print('domainome: skipped (every row is a single mutant; needs --single_ckpt and --mutants single|both)')
        return
    model_pl = models['single']
    os.makedirs(args.out, exist_ok=True)
    df = _limit(domainome_rows(), args.max_proteins)
    preds = np.full(len(df), np.nan)
    bad = 0
    with tracker.scope('dataset', 'domainome', n_rows=len(df)):
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
                preds[rows] = predict(model_pl, items, tracker=tracker, groups=[code] * len(items), tag={'model': 'single'}, batching=args.batching)
            if n % 50 == 0:
                print(f'  {n}/522 domains', flush=True)
    df['pred_single_additive'] = -preds
    df['pred_combined'] = df['pred_single_additive']   # identical: no row has two mutations
    df.to_csv(os.path.join(args.out, 'domainome.csv'), index=False)
    print(f'domainome: {len(df)} rows, {bad} not mappable onto the parsed structure, '
          f'{int(df.pred_single_additive.isna().sum())} without a prediction', flush=True)


def external(args, tracker, models, mode):
    """The 12 external sets (mode 'external') or the 7 DMS assays (mode 'dms')."""
    from copy import deepcopy
    from thermompnn.datasets.dataset_utils import Mutation
    from thermompnn.protein_mpnn_utils import alt_parse_PDB
    os.makedirs(args.out, exist_ok=True)
    names, rows_fn, prefix = (EXTERNAL, external_rows, 'external') if mode == 'external' else (DMS, dms_rows, 'dms')
    if mode == 'dms' and args.max_proteins:          # each assay is one protein: the smoke subset is the first N assays
        names = names[:args.max_proteins]
    for name in names:
        df = _limit(rows_fn(name), args.max_proteins)
        n_mut = _n_mut(df.mut_type)
        wanted = _wanted(n_mut, args.mutants)
        with tracker.scope('dataset', f'{prefix}_{name}', n_rows=int(wanted.sum())):
            pdbs, items = {}, {'single': [], 'epistatic': []}
            bad = 0
            for i, r in enumerate(df.itertuples()):
                if not wanted[i]:
                    continue
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
                # singles of every row (the additive prediction of a multi-mutant sums them); doubles also go to the epistatic network
                if 'single' in models:
                    for k, (w, _, t) in zip(idx, muts):
                        it = deepcopy(p); it['mutation'] = Mutation([k], [w], [t], 0.0, r.code); items['single'].append((i, it))
                if len(muts) == 2 and 'epistatic' in models:
                    it = deepcopy(p)
                    it['mutation'] = Mutation(idx, [m[0] for m in muts], [m[2] for m in muts], 0.0, r.code)
                    items['epistatic'].append((i, it))
            add = np.full(len(df), np.nan); epi = np.full(len(df), np.nan)
            codes = df['code'].to_numpy()
            if items['single']:
                rows, its = zip(*items['single'])
                s = predict(models['single'], list(its), tracker=tracker, groups=codes[list(rows)],
                            tag={'model': 'single'}, batching=args.batching)
                add_s = pd.Series(s).groupby(np.array(rows)).sum()
                add[add_s.index.values] = add_s.values
            if items['epistatic']:
                rows, its = zip(*items['epistatic'])
                epi[list(rows)] = predict(models['epistatic'], list(its), siamese=True, tracker=tracker,
                                          groups=codes[list(rows)], tag={'model': 'epistatic'}, batching=args.batching)
            df['pred_single_additive'], df['pred_combined'] = _combine(add, epi, n_mut, models)
            df.to_csv(os.path.join(args.out, f'{prefix}_{name}.csv'), index=False)
        n2 = int((n_mut == 2).sum()); nhi = int((n_mut > 2).sum())
        print(f'{name}: {len(df)} rows ({int((n_mut == 1).sum())} single, {n2} double, {nhi} order>2; scoring {args.mutants}), '
              f'{bad} not mappable onto the parsed structure. ThermoMPNN-D scores the {n2} doubles; '
              f'order>2 rows get only the additive prediction', flush=True)


def pgym(args, tracker, models):
    """ProteinGym (all DMS, pgym_inputs from preprocessing/pgym_preprocess.py) through the shared pgym_runner loop, so
    the outputs (per-DMS CSVs + summary.csv, resumable) match ESM-MSR's. Same readouts as external(): the single
    network's additive prediction for every row, the epistatic network on rows with exactly two mutations, additive for order>2.

    Each distinct single of a DMS is scored once (the single network is deterministic per structure and mutation, and
    the large multi-mutant DMS reuse the same singles across hundreds of thousands of rows); items are built and
    predicted in chunks so a DMS never holds more than `chunk` structure copies at once."""
    from copy import deepcopy
    from thermompnn.datasets.dataset_utils import Mutation
    from thermompnn.protein_mpnn_utils import alt_parse_PDB
    import pgym_runner

    def chunked_predict(model_pl, p, code, muts_list, siamese):
        out = np.full(len(muts_list), np.nan)
        for s in range(0, len(muts_list), args.chunk):
            items = []
            for muts in muts_list[s:s + args.chunk]:
                it = deepcopy(p)
                it['mutation'] = Mutation([k for k, _, _ in muts], [w for _, w, _ in muts], [t for _, _, t in muts], 0.0, code)
                items.append(it)
            out[s:s + len(items)] = predict(model_pl, items, siamese=siamese, tracker=tracker, groups=[code] * len(items),
                                            tag={'model': 'epistatic' if siamese else 'single'}, batching=args.batching)
        return out

    def score_fn(df):
        code = df['code'].iloc[0]
        p = alt_parse_PDB(df['pdb_file'].iloc[0], input_chain_list=[df['chain'].iloc[0]])[0]
        nongap = [k for k, a in enumerate(p['seq']) if a != '-']
        muts = [[(nongap[int(m[1:-1]) - 1], m[0], m[-1]) for m in s.split(':')] for s in df['mut_type_renumbered']]
        bad = sum(p['seq'][k] != w for row in muts for k, w, _ in row)
        assert bad == 0, f'{bad} mutations disagree with the parsed structure'
        n_mut = np.array([len(row) for row in muts])
        wanted = _wanted(n_mut, args.mutants)
        add = np.full(len(df), np.nan); epi = np.full(len(df), np.nan)
        if 'single' in models:
            singles = sorted({m for row, w in zip(muts, wanted) if w for m in row})
            s_pred = dict(zip(singles, chunked_predict(models['single'], p, code, [[m] for m in singles], False))) if singles else {}
            add = np.array([sum(s_pred[m] for m in row) if w else np.nan for row, w in zip(muts, wanted)])
        dbl = np.where((n_mut == 2) & wanted)[0]
        if 'epistatic' in models and len(dbl):
            epi[dbl] = chunked_predict(models['epistatic'], p, code, [muts[i] for i in dbl], True)
        torch.cuda.empty_cache()
        pred_add, comb = _combine(add, epi, n_mut, models)
        return pd.DataFrame({'pred_single_additive': pred_add, 'pred_combined': comb, 'n_mut': n_mut})

    pgym_runner.run_protein_gym(args.pgym_dir, args.out, score_fn, pred_col='pred_combined', dms=args.pgym_dms,
                                tag=f'ThermoMPNN single={os.path.basename(args.single_ckpt or "-")} '
                                    f'epistatic={os.path.basename(args.epistatic_ckpt or "-")} mutants={args.mutants}',
                                extra_cols=('pred_single_additive',), tracker=tracker)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=['train', 'test', 'external', 'dms', 'domainome', 'pgym', 'all'])
    ap.add_argument('--model', choices=list(CONFIGS), help='train: which network')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--out', required=True)
    ap.add_argument('--num_workers', type=int, default=None)
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--batch_fraction', type=float, default=None)
    ap.add_argument('--split', default='test', help='test: which prepared item table (test | val); `all` does both')
    ap.add_argument('--single_ckpt')
    ap.add_argument('--epistatic_ckpt')
    ap.add_argument('--max_proteins', type=int, default=0,
                    help='smoke test: only the first N proteins of every benchmark (0 = all); ProteinGym: use --pgym_dms')
    ap.add_argument('--mutants', choices=['single', 'multi', 'both'], default='both',
                    help='which rows to score (the others get NaN predictions and no inference)')
    ap.add_argument('--pgym_dir', default=str(paths.PGYM_INPUTS))
    ap.add_argument('--pgym_dms', default='all', help="comma-separated DMS ids, or 'all'")
    ap.add_argument('--chunk', type=int, default=20000, help='pgym: structure copies built per predict() call')
    batching.add_batch_args(ap)
    resource_tracker.add_tracking_args(ap)
    a = ap.parse_args()
    for k in ('out', 'single_ckpt', 'epistatic_ckpt', 'pgym_dir', 'resource_dir'):   # the loaders chdir into ThermoMPNN-D
        if getattr(a, k, None):
            setattr(a, k, os.path.abspath(getattr(a, k)))
    if a.mode == 'train':
        return train(a)
    a.batching = batching.Batching(a)
    tracker = _tracker(a, a.mode)
    try:
        models, cfgs = load_models(a, tracker)
        if a.mode in ('test', 'all'):
            for split in (['val', 'test'] if a.mode == 'all' else [a.split]):
                test(a, tracker, models, cfgs, split)
        if a.mode in ('external', 'all'):
            external(a, tracker, models, 'external')
        if a.mode in ('dms', 'all'):
            external(a, tracker, models, 'dms')
        if a.mode in ('domainome', 'all'):
            domainome(a, tracker, models)
        if a.mode in ('pgym', 'all'):
            pgym(a, tracker, models)
    finally:
        tracker.close()


if __name__ == '__main__':
    main()
