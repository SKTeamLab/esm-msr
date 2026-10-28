"""SPURS (luo-group/SPURS at the commit in third_party.lock, unmodified): inference on every benchmark, with resource tracking.

SPURS has two models: SPURS (one mutation per structure pass; a multi-mutant gets the sum of its singles) and SPURS-multi
(any number of mutations in one pass; also scored additively from its own single predictions). Four readouts come out,
all positive = stabilizing: SPURS_single (single rows), SPURS_single_additive, SPURS_multi, SPURS_multi_additive.

  everything: python spurs_inference.py --out DIR            (Tsuboyama val + test, 12 external sets, 7 DMS, Domainome, ProteinGym)
  --models {single,multi,both}    which model(s) to load and run (default both)
  --mutants {single,multi,both}   which rows to score (default both); the others stay NaN and cost nothing
  --checkpoint_single / --checkpoint_multi  'hf' (published, default) or a file under SPURS/checkpoints

Rows are the shared benchmark rows (benchmark_rows.py), the same every model scores. Per-dataset and per-protein wall time,
CPU, RAM, VRAM and I/O of the inference calls (not of parsing the structure) are recorded by default
(inference_scripts/resource_tracker.py; --no_resource_tracking to skip).
"""
import sys
import pkg_resources
from unittest.mock import MagicMock
import importlib.machinery

_HERE = __import__('pathlib').Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))                 # paths, benchmark_rows
sys.path.insert(0, str(_HERE.parent))          # resource_tracker, pgym_runner
sys.path.append(str(_HERE.parents[1] / 'src'))

# 1. Patch pkg_resources for the version check
mock_dist = MagicMock()
mock_dist.version = '0.11.0'
mock_dist.project_name = 'torchtext'
pkg_resources.get_distribution = MagicMock(return_value=mock_dist)

# 2. Create a recursive mock for torchtext and all sub-modules
def recursive_mock(*args, **kwargs):
    return MagicMock()

mock_torchtext = MagicMock()
mock_torchtext.data = MagicMock()
mock_torchtext.utils = MagicMock()
mock_torchtext.functional = MagicMock()
mock_torchtext.vocab = MagicMock()

dummy_spec = importlib.machinery.ModuleSpec(name="torchtext", loader=None, origin="mocked")
for m in [mock_torchtext, mock_torchtext.data, mock_torchtext.utils, mock_torchtext.vocab]:
    m.__spec__ = dummy_spec
    m.__path__ = []

# 3. Inject the main modules and the specific failing sub-module into sys.modules
sys.modules["torchtext"] = mock_torchtext
sys.modules["torchtext.data"] = mock_torchtext.data
sys.modules["torchtext.data.functional"] = mock_torchtext.data.functional
sys.modules["torchtext.utils"] = mock_torchtext.utils
sys.modules["torchtext.vocab"] = mock_torchtext.vocab

import importlib.machinery
from contextlib import contextmanager

@contextmanager
def mock_esm_context():
    """
    Temporarily hijacks sys.modules to provide mock objects for 'esm'
    and its submodules. Restores original state upon exit.
    """
    mocked_keys = ["esm", "esm.pretrained", "esm.models.esm3", "esm.utils.constants", "esm.utils.structure.protein_chain"]
    original_modules = {}
    
    for key in mocked_keys:
        if key in sys.modules:
            original_modules[key] = sys.modules[key]

    mock_esm = MagicMock()
    mock_esm.pretrained = MagicMock()
    mock_esm.models.esm3 = MagicMock()
    mock_esm.utils.constants = MagicMock()
    mock_esm.utils.structure.protein_chain = MagicMock()

    dummy_spec = importlib.machinery.ModuleSpec(name="esm", loader=None, origin="mocked")
    for m in [mock_esm, mock_esm.pretrained, mock_esm.models.esm3, mock_esm.utils.constants, mock_esm.utils.structure.protein_chain]:
        m.__spec__ = dummy_spec
        m.__path__ = []

    sys.modules["esm"] = mock_esm
    sys.modules["esm.pretrained"] = mock_esm.pretrained
    sys.modules["esm.models.esm3"] = mock_esm.models.esm3
    sys.modules["esm.utils.constants"] = mock_esm.utils.constants
    sys.modules["esm.utils.structure.protein_chain"] = mock_esm.utils.structure.protein_chain

    try:
        yield
    finally:
        for key in mocked_keys:
            if key in sys.modules:
                del sys.modules[key]
            if key in original_modules:
                sys.modules[key] = original_modules[key]

with mock_esm_context():
    from esm_msr.utils import parse_multimutant_column, sort_mutations_by_position, sum_individual_mutation_scores
    from esm_msr.preprocess_megascale import MegaScaleDatasetPreprocessor

import pandas as pd
import os
import torch
from tqdm import tqdm
import argparse
import time
from pathlib import Path
import warnings
import numpy as np

from omegaconf import OmegaConf
import collections
import typing
import omegaconf

# torch>=2.6 loads checkpoints weights-only by default; the SPURS checkpoints also pickle their omegaconf
# hyperparameters, so allow exactly those container types (and nothing executable)
torch.serialization.add_safe_globals([
    omegaconf.listconfig.ListConfig, omegaconf.dictconfig.DictConfig, omegaconf.base.ContainerMetadata,
    omegaconf.base.Metadata, omegaconf.nodes.AnyNode, typing.Any, list, dict, int, collections.defaultdict])

# SPURS Imports
from spurs.utils import seed_everything
from spurs.models.stability.spurs import SPURS
from spurs.models.stability.spurs_multi import SPURSMulti
from spurs.datamodules.datasets.data_utils import Alphabet
from spurs.inference import parse_pdb, get_SPURS_from_hub, get_SPURS_multi_from_hub, parse_pdb_for_mutation

warnings.filterwarnings('ignore') # Consider removing this to catch legitimate deprecations

import paths  # noqa: E402
import resource_tracker  # noqa: E402
import batching  # noqa: E402
import calibration  # noqa: E402
from batching import merge_info  # noqa: E402
from benchmark_rows import EXTERNAL, DMS, external_rows, dms_rows, domainome_rows, library_sequence  # noqa: E402

REPO_ROOT = paths.REPO_ROOT
MODEL_DIR = str(paths.SPURS_DIR / 'checkpoints')
SPURS_CFG_DIR = paths.SPURS_DIR / 'data' / 'checkpoints' / 'spurs' / '.hydra'

def extract_ddg_prediction(mutation: str, ddg_matrix: torch.Tensor, assert_wt_zero: bool = True) -> float:
    ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'

    wt_aa = mutation[0]
    mut_aa = mutation[-1]
    pos = int(mutation[1:-1]) - 1  

    mut_idx = ALPHABET.index(mut_aa)
    
    if assert_wt_zero:
        wt_idx = ALPHABET.index(wt_aa)
        wt_score = ddg_matrix[pos, wt_idx].item()
        # Upgraded to strictly raise instead of silently printing and passing
        assert abs(wt_score) <= 1e-6, f"Expected WT score at index {wt_idx} to be ~0.0, but got {wt_score}"

    return ddg_matrix[pos, mut_idx].item()

def load_spurs_models(checkpoint_single, checkpoint_multi, models='both'):
    """(model_single, model_multi, cfg_single, cfg_multi); a model that `models` does not ask for is not loaded (None).
    'hf' (or None) is the published checkpoint from the Hugging Face hub; anything else is a file under SPURS/checkpoints."""
    print("Loading SPURS models...")
    model_single = model_multi = cfg_single = cfg_multi = None
    # cfg_single is needed by every parse (parse_pdb reads the single model's config whichever model runs), so with
    # --models multi the single model is built only to get it and dropped
    if checkpoint_single is not None and (checkpoint_single != 'hf'):
        cfg_single = OmegaConf.load(str(SPURS_CFG_DIR / 'config.yaml'))
        del cfg_single['model']['_target_']
        seed_everything(cfg_single['train']['seed'])
        model_single = SPURS(cfg_single['model']).to('cuda')
        ckpt_single = torch.load(os.path.join(MODEL_DIR, checkpoint_single), map_location=torch.device('cpu'), weights_only=False)['state_dict']
        model_single.load_state_dict({k[6:]: v for k, v in ckpt_single.items() if 'model.' in k}, strict=True)
    else:
        model_single, cfg_single = get_SPURS_from_hub()
    model_single.eval()
    if models == 'multi':
        del model_single
        model_single = None
        torch.cuda.empty_cache()
    if models in ('multi', 'both'):
        if (checkpoint_multi is not None) and (checkpoint_multi != 'hf'):
            cfg_multi = OmegaConf.load(str(SPURS_CFG_DIR / 'config_multi.yaml'))
            del cfg_multi['model']['_target_']
            seed_everything(cfg_multi['train']['seed'])
            model_multi = SPURSMulti(cfg_multi['model']).to('cuda')
            ckpt_multi = torch.load(os.path.join(MODEL_DIR, checkpoint_multi), map_location=torch.device('cpu'), weights_only=False)['state_dict']
            model_multi.load_state_dict({k[6:]: v for k, v in ckpt_multi.items() if 'model.' in k}, strict=True)
        else:
            model_multi, cfg_multi = get_SPURS_multi_from_hub()
        model_multi.eval()
    return model_single, model_multi, cfg_single, cfg_multi


def run_multi(model_multi, pdb_data, mutation_lists, batching, label='', info_out=None):
    """(ddG array, batch info): SPURSMulti ddG for each mutation list, grouped by number of mutations and run in the batches
    `batching` decides (baselines/batching.py; fixed default 4096 variants per call); the structure/ESM2 pass is repeated
    once per call."""
    out = np.full(len(mutation_lists), np.nan)
    by_len = {}
    for i, muts in enumerate(mutation_lists):
        by_len.setdefault(len(muts), []).append(i)
    infos = []
    for n_mut, idx in by_len.items():
        def chunk_fn(a, b, idx=idx):
            sub = idx[a:b]
            mut_ids, append_tensors = parse_pdb_for_mutation([mutation_lists[i] for i in sub])
            pdb_data_group = pdb_data.copy()
            pdb_data_group['mut_ids'] = mut_ids
            pdb_data_group['append_tensors'] = append_tensors.to('cuda')
            with torch.no_grad():
                ddg_multi = model_multi(pdb_data_group)
            out[sub] = ddg_multi.view(-1).float().cpu().numpy()
        infos.append(batching.run(len(idx), chunk_fn, label=f'{label} {n_mut}-mut', default_chunk=4096))
    if info_out is not None:
        info_out.extend(infos)
    return out, merge_info(infos)


def _seq_len(pdb_data):
    try:
        return int(len(pdb_data['seq']))
    except (KeyError, TypeError):
        return None


def _limit(df, n):
    """--max_proteins: only the rows of the first n proteins (by `code`) of a frame; everything with n == 0."""
    if not n:
        return df
    return df[df['code'].isin(list(dict.fromkeys(df['code']))[:n])]


def _wanted(n_mut, mutants):
    n = np.asarray(n_mut)
    return {'single': n == 1, 'multi': n >= 2, 'both': np.ones(len(n), dtype=bool)}[mutants]


class Spurs:
    """The loaded models and what to compute with them, so every benchmark block asks one object.

    models  'single' (SPURS: one mutation per structure pass; multi-mutants get the sum of their singles), 'multi'
            (SPURS-multi: any number of mutations in one pass, and the sum of its own single predictions) or 'both'
    mutants 'single' | 'multi' | 'both': which rows are scored; the others stay NaN and cost nothing"""

    def __init__(self, args, tracker):
        self.args, self.tracker = args, tracker
        self.mutants, self.models = args.mutants, args.models
        self.batching = args.batching
        with tracker.stage('model_load'):
            self.single, self.multi, self.cfg_single, self.cfg_multi = load_spurs_models(
                args.checkpoint_single, args.checkpoint_multi, args.models)

    @property
    def pred_col(self):
        return 'SPURS_multi' if self.models != 'single' else 'SPURS_single_additive'

    def parse(self, pdb_path, code, chain, overwrite_seq=None):
        return (parse_pdb(pdb_path, code, chain, self.cfg_single, overwrite_seq=overwrite_seq) if overwrite_seq
                else parse_pdb(pdb_path, code, chain, self.cfg_single))

    def score(self, name, pdb_data, mut_strings):
        """The SPURS readouts for one structure (positive = stabilizing, as everywhere here): a dict of arrays aligned
        with mut_strings. The inference calls are one `protein` scope; parsing the structure is not in it."""
        lists = [m.split(':') for m in mut_strings]
        n_mut = np.array([len(m) for m in lists])
        wanted = np.flatnonzero(_wanted(n_mut, self.mutants))
        nan = lambda: np.full(len(lists), np.nan)
        out = {'SPURS_single': nan(), 'SPURS_single_additive': nan(), 'SPURS_multi': nan(), 'SPURS_multi_additive': nan()}
        if not len(wanted):
            return out
        with self.tracker.scope('protein', name, n_rows=len(wanted), seq_len=_seq_len(pdb_data),
                                models=self.models) as sc:
            if self.single is not None:
                with sc.phase('single_forward'), torch.no_grad():
                    ddg_local = self.single(pdb_data, return_logist=True).float().cpu()
                add = np.array([sum(extract_ddg_prediction(m, ddg_local) for m in lists[i]) for i in wanted])
                out['SPURS_single_additive'][wanted] = -add
                one = wanted[n_mut[wanted] == 1]
                out['SPURS_single'][one] = out['SPURS_single_additive'][one]
            if self.multi is not None:
                singles = sorted({m for i in wanted for m in lists[i]})
                with sc.phase('multi_singles'):
                    self.last_infos = []
                    vals, info_s = run_multi(self.multi, pdb_data, [[m] for m in singles], self.batching, f'{name} singles', self.last_infos)
                    s_pred = dict(zip(singles, vals))
                out['SPURS_multi_additive'][wanted] = -np.array([sum(s_pred[m] for m in lists[i]) for i in wanted])
                with sc.phase('multi_targets'):
                    vals, info_t = run_multi(self.multi, pdb_data, [lists[i] for i in wanted], self.batching, f'{name} targets', self.last_infos)
                    out['SPURS_multi'][wanted] = -vals
                sc.note(**merge_info([info_s, info_t]))
        torch.cuda.empty_cache()
        return out

    def score_table(self, df, label, group_cols=('pdb_file', 'code', 'chain')):
        """Score a rows frame (id, code, chain, pdb_file, mut_type, ddG) structure by structure."""
        res = {c: np.full(len(df), np.nan) for c in ('SPURS_single', 'SPURS_single_additive', 'SPURS_multi', 'SPURS_multi_additive')}
        pos = {i: k for k, i in enumerate(df.index)}
        with self.batching.dataset_scope(self.tracker, label, n_rows=len(df)):
            for (pdb_path, code, chain), g in df.groupby(list(group_cols), sort=False):
                try:
                    pdb_data = self.parse(pdb_path, code, chain)
                except Exception as e:
                    print(f'Skipping {code}{chain}: failed parsing PDB: {e}')
                    continue
                r = self.score(f'{code}{chain}', pdb_data, g['mut_type'].tolist())
                idx = [pos[i] for i in g.index]
                for c in res:
                    res[c][idx] = r[c]
        out = df.copy()
        for c, v in res.items():
            out[c] = v
        out['SPURS_multi_epi'] = out['SPURS_multi'] - out['SPURS_multi_additive']
        return out

    def calibrate(self, args):
        """--calibrate_batch_sizes: the SPURS-multi batch for the heaviest proteins of every dataset (calibration.py). The
        smallest batch over the one- and two-mutation groups of a saturated list wins."""
        only = calibration.parse_only(args.calibration_datasets)
        sets = {name: structs for name, (structs, _) in calibration.rows_structures(only).items()}
        tsu = self._tsuboyama_structures(args.split, args.remove_spurs_homologs, only)
        sets.update(tsu)

        def run_one(struct, muts):
            pdb_data = self.parse(struct['pdb_file'], struct['code'], struct['chain'], overwrite_seq=struct.get('overwrite'))
            self.score('calibration', pdb_data, muts)
            return {'batch_max': min(i['batch_max'] for i in self.last_infos), 'batch_ooms': sum(i['batch_ooms'] for i in self.last_infos)}

        sizes, detail = calibration.calibrate(sets, run_one, k=args.calibration_k, cap=args.calibration_cap)
        calibration.save_entries(args.calibrate_batch_sizes, {'spurs': sizes}, {'spurs': detail},
                                 calibration.gpu_meta(args.auto_batch_headroom, 'fp32', args.calibration_k, args.calibration_cap))

    def _tsuboyama_structures(self, split_name, remove_homologs, only):
        """{tsuboyama_val / tsuboyama_test: wild-type-backbone libraries (structure, sequence)} for the calibration."""
        want = [s for s in ('val', 'test') if only is None or f'tsuboyama_{s}' in only]
        if not want or not split_name:
            return {}
        tsu = paths.TSUBOYAMA_DIR
        ds = MegaScaleDatasetPreprocessor(
            data_file=str(paths.require(tsu / 'Tsuboyama2023_Dataset2_Dataset3_20230416.csv', 'the Tsuboyama table')),
            af_model_folder=str(paths.require(tsu / 'AlphaFold_model_PDBs', 'the AlphaFold models')), spurs_override=remove_homologs)
        ds.create_training_splits(str(REPO_ROOT / 'data' / f'{split_name}.pkl'), -1)
        out = {}
        for sc in want:
            df = ds.split_dfs[sc]
            df = df[df['mut_structure'].fillna('-') == '-']
            structs = []
            for code in df['code_wt'].unique():
                seq, pdb = library_sequence(df.loc[df['code_wt'] == code], '-')
                structs.append({'pdb_file': pdb, 'code': code, 'chain': 'A', 'seq': seq, 'overwrite': seq})
            out[f'tsuboyama_{sc}'] = structs
        return out

    def pgym_score_fn(self):
        def score_fn(df):
            code, chain, pdb_path = df['code'].iloc[0], df['chain'].iloc[0], df['pdb_file'].iloc[0]
            with torch.no_grad():
                pdb_data = self.parse(pdb_path, code, chain)
            r = self.score(str(code), pdb_data, df['mut_type_renumbered'].tolist())
            return pd.DataFrame(r)
        return score_fn

    def tsuboyama(self, split_name, out_dir, remove_homologs):
        """The Tsuboyama validation and test splits through the ESM-MSR preprocessor, one library at a time (a
        mutant-background library is scored on its own background sequence). Every row, bounded ones included
        (flagged in `cens`), as esm_msr_testing.py predicts them."""
        split_file = REPO_ROOT / 'data' / f'{split_name}.pkl'
        tsu = paths.TSUBOYAMA_DIR
        ds = MegaScaleDatasetPreprocessor(
            data_file=str(paths.require(tsu / 'Tsuboyama2023_Dataset2_Dataset3_20230416.csv', 'the Tsuboyama table')),
            af_model_folder=str(paths.require(tsu / 'AlphaFold_model_PDBs', 'the AlphaFold models')),
            spurs_override=remove_homologs)
        ds.create_training_splits(str(split_file), -1)
        for scaffold_ in ('val', 'test'):
            data_scaffold = ds.split_dfs[scaffold_]
            parts = []
            with self.batching.dataset_scope(self.tracker, f'tsuboyama_{scaffold_}', n_rows=len(data_scaffold)):
                for code in data_scaffold['code_wt'].unique()[:self.args.max_proteins or None]:
                    df_true = data_scaffold.loc[data_scaffold['code_wt'] == code].copy()
                    df_true['mut_structure'] = df_true['mut_structure'].fillna('-')
                    df_true = sum_individual_mutation_scores(df_true, 'ddG_ML', new_score_column='ddG_additive_ML')
                    df_true['dddG_ML'] = df_true['ddG_ML'] - df_true['ddG_additive_ML']
                    for c in ('SPURS_single', 'SPURS_single_additive', 'SPURS_multi', 'SPURS_multi_additive'):
                        df_true[c] = np.nan
                    for mut_structure, data in df_true.groupby('mut_structure'):
                        mutbb_seq, pdb_path = library_sequence(data, mut_structure)
                        try:
                            pdb_data = self.parse(pdb_path, code, 'A', overwrite_seq=mutbb_seq)
                        except Exception as e:
                            print(f'Skipping sequence {code} - failed parsing PDB: {e}')
                            continue
                        r = self.score(code if mut_structure == '-' else f'{code}_{mut_structure}', pdb_data, data['mut_type'].tolist())
                        for c, v in r.items():
                            df_true.loc[data.index, c] = v
                    df_true['SPURS_multi_epi'] = df_true['SPURS_multi'] - df_true['SPURS_multi_additive']
                    parts.append(df_true.reset_index().set_index('uid'))
            path = Path(out_dir) / f'tsuboyama_{scaffold_}.csv'
            pd.concat(parts, axis=0).to_csv(path)
            print('wrote', path)


def main(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.calibrate_batch_sizes:
        args.batch_mode, args.no_resource_tracking = 'auto', True
    args.batching = batching.Batching(args, model='spurs')
    tracker = resource_tracker.Tracker.from_args(
        args, label='spurs', default_dir=out / 'resources',
        meta={'model': 'SPURS', 'checkpoint_single': args.checkpoint_single, 'checkpoint_multi': args.checkpoint_multi,
              'models': args.models, 'mutants': args.mutants, 'dtype': 'fp32', **args.batching.meta()})
    try:
        spurs = Spurs(args, tracker)
        if args.calibrate_batch_sizes:
            return spurs.calibrate(args)
        if not args.skip_tsuboyama and args.split:
            spurs.tsuboyama(args.split, out, args.remove_spurs_homologs)
        if not args.skip_external:
            for name in EXTERNAL:
                dest = out / f'external_{name}.csv'
                if not (args.skip_existing and dest.exists()):
                    spurs.score_table(_limit(external_rows(name), args.max_proteins), f'external_{name}').to_csv(dest, index=False)
        if not args.skip_dms:
            for assay in DMS[:args.max_proteins or None]:     # each assay is one protein: the smoke subset is the first N assays
                dest = out / f'dms_{assay}.csv'
                if not (args.skip_existing and dest.exists()):
                    spurs.score_table(_limit(dms_rows(assay), args.max_proteins), f'dms_{assay}').to_csv(dest, index=False)
        if not args.skip_domainome:
            dest = out / 'domainome.csv'
            if not (args.skip_existing and dest.exists()):
                spurs.score_table(_limit(domainome_rows(), args.max_proteins), 'domainome').to_csv(dest, index=False)
        if not args.skip_pgym:
            import pgym_runner
            args.batching.dataset = 'proteingym'
            pgym_runner.run_protein_gym(
                args.pgym_dir, out / 'proteingym', spurs.pgym_score_fn(), pred_col=spurs.pred_col, dms=args.pgym_dms,
                tag=f'SPURS {args.checkpoint_single}+{args.checkpoint_multi} models={args.models} mutants={args.mutants}',
                extra_cols=('SPURS_single_additive', 'SPURS_multi_additive'), tracker=tracker)
    finally:
        tracker.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', required=True, help='output directory (CSV per benchmark, proteingym/, resources/)')
    parser.add_argument('--checkpoint_single', default='hf', help="'hf' (the published model) or a file under SPURS/checkpoints")
    parser.add_argument('--checkpoint_multi', default='hf')
    parser.add_argument('--models', choices=['single', 'multi', 'both'], default='both',
                        help='which SPURS models to load and run (single: SPURS; multi: SPURS-multi)')
    parser.add_argument('--mutants', choices=['single', 'multi', 'both'], default='both',
                        help='which rows to score (the others get NaN predictions and no inference)')
    parser.add_argument('--split', default='hyperopt_splits', help='Tsuboyama split pickle under data/ (without .pkl); --skip_tsuboyama to skip')
    parser.add_argument('--remove_spurs_homologs', action='store_true',
                        help='Remove homologs of SPURS training data from the Tsuboyama splits (tests generalization)')
    for b in ('tsuboyama', 'external', 'dms', 'domainome', 'pgym'):
        parser.add_argument(f'--skip_{b}', action='store_true')
    parser.add_argument('--max_proteins', type=int, default=0, help='smoke test: only the first N proteins of every benchmark (0 = all); ProteinGym: use --pgym_dms')
    parser.add_argument('--skip_existing', action='store_true', help='skip an external/DMS/Domainome output that exists')
    parser.add_argument('--pgym_dir', default=str(paths.PGYM_INPUTS), help='pgym_inputs (preprocessing/pgym_preprocess.py)')
    parser.add_argument('--pgym_dms', default='all', help="comma-separated DMS ids, or 'all'")
    batching.add_batch_args(parser)
    calibration.add_calibration_args(parser)
    resource_tracker.add_tracking_args(parser)
    main(parser.parse_args())
