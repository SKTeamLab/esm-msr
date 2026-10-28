"""MutateEverything (jozhang97/MutateEverything at the commit in third_party.lock, unmodified): inference on every
benchmark, with resource tracking. One process, one GPU.

Upstream's evaluation entry (main_train.py --eval) computes a detection metric that cannot run on the DMS assays (it
needs a protein with a strongly stabilizing mutation) and, with --dist_eval, shards the sets over several GPUs. This
script keeps upstream's model, dataset, collate function and prediction extraction (copied from engine_train.evaluate)
and drops the metric, so every set runs in one process in the same resources; scoring happens afterwards against
ESM-MSR's own rows. It writes results_<set>.csv (pdb_id, mut_info, scores, ddg_pred) like upstream.

Any MutateEverything checkpoint works: --resume stage1.pth (singles) or stage2.pth (singles + doubles). The model predicts
every single and double of a protein in one pass whatever subset is asked for; --mutants only chooses which rows are
requested. `single` requests no doubles, so the pair head is skipped (upstream's `only_eval_single`), which is the cheap
way to benchmark a singles model; `multi` requests only rows with two or more mutations; `both` is everything.

  everything: python mutate_everything_inference.py --resume CKPT --output_dir OUT --sets all
              (Tsuboyama val/test/testcomp, the 12 external sets, the 7 DMS, Domainome, ProteinGym)
  some:       --sets tsuboyama | external | dms | domainome | pgym | s669,ssym,val,test,DLG4_HUMAN_Faure_2021_abundance_domain,...

The prepared inputs live under $MUTATE_EVERYTHING_DIR/data/ (prepare/build_*_me_data.py write them; the MSAs are data).
Per-protein wall time, CPU, RAM, VRAM and I/O of the forward pass (and, separately, of the batch featurization, which
runs in this process) are recorded by default; --no_resource_tracking to skip.
"""
import os
import sys
from functools import partial
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                 # paths
sys.path.insert(0, str(HERE.parent))          # resource_tracker
import paths  # noqa: E402
import resource_tracker  # noqa: E402

ME_DIR = str(paths.MUTATE_EVERYTHING_DIR)
if not os.path.isdir(ME_DIR):
    sys.exit(f'MutateEverything checkout not found at {ME_DIR} (inference_scripts/baselines/setup_third_party.sh, or MUTATE_EVERYTHING_DIR)')
os.chdir(ME_DIR)                              # upstream resolves models/, data/ and its own imports relative to its checkout
sys.path.insert(0, ME_DIR)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import misc  # noqa: E402  (upstream)
from data import SeqDetDatatset, protein_collate_fn, get_dense_mut_infos  # noqa: E402
from engine_train import mem_inputs_to_device  # noqa: E402
from main_train import get_args_parser  # noqa: E402
from modeling.mutate_everything import MutateEverything  # noqa: E402

EXTERNAL_E = ['s669', 'ssym', 'q3421', 'k3822', 'ptmul']           # data/external_test_feb
EXTERNAL_A = ['s571', 's783', 's2648', 's8754']                    # data/additional_test_march
DMS = ['DLG4_HUMAN_Faure_2021_abundance_domain', 'DLG4_HUMAN_Faure_2021_binding_domain',
       'GRB2_HUMAN_Faure_2021_abundance_domain', 'GRB2_HUMAN_Faure_2021_binding_domain',
       'MYO_HUMAN_Kung_2025_display', 'ESTA_BACSU_Nutschel_2020_dTm', 'GB1_Wu_2016_binding_domain']


def resolve_sets(sets, split_name):
    """--sets (group names and/or single set names) -> the list of input CSVs."""
    root = Path(ME_DIR) / 'data'
    tsu = lambda n: root / split_name / 'mutations' / f'{split_name}_{n}.csv'
    one = {**{n: root / 'external_test_feb' / 'mutations' / f'{n}.csv' for n in EXTERNAL_E},
           **{n: root / 'additional_test_march' / 'mutations' / f'{n}.csv' for n in EXTERNAL_A},
           's461': root / 's461' / 'mutations' / 's461.csv',
           **{n: tsu(n) for n in ('val', 'valfull', 'test', 'testcomp')},
           **{n: root / 'dms_oct06' / 'mutations' / f'{n}.csv' for n in DMS},
           'domainome': root / 'domainome_oct06' / 'mutations' / 'domainome.csv',
           **{p.stem: p for p in (root / 'pgym_oct06' / 'mutations').glob('*.csv')}}      # a single ProteinGym DMS by id
    groups = {'tsuboyama': [tsu(n) for n in ('val', 'test', 'testcomp')],
              'external': [one[n] for n in EXTERNAL_E + EXTERNAL_A + ['s461']],
              'dms': [one[n] for n in DMS], 'domainome': [one['domainome']],
              'pgym': sorted((root / 'pgym_oct06' / 'mutations').glob('*.csv'))}
    groups['all'] = groups['tsuboyama'] + groups['external'] + groups['dms'] + groups['domainome'] + groups['pgym']
    out = []
    for s in [x.strip() for x in sets.split(',') if x.strip()]:
        if s in groups:
            out += groups[s]
        elif s in one:
            out.append(one[s])
        else:
            sys.exit(f"unknown set '{s}' (groups: {sorted(groups)}; sets: {sorted(one)})")
    missing = [str(p) for p in out if not Path(p).exists()]
    if missing:
        sys.exit(f'missing inputs (prepare/build_*_me_data.py writes them): {missing[:5]}{" ..." if len(missing) > 5 else ""}')
    return out


def filter_mutants(path, mutants, out_dir, max_proteins=0):
    """The input CSV restricted to the rows --mutants asks for and, for a smoke test, to the first `max_proteins` proteins
    (written under out_dir/inputs), or the CSV itself when neither applies."""
    if mutants == 'both' and not max_proteins:
        return str(path)
    df = pd.read_csv(path, low_memory=False)
    if max_proteins:
        df = df[df['pdb_id'].isin(list(dict.fromkeys(df['pdb_id']))[:max_proteins])]
    multi = df['mut_info'].astype(str).str.contains(':')
    df = df if mutants == 'both' else (df[~multi] if mutants == 'single' else df[multi])
    dest = Path(out_dir) / 'inputs' / f'{Path(path).stem}.{mutants}{f".first{max_proteins}" if max_proteins else ""}.csv'
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    return str(dest)


@torch.no_grad()
def predict(model, dl, device, args, tracker=None):
    """The prediction half of engine_train.evaluate, verbatim, with the metric call removed.

    Resource tracking (batch size 1, so a batch is one protein): fetching the batch (the dataset's featurization, MSAs
    included) is a `load` scope; moving it to the GPU, the model forward and turning the dense output into per-mutation
    scores are phases of that protein's `protein` scope. The model predicts every single and double of the protein in
    one pass whatever subset is asked for, so the cost is per protein, not per variant."""
    tracker = tracker or resource_tracker.NullTracker()
    model.eval()
    all_preds = {}
    it = iter(dl)
    k = 0
    while True:
        k += 1
        try:
            with tracker.scope('load', f'batch{k}'):
                batch = next(it)
        except StopIteration:
            break
        with tracker.scope('protein', str(batch['pdb_ids'][0]), seq_len=len(batch['seqs'][0]),
                           batch_size=len(batch['seqs'])) as sc:
            all_preds.update(_predict_batch(model, batch, device, args, sc))
    return all_preds


def _predict_batch(model, batch, device, args, sc):
    all_preds = {}
    with sc.phase('h2d'):
        x = mem_inputs_to_device(batch, device, args)
    batch['known_mask1'] = known_mask1 = batch['ddg_dense'] != 999
    batch['known_mask2'] = known_mask2 = batch['ddg_dense2'] != 999
    eval_single = known_mask1.any()
    eval_double = known_mask2.any()
    eval_list = 'mut_info_list' in batch and sum([len(y) for y in batch['mut_info_list']]) > 0
    batch['only_eval_single'] = eval_single and not eval_double and not eval_list
    batch['eval_list'] = eval_list
    with sc.phase('count'):      # the requested variants, so rows/s can be computed; its own phase so it can be excluded
        sc.note(n_rows=int(known_mask1.sum().item() + known_mask2.sum().item()
                           + (sum(len(y) for y in batch['mut_info_list']) if eval_list else 0)))

    with sc.phase('forward'):
        pred_dict = model(x, batch)
    pr1 = pred_dict['mut1_ddg']
    pr2 = pred_dict.get('mut2_ddg', None)

    with sc.phase('extract'):
        for b in range(len(x)):
            pdb_id = batch['pdb_ids'][b]
            seq = batch['seqs'][b]
            muts, scores = [], []
            if eval_list:
                muts.append(np.array(batch['mut_info_list'][b]))
                scores.append(-pred_dict['pr_ddgs_list'][b].detach().cpu().numpy())
            if eval_double:
                # same entries, same order as get_dense_double_mut_infos(seq)[keep & valid_mask], but the mut_info
                # strings are built only for the kept entries: the dense list has (L*20)^2 strings (237M at L=770)
                singles = np.array(get_dense_mut_infos(seq))
                pos = np.repeat(np.arange(1, len(seq) + 1), 20)
                sel = np.flatnonzero(known_mask2[b].flatten().cpu().numpy())
                i, j = np.divmod(sel, len(singles))
                valid = pos[i] < pos[j]
                sel, i, j = sel[valid], i[valid], j[valid]
                muts.append(np.char.add(np.char.add(singles[i], ':'), singles[j]))
                scores.append(-pr2[b].flatten()[torch.from_numpy(sel).to(pr2.device)].detach().cpu().numpy())
            if eval_single:
                mutations = np.array(get_dense_mut_infos(seq))
                pr_ddgs = pr1[b].flatten()
                keep = known_mask1[b].flatten().cpu().numpy()
                muts.append(mutations[keep])
                scores.append(-pr_ddgs[keep].detach().cpu().numpy())
            all_preds[pdb_id] = {'mutations': np.concatenate(muts), 'scores': np.concatenate(scores)}
    return all_preds


def main(args):
    if int(os.environ.get('WORLD_SIZE', '1')) > 1:
        sys.exit('mutate_everything_inference.py is a single-process script (WORLD_SIZE > 1 would repeat every set on every rank)')
    misc.init_distributed_mode(args)
    device = torch.device(args.device)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    tracker = resource_tracker.Tracker.from_args(
        args, label='mutate_everything', default_dir=os.path.join(args.output_dir, 'resources'),
        meta={'model': 'MutateEverything', 'resume': os.path.basename(str(args.resume)), 'backbone': args.backbone,
              'seed': args.seed, 'dtype': 'fp32', 'batch_size': 1, 'dataloader_workers': 0, 'mutants': args.mutants,
              'sets': args.sets})
    try:
        with tracker.stage('model_load'):
            model = MutateEverything(args)
            alphabet = model.backbone.get_alphabet()
            model.to(args.device)
            misc.load_model(args, model, None, None)
        for path in resolve_sets(args.sets, args.split_name):
            name = os.path.basename(str(path)).replace('.csv', '')
            dest = os.path.join(args.output_dir, f'results_{name}.csv')
            if args.skip_existing and os.path.exists(dest):
                print(f'{name}: SKIP (results exist)', flush=True)
                continue
            with tracker.scope('dataset', name) as ds_scope:
                ds = SeqDetDatatset(filter_mutants(path, args.mutants, args.output_dir, args.max_proteins), args, train=False)
                ds_scope.note(n_proteins=len(ds))
                collate_fn = partial(protein_collate_fn, alphabet=alphabet, args=args)
                # num_workers=0: the batch is featurized in this process, so its CPU and RAM show up in the `load` scopes
                dl = torch.utils.data.DataLoader(ds, batch_size=1, collate_fn=collate_fn, shuffle=False, num_workers=0)
                try:
                    preds = predict(model, dl, device, args, tracker)
                except Exception as e:           # one set that does not fit (the AF2 trunk is uncropped) must not end the run
                    if not args.continue_on_error:
                        raise
                    print(f'{name}: FAIL {type(e).__name__}: {str(e)[:300]}', flush=True)
                    torch.cuda.empty_cache()
                    ds_scope.note(failed=f'{type(e).__name__}')
                    continue
                out = pd.concat([pd.DataFrame({'pdb_id': pdb_id, 'mut_info': p['mutations'], 'scores': p['scores']})
                                 for pdb_id, p in preds.items()])
                out['ddg_pred'] = -out['scores']
                out.to_csv(dest, index=False)
                print(f'{name}: {len(out)} predictions -> {dest}', flush=True)
    finally:
        tracker.close()


if __name__ == '__main__':
    p = get_args_parser()
    # the recipe the corrected baselines were evaluated with
    p.set_defaults(backbone='af', finetune_backbone='models/finetuning_ptm_2.pt', batch_size=1, eval=True, disable_wandb=True)
    p.add_argument('--sets', default='all', help='all | tsuboyama | external | dms | domainome | pgym | comma-separated set names')
    p.add_argument('--split_name', default='oct06_capped_allbl', help='the Tsuboyama split directory under data/')
    p.add_argument('--mutants', choices=['single', 'multi', 'both'], default='both',
                   help='which rows of each set to request (single: no doubles, so the pair head is skipped)')
    p.add_argument('--max_proteins', type=int, default=0, help='smoke test: only the first N proteins of every set (0 = all)')
    p.add_argument('--skip_existing', action='store_true', help='skip a set whose results_<set>.csv exists (resumable ProteinGym)')
    p.add_argument('--continue_on_error', action='store_true', help='log a failing set (e.g. too long for the GPU) and go on')
    resource_tracker.add_tracking_args(p)
    a = p.parse_args()
    a.output_dir = Path(a.output_dir)
    main(a)
