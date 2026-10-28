"""Save MutateEverything's predictions for the DMS assays, without upstream's detection metric.

Why this exists: engine_train.evaluate writes results_<set>.csv only AFTER eval_ddg returns, and eval_ddg's
compute_detection_metrics asserts `len(metrics_pdb) > 0`. It counts a protein only if that protein has at least one
row with ddg <= -0.5, i.e. a strongly stabilizing mutation in upstream's convention. These DMS assays are scaled so
that never happens (DLG4's ddG_ML peaks at 0.374, so ddg bottoms out at -0.374), and each assay is a single protein,
so metrics_pdb comes back empty and the run dies before saving anything.

Nothing about the model or the data is changed here: the model, SeqDetDatatset, the collate function and the
prediction extraction are upstream's, copied from engine_train.evaluate. Only the metric block is dropped -- the
scoring is done afterwards by baselines/score_baselines.py against ESM-MSR's own rows anyway.
"""
import os
import sys
import time
from functools import partial

import numpy as np
import pandas as pd
import torch

import misc
from data import SeqDetDatatset, protein_collate_fn, get_dense_double_mut_infos, get_dense_mut_infos
from engine_train import mem_inputs_to_device
from main_train import get_args_parser
from modeling.mutate_everything import MutateEverything

# the shared resource tracker lives in the ESM-MSR repo's inference_scripts (the same module the other wrappers use)
INFERENCE_SCRIPTS = os.environ.get('MSR_INFERENCE_SCRIPTS', '/home/sreeves/esm-msr-devel/inference_scripts')
sys.path.insert(0, INFERENCE_SCRIPTS)
import resource_tracker


@torch.no_grad()
def predict(model, dl, device, args, tracker=None):
    """The prediction half of engine_train.evaluate, verbatim, with the metric call removed.

    Resource tracking (tracker, batch size 1 so a batch is one protein): fetching the batch (the dataset's
    featurization, MSAs included) is a `load` scope; moving it to the GPU, the model forward and turning the dense
    output into per-mutation scores are phases of that protein's `protein` scope. The model predicts every single and
    double of the protein in one pass whatever subset is asked for, so the cost is per protein, not per variant."""
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
        with tracker.scope('protein', str(batch['pdb_ids'][0]), seq_len=len(batch['seqs'][0]), batch_size=len(batch['seqs'])) as sc:
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
    misc.init_distributed_mode(args)
    device = torch.device(args.device)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    world = int(os.environ.get('WORLD_SIZE', '1'))
    if world > 1:
        print(f'WARNING: WORLD_SIZE={world}: predict_dms.py does not shard, every rank would repeat the whole run; '
              'resource tracking is off. Use NPROC=1.', flush=True)
        tracker = resource_tracker.NullTracker()
    else:
        tracker = resource_tracker.Tracker.from_args(
            args, label='mutate_everything', default_dir=os.path.join(args.output_dir, 'resources'),
            meta={'model': 'MutateEverything', 'resume': os.path.basename(str(args.resume)), 'backbone': args.backbone,
                  'seed': args.seed, 'dtype': 'fp32', 'batch_size': 1, 'dataloader_workers': 0})
    with tracker.stage('model_load'):
        model = MutateEverything(args)
        alphabet = model.backbone.get_alphabet()
        model.to(args.device)
        misc.load_model(args, model, None, None)

    paths = args.eval_data_paths.split(',')
    if len(paths) == 1 and paths[0].endswith('.txt'):   # a file listing the sets, one per line (ProteinGym has 217)
        paths = [p.strip() for p in open(paths[0]) if p.strip()]
    for path in paths:
        name = os.path.basename(path).replace('.csv', '')
        dest = os.path.join(args.output_dir, f'results_{name}.csv')
        if os.environ.get('SKIP_EXISTING') == '1' and os.path.exists(dest):
            print(f'{name}: SKIP (results exist)', flush=True)
            continue
        _ds_scope = tracker.scope('dataset', name)
        _ds_scope.__enter__()
        ds = SeqDetDatatset(path, args, train=False)
        _ds_scope.note(n_proteins=len(ds))
        collate_fn = partial(protein_collate_fn, alphabet=alphabet, args=args)
        # num_workers=0: the batch is featurized in this process, so its CPU and RAM show up in the `load` scopes
        dl = torch.utils.data.DataLoader(ds, batch_size=1, collate_fn=collate_fn, shuffle=False, num_workers=0)
        if os.environ.get('CONTINUE_ON_ERROR') == '1':
            # one set that does not fit (the AF2 trunk is uncropped below max_context_length) must not end the run
            try:
                preds = predict(model, dl, device, args, tracker)
            except Exception as e:
                print(f'{name}: FAIL {type(e).__name__}: {str(e)[:300]}', flush=True)
                torch.cuda.empty_cache()
                _ds_scope.__exit__(type(e), e, None)
                continue
        else:
            preds = predict(model, dl, device, args, tracker)
        rows = []
        for pdb_id, p in preds.items():
            rows.append(pd.DataFrame({'pdb_id': pdb_id, 'mut_info': p['mutations'], 'scores': p['scores']}))
        out = pd.concat(rows)
        out['ddg_pred'] = -out['scores']
        name = os.path.basename(path).replace('.csv', '')
        dest = os.path.join(args.output_dir, f'results_{name}.csv')
        out.to_csv(dest, index=False)
        print(f'{name}: {len(out)} predictions -> {dest}', flush=True)
        _ds_scope.__exit__(None, None, None)
    tracker.close()


if __name__ == '__main__':
    p = get_args_parser()
    resource_tracker.add_tracking_args(p)
    a = p.parse_args()
    from pathlib import Path
    a.output_dir = Path(a.output_dir)
    main(a)
