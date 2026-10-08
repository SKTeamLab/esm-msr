"""
Gradient-share probe: how large is each loss term's gradient on the trainable parameters, per parameter group, for real training batches?

CPU, fp32, the real backbone. For every batch the full composition (esm_msr.training) runs once; each loss term, as it enters the total loss
(already multiplied by its lambda and its batch normalisation), is differentiated separately with torch.autograd.grad (the terms are tapped
by name through ESM3EpistasisLightningModule._tap). Component terms are recorded at UNIT weight, so every figure is "per unit of weight".

Usage: PYTHONPATH=src python scripts/grad_share_probe.py OUT.json [--ckpt PATH] [--batches 3] [--batch_size 96] [--proteins 4] [--seed 1]
The flags of scripts/run_arm.sh are used as they are (new split, link on, out-of-range items in), except: CPU/fp32, micro_batch_size 38 (the
smallest that lets the component path run), every lambda 1 and mt_comp_* (1, 1, 1.0001) so the component path is taken but at unit weight.
"""
import argparse, json, os, re, shlex, sys
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('out')
ap.add_argument('--ckpt', default=None)
ap.add_argument('--batches', type=int, default=3)
ap.add_argument('--batch_size', type=int, default=96)
ap.add_argument('--proteins', type=int, default=4)
ap.add_argument('--seed', type=int, default=1)
ap.add_argument('--threads', type=int, default=4)
ap.add_argument('--gpu', action='store_true', help='the real training setup: cuda, bf16 mixed, frozen Linears in bf16 (use batch_size 256, micro 64)')
ap.add_argument('--min_cond', type=int, default=30)
ap.add_argument('--min_single', type=int, default=0)
ap.add_argument('--balanced', action='store_true', help='run with --reg_balance: the regression terms are then in rank-gradient units, so ratios should be ~1')
ap.add_argument('--plain', action='store_true', help='plain MT regression (no component path), micro-batch 16: far less memory, no comp_* terms')
ap.add_argument('--micro', type=int, default=38)
ap.add_argument('--max_len', type=int, default=64, help='only batches whose padded token length is at most this: activation memory scales with it')
ap.add_argument('--rss_cap_gb', type=float, default=26.0)
pa = ap.parse_args()
if not pa.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.set_num_threads(pa.threads)

import threading, time, resource
def _rss_gb():
    with open('/proc/self/statm') as f:
        return int(f.read().split()[1]) * os.sysconf('SC_PAGE_SIZE') / 2 ** 30
def _watchdog():                       # never let the probe take the machine down with a training run on it
    while True:
        r = _rss_gb()
        if r > pa.rss_cap_gb:
            print(f'WATCHDOG: RSS {r:.1f} GB > cap, exiting', flush=True); os._exit(3)
        time.sleep(1)
threading.Thread(target=_watchdog, daemon=True).start()

HERE = os.path.dirname(os.path.abspath(__file__))
WT = os.path.dirname(HERE)
txt = open(os.path.join(HERE, 'run_arm.sh')).read().split('training.py', 1)[1].replace('\\\n', ' ')
line = [l for l in txt.split('\n') if l.strip()][0]
line = (line.replace('$WT', WT).replace('${SPLIT:-' + WT + '/data/splits_oct06_capped.pkl}', WT + '/data/splits_oct06_capped.pkl')
        .replace('${COMET_PROJECT:-esm-msr-agent-oct06-capped}', 'probe').replace('$NAME', 'probe').replace('$EPOCHS', '1').replace('$SEED', str(pa.seed)))
line = re.sub(r'--comet_api_key.*$', '', line)
line = re.sub(r'--cache_path cache_v7', '--cache_path /home/sareeves/playground/esm-msr-devel/cache_v7', line)
argv = shlex.split(line) + ['--include_out_of_range', '--precision', 'bf16-mixed' if pa.gpu else '32', '--batch_size', str(pa.batch_size),
                            '--micro_batch_size', '16' if pa.plain else str(pa.micro),
                            '--mt_comp_offset', '1', '--mt_comp_subst', '1', '--mt_comp_int', '1.0' if pa.plain else '1.0001', '--max_train_proteins', str(pa.proteins),
                            '--num_workers', '0'] + (['--reg_balance'] if pa.balanced else []) + [ '--log_dir', '/tmp/probe_logs', '--checkpoint_path', '/tmp/probe_ckpt']
sys.argv = ['probe'] + argv

from esm_msr import training, utils
from esm_msr.config import parse_arguments

args = parse_arguments()
torch.manual_seed(pa.seed); np.random.seed(pa.seed)
tokenizer = training.EsmSequenceTokenizer('cpu')
structure_encoder = training.ESM3_structure_encoder_v0('cpu')
tr, va, be, tn, vn, bn = training.setup_dataloaders(args, tokenizer, structure_encoder, add_benchmarks_to_val=False)
module = training.ESM3EpistasisLightningModule(**vars(args), train_dataloader_names=tn, val_dataloader_names=vn[:1], tokenizer=tokenizer, model_device='cuda:0' if pa.gpu else 'cpu')
if pa.ckpt:
    sd = torch.load(pa.ckpt, map_location='cpu', weights_only=False)['state_dict']
    res = module.load_state_dict(sd, strict=False)
    print(f'loaded {pa.ckpt}: {len(sd)} tensors; unexpected {len(res.unexpected_keys)}', flush=True)
DEV = torch.device('cuda:0' if pa.gpu else 'cpu')
if pa.gpu:
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get('PROBE_GPU_FRACTION', '0.6')))     # fail with a clean OOM, never crowd the card
if pa.gpu:
    module.to(DEV)
    module._cast_frozen_linears_bf16()
else:
    module._cast_frozen_linears_bf16 = lambda: None      # fp32 on CPU (bf16 autocast breaks ESM3's geometric attention on CPU)
module.model.train()
def to_dev(x):
    if torch.is_tensor(x): return x.to(DEV)
    if isinstance(x, dict): return {k: to_dev(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)) and x and all(torch.is_tensor(v) for v in x): return type(x)(v.to(DEV) for v in x)
    return x
if hasattr(module, 'on_train_start'):
    try: module._reshare_base_params_on_device()
    except Exception as e: print('reshare skipped:', e)

named = [(n, p) for n, p in module.model.named_parameters() if p.requires_grad]
groups = {'lora_wt': [], 'calib_wt': [], 'lora_mt': [], 'calib_mt': [], 'link': []}
for n, p in named:
    if 'calibration_head_wt' in n: groups['calib_wt'].append(p)
    elif 'calibration_head_mt' in n: groups['calib_mt'].append(p)
    elif 'wt_adapter' in n: groups['lora_wt'].append(p)
    elif 'mt_adapter' in n: groups['lora_mt'].append(p)
if module.link_head is not None:
    groups['link'] = [p for p in module.link_head.parameters() if p.requires_grad]
print({k: len(v) for k, v in groups.items()}, flush=True)
assert all(len(v) for k, v in groups.items() if k != 'link'), 'a parameter group is empty: the name classification is wrong'
all_params = [p for v in groups.values() for p in v]
slices, o = {}, 0
for k, v in groups.items():
    slices[k] = slice(o, o + len(v)); o += len(v)

acc = {}                                   # term -> list over parameters of summed gradient tensors (summed over the units of a batch)
def hook(total):
    live = [(n, t) for n, t in module._probe if t.requires_grad]
    for j, (name, term) in enumerate(live):
        g = torch.autograd.grad(term, all_params, retain_graph=j < len(live) - 1, allow_unused=True)   # the last term frees the unit's graph
        cur = acc.setdefault(name, [None] * len(all_params))
        for i, gi in enumerate(g):
            if gi is not None:
                cur[i] = gi.detach().clone() if cur[i] is None else cur[i] + gi.detach()
    module._probe.clear()
module.manual_backward = hook

def norms(vecs):
    return {k: float(torch.sqrt(sum((vecs[i].float() ** 2).sum() for i in range(s.start, s.stop) if vecs[i] is not None) or torch.zeros(()))) for k, s in slices.items()}
def combine(names):
    out = [None] * len(all_params)
    for nme in names:
        for i, gi in enumerate(acc.get(nme, [None] * len(all_params))):
            if gi is not None:
                out[i] = gi.clone() if out[i] is None else out[i] + gi
    return out
def dot_cos(a, b, s):
    num = sum((a[i].float() * b[i].float()).sum() for i in range(s.start, s.stop) if a[i] is not None and b[i] is not None)
    na = sum((a[i].float() ** 2).sum() for i in range(s.start, s.stop) if a[i] is not None) ** 0.5
    nb = sum((b[i].float() ** 2).sum() for i in range(s.start, s.stop) if b[i] is not None) ** 0.5
    return float(num / (na * nb)) if (na and nb) else float('nan')

results = []
import itertools
it = itertools.chain.from_iterable(tr)      # every protein's loader in turn
scanned = 0
for bi in range(pa.batches):
    acc.clear()
    while True:                       # only batches that carry conditional items exercise the MT rank and component terms
        batch = utils._normalize_batch(next(it)); scanned += 1
        if (sum(1 for x in batch.get('subset_type', []) if x == 'cond') >= pa.min_cond and sum(1 for x in batch.get('subset_type', []) if x == 'single') >= pa.min_single and batch['wt_sequence_tokens'].shape[-1] <= pa.max_len) or scanned > 3000:
            break
    print(f'batch {bi}: scanned {scanned}, rows {len(batch["subset_type"])}, rss {_rss_gb():.1f} GB', flush=True)
    module._probe = []
    batch = to_dev(batch)
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=pa.gpu):
        logs = module._compose_losses_streaming_and_backward(batch)
    module._probe = None
    print(f'  done, rss {_rss_gb():.1f} GB', flush=True)
    st = list(batch.get('subset_type', []))
    rec = {'batch': bi, 'scanned': scanned, 'rows': len(st), 'subsets': {s: st.count(s) for s in set(st)}, 'logs': {k: float(v) for k, v in logs.items()}, 'norms': {}, 'cos': {}}
    for nme in list(acc):
        rec['norms'][nme] = norms(acc[nme])
    mt_reg = combine(['comp_off', 'comp_subst', 'comp_int', 'reg_mt'])
    wt_reg = combine(['reg_wt'])
    rec['norms']['reg_mt_total'] = norms(mt_reg); rec['norms']['reg_wt_total'] = norms(wt_reg)
    for hd, (rk, rg) in {'wt': ('rank_wt', wt_reg), 'mt': ('rank_mt', mt_reg)}.items():
        if rk in acc:
            rec['cos'][hd] = {g: dot_cos(acc[rk], rg, slices[g]) for g in groups if hd in g or g == 'link'}
    results.append(rec)
    print(json.dumps(rec['norms']), flush=True)
json.dump({'ckpt': pa.ckpt, 'args': {'batch_size': pa.batch_size, 'proteins': pa.proteins}, 'results': results}, open(pa.out, 'w'), indent=1)
print('PROBE_DONE', flush=True)
