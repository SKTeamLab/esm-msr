#!/usr/bin/env python
"""Convert ESM-MSR Lightning checkpoints (.ckpt) into release files: one LoRA .safetensors per seed plus a shared hparams.yaml.

    python inference_scripts/convert_lora_checkpoint.py OUT_DIR SEED1.ckpt [SEED2.ckpt ...]
    python inference_scripts/convert_lora_checkpoint.py OUT_DIR --check     # re-verify an existing OUT_DIR against its sources

Each checkpoint becomes OUT_DIR/lora_seed<k>.safetensors (k = the `seed` in the hparams.yaml next to the checkpoint), and
OUT_DIR/hparams.yaml is written once. The seeds' hparams must agree on everything except `seed` and `experiment_name`, which
is what the loader reads (``esm_msr.inference.parse_hparams_to_lora_config`` finds hparams.yaml next to the weights).

What is kept: the trainable tensors the loader maps (``ESM_MSR.load_lora_weights``): the WT adapter (``model.model.*``), the MT
adapter (``model.peft_mt.*``) and the four calibration scalars, in float32 with their original names, as in the released
lora_seed*.safetensors. Dropped: optimizer and loop state, and ``model.peft_wt.*``, which is the WT adapter registered a second
time (``self.model = self.peft_wt``; the script checks the tensors are identical before dropping them). Any other tensor name
stops the conversion rather than being silently dropped or kept.

hparams.yaml is sanitised: keys that look like credentials (``*key*``, ``*token*``, ``*secret*``, ``*password*``) are removed.
The safetensors metadata records the source checkpoint name, its epoch, the seed and the SHA-256 of the source file.
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import torch
import yaml
from safetensors.torch import load_file, save_file

KEEP = ('model.model.', 'model.peft_mt.', 'model.calibration_head_wt.', 'model.calibration_head_mt.')
ALIAS = ('model.peft_wt.', 'model.model.')            # (dropped alias, the prefix it duplicates)
SECRET = re.compile(r'key|token|secret|password', re.I)
VARYING = {'seed', 'experiment_name'}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def lora_tensors(ckpt_path):
    """The tensors to release from one checkpoint, after checking the alias block and that nothing unknown is present."""
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ck.get('state_dict', ck)
    alias, target = ALIAS
    keep, dropped = {}, {}
    for k, v in sd.items():
        if k.startswith(KEEP):
            keep[k] = v.detach().to(torch.float32).contiguous().clone()
        elif k.startswith(alias):
            dropped[k] = v
        else:
            raise SystemExit(f'{ckpt_path}: unexpected tensor {k!r}; refusing to guess whether the loader needs it')
    for k, v in dropped.items():
        twin = target + k[len(alias):]
        if twin not in keep or not torch.equal(keep[twin], v.to(torch.float32)):
            raise SystemExit(f'{ckpt_path}: {k} is not a copy of {twin}; the WT alias assumption does not hold')
    return keep, int(ck.get('epoch', -1))


def load_hparams(ckpt_path):
    p = Path(ckpt_path).with_name('hparams.yaml')
    h = yaml.safe_load(p.read_text())
    return h.get('hparams', h) if isinstance(h, dict) and 'hparams' in h else h


def sanitise(h):
    return {k: v for k, v in h.items() if not SECRET.search(k)}


def convert(out_dir, ckpts):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    shared, sources = None, {}
    for c in ckpts:
        h = load_hparams(c)
        seed = int(h['seed'])
        core = {k: v for k, v in sanitise(h).items() if k not in VARYING}
        if shared is None:
            shared, first = core, h
        elif core != shared:
            diff = sorted(k for k in set(core) | set(shared) if core.get(k) != shared.get(k))
            raise SystemExit(f'{c}: hparams differ from the first seed in {diff}; release them separately')
        tensors, epoch = lora_tensors(c)
        meta = {'format': 'esm-msr-lora', 'source_checkpoint': Path(c).name, 'source_sha256': sha256(c),
                'epoch': str(epoch), 'seed': str(seed), 'n_tensors': str(len(tensors))}
        dst = out / f'lora_seed{seed}.safetensors'
        save_file(tensors, str(dst), metadata=meta)
        sources[seed] = str(Path(c).resolve())
        print(f'seed {seed}: {len(tensors)} tensors from {c} (epoch {epoch}) -> {dst} ({dst.stat().st_size / 1e6:.1f} MB)')
    hp = sanitise(first)
    hp['seed'] = 1 if 1 in sources else min(sources)
    hp.pop('experiment_name', None)
    with open(out / 'hparams.yaml', 'w') as f:
        f.write('# Shared by every lora_seed<k>.safetensors in this folder; the seeds differ only in `seed`.\n')
        yaml.safe_dump(hp, f, sort_keys=True)
    (out / 'conversion_sources.json').write_text(json.dumps(sources, indent=2) + '\n')
    return sources


def check(out_dir, sources=None):
    """Every released tensor equals its source checkpoint's tensor bit for bit, and nothing the loader maps is missing."""
    out = Path(out_dir)
    sources = sources or {int(k): v for k, v in json.loads((out / 'conversion_sources.json').read_text()).items()}
    ok = True
    for seed, c in sorted(sources.items()):
        rel = load_file(str(out / f'lora_seed{seed}.safetensors'))
        ref, _ = lora_tensors(c)
        same = set(rel) == set(ref) and all(torch.equal(rel[k], ref[k]) for k in ref)
        print(f'seed {seed}: {len(rel)} tensors, identical to {Path(c).name}: {same}')
        ok &= same
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('out_dir')
    ap.add_argument('ckpts', nargs='*')
    ap.add_argument('--check', action='store_true', help='only verify OUT_DIR against conversion_sources.json')
    a = ap.parse_args()
    if not a.check:
        if not a.ckpts:
            ap.error('give the checkpoints to convert')
        convert(a.out_dir, a.ckpts)
    sys.exit(0 if check(a.out_dir) else 1)


if __name__ == '__main__':
    main()
