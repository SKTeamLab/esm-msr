"""
--reg_balance constants from grad_share_probe.py outputs.

For each regression-type term T the constant is K_T = rms(|grad rank|) / rms(|grad T|) over the probed batches, on the adapter of T's head
(lora_wt for reg_wt, against rank_wt; lora_mt for reg_mt, comp_off, comp_subst, comp_int, flip_mt, against rank_mt), with every term
recorded at unit weight. A weight of 1 under --reg_balance then gives the term about as much gradient as the head's rank loss.

  python scripts/balance_from_probe.py OUT.json --plain A.json [C.json ...] [--packed B.json]

--plain probes (micro-batch slices) give the plain keys, as the geometric mean over the files; --packed probes (--pack_pair_matrices) give
comp_*_packed and flip_mt_packed. A key that no probe measured keeps training.REG_BALANCE's value and is listed under 'fallback'.
"""
import argparse, json, math, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
DEFAULTS = {'reg_wt': 18.0, 'reg_mt': 84.0, 'comp_off': 23.0, 'comp_subst': 63.0, 'comp_int': 521.0, 'flip_mt': 1.0}
HEAD = {'reg_wt': ('rank_wt', 'lora_wt'), 'reg_mt': ('rank_mt', 'lora_mt'), 'comp_off': ('rank_mt', 'lora_mt'), 'comp_subst': ('rank_mt', 'lora_mt'),
        'comp_int': ('rank_mt', 'lora_mt'), 'flip_mt': ('rank_mt', 'lora_mt')}


def ratios(path):
    """{term: K} from one probe file; terms without both norms in any batch are absent."""
    res = json.load(open(path))['results']
    out = {}
    for term, (rank, grp) in HEAD.items():
        r = [b['norms'][rank][grp] for b in res if rank in b['norms'] and term in b['norms'] and b['norms'][term][grp] > 0]
        t = [b['norms'][term][grp] for b in res if rank in b['norms'] and term in b['norms'] and b['norms'][term][grp] > 0]
        if len(t) >= 3:
            k = math.sqrt(np.mean(np.square(r))) / math.sqrt(np.mean(np.square(t)))
            if np.isfinite(k) and k > 0:
                out[term] = float(k)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--plain', nargs='*', default=[])
    ap.add_argument('--packed', nargs='*', default=[])
    a = ap.parse_args()
    per = {p: ratios(p) for p in a.plain + a.packed if os.path.exists(p)}
    consts, fallback = {}, []
    for term, dflt in DEFAULTS.items():
        vals = [per[p][term] for p in a.plain if p in per and term in per[p]]
        consts[term] = float(np.exp(np.mean(np.log(vals)))) if vals else dflt
        if not vals:
            fallback.append(term)
        if term != 'reg_wt':
            pv = [per[p][term] for p in a.packed if p in per and term in per[p]]
            if pv:
                consts[term + '_packed'] = float(np.exp(np.mean(np.log(pv))))
    out = {'constants': consts, 'per_probe': per, 'fallback': fallback, 'plain': a.plain, 'packed': a.packed,
           'note': 'K = rms rank-loss gradient / rms term gradient (unit weight) on the same head adapter; geometric mean over probes'}
    json.dump(out, open(a.out, 'w'), indent=1)
    print(json.dumps(out['constants'], indent=1))
    if fallback:
        print('fallback (defaults kept):', fallback)


if __name__ == '__main__':
    main()
