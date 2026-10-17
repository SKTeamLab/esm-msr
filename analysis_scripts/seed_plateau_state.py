"""
Copy a checkpoint and seed the MT plateau's history into it, so a resumed run behaves as if it had watched the
earlier epochs.

Why this is needed. A plateau's state lives in the checkpoint's ``transition_state``, and dc8 ran with
--mt_plateau_patience 0, so _mt_plateau_step returned before ever writing ``mt_plateau_best`` - the key is not
even present. Resume from such a checkpoint with the plateau ON and it starts from -inf, so the FIRST epoch after
the resume counts as an improvement whatever its value, and the decay that should fire there does not. The run
then looks like a plateau experiment and is not one.

  PYTHONPATH=src python analysis_scripts/seed_plateau_state.py IN.ckpt OUT.ckpt --mt_best 0.5331

--mt_best is the best value of the watched metric over the epochs the checkpoint already contains, i.e. the peak
the plateau would be holding at that point. --mt_bad defaults to 0 (the peak epoch itself was an improvement).
"""
import argparse
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('src'); p.add_argument('dst')
    p.add_argument('--mt_best', type=float, required=True)
    p.add_argument('--mt_bad', type=int, default=0)
    p.add_argument('--mt_decays', type=int, default=0)
    a = p.parse_args()

    c = torch.load(a.src, map_location='cpu', weights_only=False)
    ts = dict(c.get('transition_state') or {})
    before = {k: ts.get(k) for k in ('mt_plateau_best', 'mt_plateau_bad', 'mt_decays')}
    ts.update(mt_plateau_best=float(a.mt_best), mt_plateau_bad=int(a.mt_bad), mt_decays=int(a.mt_decays))
    c['transition_state'] = ts
    torch.save(c, a.dst)

    back = torch.load(a.dst, map_location='cpu', weights_only=False)
    assert back['transition_state']['mt_plateau_best'] == float(a.mt_best), 'did not round-trip'
    assert back['epoch'] == c['epoch'] and back['global_step'] == c['global_step'], 'epoch/step changed'
    assert len(back.get('optimizer_states') or []) == len(c.get('optimizer_states') or []), 'optimizer lost'
    print(f"  {a.src.split('/')[-1]} -> {a.dst.split('/')[-1]}")
    print(f"    epoch={back['epoch']} step={back['global_step']}  was {before}")
    print(f"    now mt_plateau_best={back['transition_state']['mt_plateau_best']:.4f} "
          f"bad={back['transition_state']['mt_plateau_bad']} decays={back['transition_state']['mt_decays']}")


if __name__ == '__main__':
    main()
