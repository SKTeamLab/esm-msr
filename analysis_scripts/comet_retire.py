"""
Retire or label a Comet experiment without deleting it. Comet has no per-metric overwrite and its only destructive
operation removes the whole experiment, irreversibly, so a run that was abandoned, superseded or invalidated keeps
all of its curves and stays comparable; a tag, and optionally a name prefix, is what says not to read it as a
current result.

  # label only: add tags, leave the name alone
  PYTHONPATH=src python analysis_scripts/comet_retire.py --tag pre_rank_fix KEY_OR_URL [KEY_OR_URL ...]

  # retire: add a reason tag AND prefix the name
  PYTHONPATH=src python analysis_scripts/comet_retire.py --reason aborted KEY_OR_URL
  PYTHONPATH=src python analysis_scripts/comet_retire.py --reason superseded --tag pre-5352574 KEY

  # find things: lists ARCHIVED experiments too, which the default Comet listing hides
  PYTHONPATH=src python analysis_scripts/comet_retire.py --list-project esm-msr-agent-oct06-capped [--match '^dc[1-7]']

--reason is both a tag and an uppercase name prefix; --tag adds tags only. Re-running is safe: an already-prefixed
name is left alone and tags never duplicate. Nothing here archives: archiving hides a run from the default view,
which is the opposite of keeping it as a reference.
"""
import argparse
import re
import sys

import comet_ml

# The key is NOT in this file: comet_ml.API() reads COMET_API_KEY or ~/.comet.config (it used to be hard-coded here, and so sat in git history).
WORKSPACE = 'sareeves96'
REASONS = ('aborted', 'superseded', 'invalid', 'confounded', 'reference')
KEY = re.compile(r'([0-9a-f]{32})')


def key_of(s):
    m = KEY.search(s)
    if not m:
        sys.exit(f"not an experiment key or Comet URL: {s}")
    return m.group(1)


def listing(api, project, match=None):
    """Every experiment in the project, archived ones included.

    api.get_experiments() and the REST endpoint's default both return only the UNARCHIVED set, silently - which is
    how dc1 and dc2 came to look deleted when they were merely archived. Both pages are read here.
    """
    pat = re.compile(match, re.I) if match else None
    out = []
    for archived in (False, True):
        r = api._client.get_from_endpoint('experiments', {'workspaceName': WORKSPACE, 'projectName': project,
                                                          'archived': archived})
        for e in r['experiments']:
            name = e.get('experimentName') or ''
            if pat is None or pat.search(name):
                out.append((e['experimentKey'], name, archived))
    return sorted(out, key=lambda x: (x[1].lower(), x[0]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('keys', nargs='*', help='experiment keys, or Comet URLs containing them')
    p.add_argument('--reason', choices=REASONS, help='tag AND uppercase name prefix')
    p.add_argument('--tag', action='append', default=[], help='add a tag only (repeatable)')
    p.add_argument('--list-project', help='list a project (archived included) and exit')
    p.add_argument('--match', help='with --list-project: only names matching this regex')
    a = p.parse_args()
    api = comet_ml.API()

    if a.list_project:
        rows = listing(api, a.list_project, a.match)
        print(f"{len(rows)} experiments:")
        for k, n, arch in rows:
            print(f"  {k}  archived={str(arch):5s}  {n}")
        return
    if not a.keys or not (a.reason or a.tag):
        p.error('keys plus --reason and/or --tag are required unless --list-project is given')

    want = set(a.tag) | ({a.reason} if a.reason else set())
    prefix = (a.reason.upper() + '_') if a.reason else None
    for raw in a.keys:
        k = key_of(raw)
        e = api.get_experiment_by_key(k)
        if e is None:
            print(f"  {k}: NOT FOUND (deleted experiments cannot be recovered)")
            continue
        name = e.get_name() or ''
        new_tags = sorted(want - set(e.get_tags() or ()))
        if new_tags:
            e.add_tags(new_tags)
        new_name = name
        if prefix and not name.startswith(prefix):
            new_name = prefix + name
            e.set_name(new_name)
        change = f"name {name!r} -> {new_name!r}; " if new_name != name else ""
        print(f"  {k}  {name[:52]:52s}  {change}tags += {new_tags or 'none (already set)'}")


if __name__ == '__main__':
    main()
