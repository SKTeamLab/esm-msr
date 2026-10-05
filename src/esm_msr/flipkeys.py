"""Parsing of flip-column keys.

A flip column is one scored position with one fixed partner identity: ``code|scored position|partner position + partner residue``
(for example ``1AOY|47|55K``). Columns that share the scored position AND the partner position form one *position-pair matrix*:
rows are the scored substitutions, columns the partner residues. Native-conditional keys (``code|pos|native``) have no partner position
and are their own group.
"""
import re
from typing import Optional, Tuple

_PARTNER_RE = re.compile(r'^(\d+)(.+)$')


def split_flip_key(key: str) -> Tuple[Optional[str], Optional[str]]:
    """``(pair_key, partner_residue)``, or ``(key, None)`` when the key has no partner position. ``('', None)`` for no key."""
    if not key:
        return '', None
    parts = str(key).split('|')
    if len(parts) < 3:
        return str(key), None
    m = _PARTNER_RE.match(parts[2])
    if not m:
        return str(key), None
    return '|'.join(parts[:2]) + '|' + m.group(1), m.group(2)
