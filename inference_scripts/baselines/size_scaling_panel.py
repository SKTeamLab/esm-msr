"""Prepare the shared protein size-scaling panel at exact powers of two: 32, 64, 128, ..., 65,536.

For L <= 2,048, each bin uses a real ProteinGym AF2 structure (and its real ProteinGym .a3m MSA):
exact-length where available (64, 128) or cleanly truncated from the smallest slightly longer real ProteinGym
protein (32, 256, 512, 1024, 2048). For L >= 4,096, valid non-clashing single-chain PDBs are tiled on a
3D lattice from BRCA2_HUMAN so every model can be driven until it runs out of VRAM at minimum batch size (B=1).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# (exact target L, source ProteinGym AF2 structure name)
REAL_POW2_PANEL = [
    (32, 'TCRG1_MOUSE'),      # L_src = 37   -> truncated cleanly to 32
    (64, 'BBC1_YEAST'),       # L_src = 64   -> exact
    (128, 'RL40A_YEAST'),     # L_src = 128  -> exact
    (256, 'CASP3_HUMAN'),     # L_src = 258  -> truncated cleanly to 256
    (512, 'SRC_HUMAN'),       # L_src = 536  -> truncated cleanly to 512
    (1024, 'CAR11_HUMAN'),    # L_src = 1154 -> truncated cleanly to 1024
    (2048, 'POLG_CXB3N'),     # L_src = 2185 -> truncated cleanly to 2048
]

TILED_POW2_LENGTHS = [4096, 8192, 16384, 32768, 65536]

AA3_TO_1 = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'MSE': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q',
    'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
}
ICODES = [' '] + [chr(c) for c in range(ord('A'), ord('Z') + 1)]


def read_chain_a_residues(pdb_path: Path) -> list[tuple[str, list[str]]]:
    """Return ordered list of (aa1, [ATOM lines]) for chain A residues."""
    residues: list[tuple[str, list[str]]] = []
    cur_key = None
    cur_lines: list[str] = []
    cur_aa = 'A'
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith('ATOM  ') or line[21] != 'A':
                continue
            key = line[22:27]
            if key != cur_key:
                if cur_lines:
                    residues.append((cur_aa, cur_lines))
                cur_key = key
                cur_lines = []
                cur_aa = AA3_TO_1.get(line[17:20].strip(), 'A')
            cur_lines.append(line)
    if cur_lines:
        residues.append((cur_aa, cur_lines))
    return residues


def format_atom_line(line: str, serial: int, res_idx_0: int, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> str:
    """Rewrite serial, resSeq (1..9999) + iCode (' ', 'A', ...), chain 'A', and shifted XYZ safely in PDB width."""
    res_num = (res_idx_0 % 9999) + 1
    icode = ICODES[res_idx_0 // 9999]
    x = float(line[30:38]) + dx
    y = float(line[38:46]) + dy
    z = float(line[46:54]) + dz
    s_num = (serial % 99999) + 1
    return (
        f'{line[:6]}{s_num:5d}{line[11:21]}A'
        f'{res_num:4d}{icode}{line[27:30]}'
        f'{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}'
    )


def write_truncated_pdb_and_a3m(
    src_pdb: Path, src_a3m: Path, out_pdb: Path, out_a3m: Path, length: int
) -> tuple[str, int]:
    residues = read_chain_a_residues(src_pdb)[:length]
    assert len(residues) == length, f'Expected >= {length} residues in {src_pdb}, got {len(residues)}'
    seq = ''.join(aa for aa, _ in residues)
    serial = 0
    with open(out_pdb, 'w') as f:
        for r_idx, (_, lines) in enumerate(residues):
            for ln in lines:
                f.write(format_atom_line(ln, serial, r_idx))
                serial += 1
        f.write('TER\nEND\n')

    # Truncate A3M to `length` aligned columns (uppercase + '-'), skipping lowercase insertions past length
    msa_seqs = 0
    if src_a3m.exists():
        with open(src_a3m) as fin, open(out_a3m, 'w') as fout:
            for ln in fin:
                if ln.startswith('>'):
                    fout.write(ln)
                    msa_seqs += 1
                else:
                    s = ln.strip()
                    out_chars = []
                    aligned = 0
                    for ch in s:
                        if ch.islower():
                            if aligned < length:
                                out_chars.append(ch)
                        else:
                            if aligned < length:
                                out_chars.append(ch)
                                aligned += 1
                            else:
                                break
                    fout.write(''.join(out_chars) + '\n')
    return seq, msa_seqs


def write_tiled_pdb_and_a3m(src_pdb: Path, out_pdb: Path, out_a3m: Path, target_len: int) -> str:
    """Tile a real AF2 structure onto a 3D lattice to construct a valid non-clashing chain of exact length target_len."""
    base_res = read_chain_a_residues(src_pdb)
    xs, ys, zs = [], [], []
    for _, lines in base_res:
        for ln in lines:
            if ln[12:16].strip() == 'CA':
                xs.append(float(ln[30:38]))
                ys.append(float(ln[38:46]))
                zs.append(float(ln[46:54]))
    cx, cy, cz = sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs)
    span = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)) + 25.0
    step = min(max(span, 180.0), 350.0)
    n_tiles = math.ceil(target_len / len(base_res))
    grid = max(1, math.ceil(n_tiles ** (1.0 / 3.0)))

    seq_chars = []
    serial = 0
    r_out = 0
    with open(out_pdb, 'w') as f:
        for tile_idx in range(n_tiles):
            ix = tile_idx % grid
            iy = (tile_idx // grid) % grid
            iz = tile_idx // (grid * grid)
            dx = ix * step - cx
            dy = iy * step - cy
            dz = iz * step - cz
            for aa, lines in base_res:
                if r_out >= target_len:
                    break
                seq_chars.append(aa)
                for ln in lines:
                    f.write(format_atom_line(ln, serial, r_out, dx=dx, dy=dy, dz=dz))
                    serial += 1
                r_out += 1
        f.write('TER\nEND\n')

    seq = ''.join(seq_chars)
    with open(out_a3m, 'w') as f:
        f.write(f'>TILED_{target_len}\n{seq}\n')
    return seq


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pgym_pdb_dir', default='/home/sreeves/ProteinGym/ProteinGym_AF2_structures')
    ap.add_argument('--me_msa_dir', default='/home/sreeves/software/MutateEverything-oct06/data/pgym_oct06/msa')
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()

    pg_dir = Path(args.pgym_pdb_dir)
    msa_dir = Path(args.me_msa_dir)
    out_dir = Path(args.out_dir)
    (out_dir / 'pdbs').mkdir(parents=True, exist_ok=True)
    (out_dir / 'a3m').mkdir(parents=True, exist_ok=True)
    (out_dir / 'fasta').mkdir(parents=True, exist_ok=True)

    panel = []
    for target_L, src_name in REAL_POW2_PANEL:
        src_pdb = pg_dir / f'{src_name}.pdb'
        src_a3m = msa_dir / src_name / f'{src_name}.a3m'
        assert src_pdb.exists(), f'Missing {src_pdb}'
        residues = read_chain_a_residues(src_pdb)
        if len(residues) == target_L:
            seq = ''.join(aa for aa, _ in residues)
            msa_n = sum(1 for ln in open(src_a3m) if ln.startswith('>')) if src_a3m.exists() else 0
            entry_name = src_name
            pdb_path, a3m_path = str(src_pdb), (str(src_a3m) if src_a3m.exists() else '')
            source = 'real_pgym'
        else:
            entry_name = f'{src_name}_{target_L}'
            out_pdb = out_dir / 'pdbs' / f'{entry_name}.pdb'
            out_a3m = out_dir / 'a3m' / f'{entry_name}.a3m'
            seq, msa_n = write_truncated_pdb_and_a3m(src_pdb, src_a3m, out_pdb, out_a3m, target_L)
            pdb_path, a3m_path = str(out_pdb), str(out_a3m)
            source = 'truncated_pgym'
        panel.append(dict(
            bin=target_L, name=entry_name, seq_len=target_L, source=source,
            pdb_path=pdb_path, a3m_path=a3m_path, msa_depth=msa_n, seq=seq,
        ))

    brca2_pdb = pg_dir / 'BRCA2_HUMAN.pdb'
    for L in TILED_POW2_LENGTHS:
        t_name = f'TILED_{L}'
        t_pdb = out_dir / 'pdbs' / f'{t_name}.pdb'
        t_a3m = out_dir / 'a3m' / f'{t_name}.a3m'
        t_seq = write_tiled_pdb_and_a3m(brca2_pdb, t_pdb, t_a3m, L)
        panel.append(dict(
            bin=L, name=t_name, seq_len=L, source='tiled_extension',
            pdb_path=str(t_pdb), a3m_path=str(t_a3m), msa_depth=1, seq=t_seq,
        ))

    panel.sort(key=lambda r: r['seq_len'])
    for row in panel:
        fa = out_dir / 'fasta' / f'{row["name"]}.fasta'
        fa.write_text(f'>{row["name"]}\n{row["seq"]}\n')
        row['fasta_path'] = str(fa)
        del row['seq']
    manifest_path = out_dir / 'panel.json'
    manifest_path.write_text(json.dumps(panel, indent=2) + '\n')
    print(f'Wrote {len(panel)} exact power-of-2 panel entries to {manifest_path}')
    for row in panel:
        print(f"  L={row['seq_len']:5d} {row['name']:20s} ({row['source']:15s}) msa={row['msa_depth']}")


if __name__ == '__main__':
    main()
