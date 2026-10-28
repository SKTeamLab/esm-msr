"""Index existing ColabFold a3m files by their exact query sequence (first record), so a library reuses an MSA
only when the MSA was built for exactly its sequence."""
import glob
import json
import sys


def query_of(path):
    with open(path) as f:
        seq, started = [], False
        for line in f:
            if line.startswith('#'):
                continue
            if line.startswith('>'):
                if started:
                    break
                started = True
                continue
            if started:
                seq.append(line.strip())
    return ''.join(seq)


if __name__ == '__main__':
    out = sys.argv[1]
    roots = sys.argv[2:]
    index = {}
    for root in roots:
        for p in sorted(glob.glob(f'{root}/**/*.a3m', recursive=True)):
            try:
                q = query_of(p)
            except (OSError, UnicodeDecodeError):
                continue
            if q and q not in index:
                index[q] = p
    with open(out, 'w') as f:
        json.dump(index, f)
    print(f'{len(index)} distinct query sequences indexed')
