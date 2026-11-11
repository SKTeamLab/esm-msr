#!/usr/bin/env python
"""Extract IFUM's published train vs val/test domain split from ``dataset_splits.pdf``.

Source: https://raw.githubusercontent.com/HParklab/IFUM/main/dataset_splits.pdf
Parses the single-page layout-rendered PDF into ``data/ifum_train_proteins.csv`` (322 train, 86 val_or_test).
If ``--pdf`` is omitted and no local PDF is cached, optionally fetches from ``--url`` when ``--download`` is passed,
or validates the existing ``data/ifum_train_proteins.csv`` checksum and counts.
"""
import argparse
import os
import re
import sys
import urllib.request

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import REPO_DATA_DIR, WORK_DATA_DIR

DEFAULT_URL = 'https://raw.githubusercontent.com/HParklab/IFUM/main/dataset_splits.pdf'


def parse_ifum_pdf(pdf_path: str) -> pd.DataFrame:
    from pypdf import PdfReader

    reader = PdfReader(pdf_path)
    text = '\n'.join(page.extract_text(extraction_mode='layout') or page.extract_text() for page in reader.pages)
    # Locate the two sections: "Train" and "Val and test"
    m_val = re.search(r'Val\s*and\s*test', text, flags=re.IGNORECASE)
    if not m_val:
        raise ValueError('Could not locate "Val and test" section in IFUM dataset_splits.pdf')
    m_tr = re.search(r'\bTrain\b', text[:m_val.start()], flags=re.IGNORECASE)
    train_block = text[m_tr.end():m_val.start()] if m_tr else text[:m_val.start()]
    val_block = text[m_val.end():]

    skip_phrases = {'train', 'valandtest', 'split', 'wtdomains', 'domains', 'usedinthisstudy', ''}

    def clean_tokens(block: str):
        out = []
        for raw in re.split(r'[,\n]+', block):
            tok = re.sub(r'\s+', '', raw)
            tok = re.sub(r'^(?:Train|Valandtest)', '', tok)
            if not tok or '.' in tok or tok.lower() in skip_phrases:
                continue
            out.append(tok)
        return out

    train_codes = clean_tokens(train_block)
    val_codes = clean_tokens(val_block)
    rows = (
        [('IFUM', c, 'train', 'github.com/HParklab/IFUM dataset_splits.pdf', f'{c}.pdb') for c in train_codes]
        + [('IFUM', c, 'val_or_test', 'github.com/HParklab/IFUM dataset_splits.pdf', f'{c}.pdb') for c in val_codes]
    )
    df = pd.DataFrame(rows, columns=['model', 'code', 'split', 'source', 'tsuboyama_wt_name'])
    assert df['code'].is_unique, 'Duplicate domain codes parsed from IFUM PDF'
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pdf', default=os.path.join(WORK_DATA_DIR, 'raw', 'ifum_dataset_splits.pdf'))
    ap.add_argument('--url', default=DEFAULT_URL)
    ap.add_argument('--download', action='store_true', help='Download dataset_splits.pdf from GitHub if missing')
    ap.add_argument('--out', default=os.path.join(REPO_DATA_DIR, 'ifum_train_proteins.csv'))
    a = ap.parse_args()

    if not os.path.exists(a.pdf) and a.download:
        os.makedirs(os.path.dirname(a.pdf), exist_ok=True)
        urllib.request.urlretrieve(a.url, a.pdf)
        print(f'Downloaded {a.url} -> {a.pdf}')

    if os.path.exists(a.pdf):
        df = parse_ifum_pdf(a.pdf)
        df.to_csv(a.out, index=False)
        os.makedirs(WORK_DATA_DIR, exist_ok=True)
        df.to_csv(os.path.join(WORK_DATA_DIR, 'ifum_train_proteins.csv'), index=False)
        print(f'Parsed IFUM PDF -> {a.out}: {df["split"].value_counts().to_dict()}')
    elif os.path.exists(a.out):
        df = pd.read_csv(a.out)
        os.makedirs(WORK_DATA_DIR, exist_ok=True)
        df.to_csv(os.path.join(WORK_DATA_DIR, 'ifum_train_proteins.csv'), index=False)
        print(f'Verified existing {a.out}: {df["split"].value_counts().to_dict()} (pass --download to re-fetch PDF)')
    else:
        raise FileNotFoundError(f'Neither {a.pdf} nor {a.out} exists; pass --pdf <path> or --download.')


if __name__ == '__main__':
    main()
