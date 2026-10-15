"""Build the per-library item cache without starting a training run.

Cache generation never needed the GPU: training.py creates the structure encoder with
``ESM3_structure_encoder_v0("cpu")``, so the whole thing is CPU work. This calls the same
``setup_dataloaders`` the trainer calls and then exits, so the cache a run would build lazily on startup can be
built ahead of time - and built while another run has the card.

Takes the training CLI verbatim; pass the run's own data flags plus --regenerate_cache so the content matches what
that run would have produced. CUDA is hidden so an accidental device init cannot contend with a live job.
"""
import logging
import os
import sys

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

from esm.pretrained import ESM3_structure_encoder_v0
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

from esm_msr.config import parse_arguments
from esm_msr.preprocess_megascale import setup_dataloaders


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    args = parse_arguments()
    logging.info(f"cache_path={args.cache_path}  regenerate={args.regenerate_cache}  "
                 f"censor_range={getattr(args, 'censor_range', None)}  score_column={args.score_column}")
    tokenizer = EsmSequenceTokenizer("cpu")
    structure_encoder = ESM3_structure_encoder_v0("cpu")
    train, val, bench, tn, vn, bn = setup_dataloaders(args, tokenizer, structure_encoder, add_benchmarks_to_val=True)
    logging.info(f"done: {len(tn)} train loader(s), {len(vn)} val libraries, {len(bn)} benchmark loaders")


if __name__ == '__main__':
    sys.exit(main())
