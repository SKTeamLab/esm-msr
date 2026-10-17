"""Build the training caches (train/val/benchmarks) and stop, without training. Takes the same arguments as training.py."""
import logging
from esm.tokenization import EsmSequenceTokenizer
from esm.pretrained import ESM3_structure_encoder_v0
from esm_msr.config import parse_arguments
from esm_msr.preprocess_megascale import setup_dataloaders

args = parse_arguments()
train, val, bench, tn, vn, bn = setup_dataloaders(args, EsmSequenceTokenizer("cpu"), ESM3_structure_encoder_v0("cpu"), add_benchmarks_to_val=True)
logging.info(f"train loaders {len(tn)}, val loaders {len(vn)}, benchmark loaders {bn}")
