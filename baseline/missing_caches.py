"""Prints the number of libraries (train + val) of the capped split whose cache file is missing from cache_main_capped (0 = ready)."""
import logging, os
logging.disable(logging.CRITICAL)
from esm_msr.preprocess_megascale import MegaScaleDatasetPreprocessor
R = '/home/sareeves/software/esm-msr/data/tsuboyama/'
pp = MegaScaleDatasetPreprocessor(R + 'Tsuboyama2023_Dataset2_Dataset3_20230416.csv', R + 'AlphaFold_model_PDBs', spurs_override=False)
pp.create_training_splits('/home/sareeves/playground/esm-msr-devel/repo/data/splits_oct06_capped.pkl', -1)
miss = [c for sc in ('train', 'val') for c in pp.split_dfs[sc]['code'].unique() if not os.path.exists(f'/home/sareeves/playground/esm-msr-devel/cache_main_capped/{c}_ddG_ML_MAX_Smasked0.pkl')]
print(len(miss))
