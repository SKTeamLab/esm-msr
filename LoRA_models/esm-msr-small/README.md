# esm-msr-small

The small ESM-MSR model shipped with the repository: three training seeds of the release recipe with rank-2 adapters on
both passes (WT rank 2 / alpha 4, MT rank 2 / alpha 4; run FINAL_p0_r2a4), trained on the homology-restricted MegaScale
split (`data/splits_oct06_capped.pkl`), both passes unmasked. The full model (MT rank 16) is on
[HuggingFace](https://huggingface.co/sareeves96/esm-msr).

| File | Seed | Epoch | Validation ρ (combined) | SHA-256 |
|---|---|---|---|---|
| `lora_seed1.safetensors` | 1 | 5 | 0.8504 | `b38434473bef39e7ef9188d857e271d41fdc77f40cb4f5efb92f55d35aeb3929` |
| `lora_seed2.safetensors` | 2 | 5 | 0.8490 | `a1ed2fc61ad6eb7f3b232a5674f7ebb85af2486cca0fda78fc445816d10cdb84` |
| `lora_seed3.safetensors` | 3 | 6 | 0.8545 | `7df0ed9d401d577ec3b07907507e000a55aeb9ac9b7f63f42ed62c739b87d996` |

`hparams.yaml` is the shared training configuration (the seeds differ only in `seed`); inference reads it from this folder.
Each file holds the WT and MT adapters and the four calibration scalars (float32, 588 tensors), converted from the training
checkpoints with `inference_scripts/convert_lora_checkpoint.py`. The tensors equal the checkpoints' bit for bit, and
predictions from either form are identical.
