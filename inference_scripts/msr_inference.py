import argparse
import subprocess
import sys
import os
import queue
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from esm_msr import preprocessing

def main():
    parser = argparse.ArgumentParser(description="Process dataset and run inference.")
    
    parser.add_argument("--split", type=str, default=None, help="Name of the split to use.")
    parser.add_argument("--num_gpus", type=int, default=1, 
                        help="Number of GPUs to distribute jobs across.")
    
    # Paths for preprocessing
    parser.add_argument("--data_file", type=str, default="data/tsuboyama/Tsuboyama2023_Dataset2_Dataset3_20230416.csv")
    parser.add_argument("--af_model_folder", type=str, default="data/tsuboyama/AlphaFold_model_PDBs")
    
    # Inference args
    parser.add_argument("--checkpoint_path", type=str, default=None) #"logs/msr_dual_ensemble_small_large/1/epoch=02-val_rho_combined_avg=0.810.ckpt"
    parser.add_argument("--hparams_path", type=str, default="logs/msr_dual_ensemble_small_large_mm/1/hparams.yaml") #"logs/msr_dual_ensemble_small_large_mm/1/hparams_zs.yaml"
    parser.add_argument("--output_dir", type=str, default="data/inference_zs")
    parser.add_argument("--mode", type=str, default="both")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--distance_threshold", type=float, default=6.0)
    parser.add_argument("--hf_token", type=str, default="")
    parser.add_argument("--model_dtype", type=str, default="float32")
    parser.add_argument("--adapter_mode", type=str, default="dual")
    parser.add_argument("--lora_mode", type=str, default="ensemble")
    parser.add_argument("--quaternary_mode", type=str, default="single_chain")
    parser.add_argument("--base_model_loc", type=str, default="")

    args = parser.parse_args()

    REPO_ROOT = Path(__file__).resolve().parent.parent
    split_file = REPO_ROOT / "data" / f"{args.split}.pkl"
    
    print(f"Loading data from {args.data_file}...")
    ds = preprocessing.MegaScaleDatasetPreprocessor(
        data_file=os.path.join(REPO_ROOT, args.data_file), 
        af_model_folder=os.path.join(REPO_ROOT, args.af_model_folder)
    )
    
    print(f"Creating training splits at {split_file}...")
    _ = ds.create_training_splits(str(split_file), -1)

    assert 'test' in ds.split_dfs, "The 'test' key is missing from ds.split_dfs after creating splits."
    data_scaffold = ds.split_dfs['test']
    data_scaffold['mut_structure'] = data_scaffold['mut_structure'].fillna('-')                

    required_cols = ['code', 'chain', 'mut_structure', 'pdb_file']
    for col in required_cols:
        assert col in data_scaffold.columns, f"Required column '{col}' is missing from the test split data."

    output_dir = Path(os.path.join(REPO_ROOT, args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped_data = list(data_scaffold.groupby(required_cols))
    print(f"Found {len(grouped_data)} groups. Starting inference pool on {args.num_gpus} GPUs...")

    # Initialize a thread-safe queue containing available GPU IDs (0 to num_gpus - 1)
    gpu_queue = queue.Queue()
    for i in range(args.num_gpus):
        gpu_queue.put(i)

    # Base environment dictionary to pass to subprocesses
    base_env = os.environ.copy()

    def worker(job_info):
        """Worker function to execute inference on an assigned GPU."""
        group_keys, _ = job_info
        code, chain, mut_structure, pdb_file = group_keys
        
        bb_mut = mut_structure if mut_structure != '-' else None
        output_csv = REPO_ROOT / output_dir / f"{code}_{chain}_inference_dist_{int(args.distance_threshold)}.csv"
        
        cmd = [
            sys.executable, os.path.join(REPO_ROOT, "src/esm_msr/inference.py"),
            "--output_csv", str(output_csv),
            "--pdb_file", str(pdb_file),
            "--code", str(code),
            "--chain", str(chain),
            "--mode", args.mode,
            "--batch_size", str(args.batch_size),
            "--device", args.device, # Will just map to the localized 'cuda:0'
            "--distance_threshold", str(args.distance_threshold),
            "--hf_token", args.hf_token,
            "--hparams_path", os.path.join(REPO_ROOT, args.hparams_path),
            "--model_dtype", args.model_dtype,
            "--adapter_mode", args.adapter_mode,
            "--lora_mode", args.lora_mode,
            "--quaternary_mode", args.quaternary_mode,
            "--base_model_loc", args.base_model_loc,
            "--calculate_distance"
        ]

        if args.checkpoint_path:
            cmd += ["--checkpoint_path", os.path.join(REPO_ROOT, args.checkpoint_path)]
        if bb_mut:
            cmd += ["--backbone_mutation", str(bb_mut)]

        # Block until a GPU becomes available
        gpu_id = gpu_queue.get()
        try:
            # Isolate the subprocess so it only sees the assigned GPU
            local_env = base_env.copy()
            local_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            
            print(f"\n[GPU {gpu_id}] Running inference for code: {code}, chain: {chain}...")
            
            try:
                subprocess.run(cmd, env=local_env, check=True)
            except subprocess.CalledProcessError as e:
                print(f"[GPU {gpu_id}] Error: inference.py failed for {code}_{chain} with return code {e.returncode}.")
                raise AssertionError(f"Inference subroutine failed on code {code}, chain {chain}") from e
        finally:
            # Ensure the GPU is returned to the pool even if the process fails
            gpu_queue.put(gpu_id)

    # Execute jobs using a ThreadPool. max_workers matches num_gpus to ensure we 
    # don't spin up idle threads, though the queue acts as the true throttle.
    with ThreadPoolExecutor(max_workers=args.num_gpus) as executor:
        futures = {executor.submit(worker, job): job for job in grouped_data}
        
        for future in as_completed(futures):
            # Calling .result() will raise any AssertionErrors caught inside the worker
            future.result() 

if __name__ == "__main__":
    main()