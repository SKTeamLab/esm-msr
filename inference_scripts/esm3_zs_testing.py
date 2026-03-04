import os
import asyncio
import torch
import pandas as pd
import numpy as np
import argparse
from collections import deque, defaultdict
from datetime import datetime, timedelta
from tqdm import tqdm
import time

from scipy.stats import entropy
from scipy.special import softmax

from esm_msr import utils

# Import Forge Client SDK components
from esm.sdk.api import ESMProtein, LogitsConfig
from esm.utils.structure.protein_chain import ProteinChain
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
from esm.sdk.forge import ESM3ForgeInferenceClient

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "preprocessed"
MODEL_DIR = REPO_ROOT / "models"

def timed_call(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed

class RateLimiter:
    """Async-friendly Rate Limiter using a sliding window."""
    def __init__(self, max_requests=50, time_window=1.0):
        self.max_requests = max_requests
        self.time_window = time_window
        self.request_times = deque()
    
    async def wait(self):
        """Waits until a slot is available in the rate limit window."""
        while True:
            now = datetime.now()
            while self.request_times and (now - self.request_times[0]) > timedelta(seconds=self.time_window):
                self.request_times.popleft()
            if len(self.request_times) < self.max_requests:
                self.request_times.append(now)
                return
            await asyncio.sleep(0.1)

class ESM3ForgePredictor:
    def __init__(self, client, tokenizer, max_concurrency=10, rate_limit_reqs=250, rate_limit_window=60):
        self.client = client
        self.tokenizer = tokenizer
        self.sem = asyncio.Semaphore(max_concurrency)
        self.rate_limiter = RateLimiter(max_requests=rate_limit_reqs, time_window=rate_limit_window)

        if hasattr(self.tokenizer, 'get_vocab'):
            self.vocab = self.tokenizer.get_vocab()
        elif hasattr(self.tokenizer, 'vocab'):
            self.vocab = self.tokenizer.vocab
        else:
            raise ValueError("Tokenizer must have .get_vocab() or .vocab attribute")

    async def _safe_api_call(self, func, *args, **kwargs):
        """Wraps synchronous API calls in a thread with rate limiting."""
        async with self.sem:
            await self.rate_limiter.wait()
            try:
                return await asyncio.to_thread(func, *args, **kwargs)
            except Exception as e:
                print(f"API Error: {e}")
                return None

    async def _get_logits_for_sequence(self, sequence, coords):
        """
        Helper: Encodes a sequence (with mask) and returns logits.
        """
        protein_input = ESMProtein(sequence=sequence, coordinates=coords)
        
        # 1. Encode
        encoded_tensor = await self._safe_api_call(self.client.encode, protein_input)
        if encoded_tensor is None: return None

        # 2. Logits
        config = LogitsConfig(sequence=True, structure=False)
        logits_output = await self._safe_api_call(self.client.logits, encoded_tensor, config)
        if logits_output is None: return None

        # 3. Extract
        seq_logits = logits_output.logits.sequence
        if seq_logits.ndim == 3:
            seq_logits = seq_logits[0]
            
        return seq_logits.cpu()

    def infer_all_singles(self, pdb_path: str, subset_df: pd.DataFrame, chain: str = "A", backbone_mutation = None):
        """
        Computes Zero-Shot scores for single mutants.
        subset_df cols: ['wild_type', 'seq_pos', 'mutation']
        """
        # Validate DF
        req_cols = {'wild_type', 'seq_pos', 'mutation'}
        if not req_cols.issubset(subset_df.columns):
            raise ValueError(f"subset_df missing columns: {req_cols}")

        return self._run_async(self._infer_singles_async(pdb_path, chain, subset_df, backbone_mutation))

    def infer_all_doubles(self, pdb_path: str, subset_df: pd.DataFrame, chain: str = "A", backbone_mutation = None):
        """
        Computes Zero-Shot scores for double mutants using Chain Rule Average.
        subset_df cols: ['wt1', 'pos1', 'mut1', 'wt2', 'pos2', 'mut2']
        """
        # Validate DF
        req_cols = {'wt1', 'pos1', 'mut1', 'wt2', 'pos2', 'mut2'}
        if not req_cols.issubset(subset_df.columns):
            raise ValueError(f"subset_df missing columns: {req_cols}")

        return self._run_async(self._infer_doubles_async(pdb_path, chain, subset_df, backbone_mutation))

    def _run_async(self, coroutine):
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
        
        return loop.run_until_complete(coroutine)

    # --------------------------------------------------------------------------
    # Async Logic
    # --------------------------------------------------------------------------

    async def _infer_singles_async(self, pdb_path, chain_id, subset_df, backbone_mutation):
        # Load Structure
        chain_obj = ProteinChain.from_pdb(pdb_path, chain_id)
        wt_seq = chain_obj.sequence
        coords = torch.tensor(chain_obj.atom37_positions)

        if backbone_mutation:
            wt = backbone_mutation[0]
            pos = int(backbone_mutation[1:-1])
            mut = backbone_mutation[-1]
            assert wt_seq[pos-1] == wt
            wt_seq = list(wt_seq)
            wt_seq[pos-1] = mut
            wt_seq = ''.join(wt_seq)

        # Group by position to minimize API calls
        tasks_map = defaultdict(list)
        for idx, row in subset_df.iterrows():
            p = int(row['seq_pos'])
            if 1 <= p <= len(wt_seq):
                tasks_map[p].append((row['mutation'], idx))

        # Queue Requests
        unique_positions = list(tasks_map.keys())
        tasks = []
        print(f"Singles: Queueing {len(unique_positions)} masked requests...")
        
        for p in unique_positions:
            masked_seq = wt_seq[:p-1] + "_" + wt_seq[p:]
            tasks.append(self._get_logits_for_sequence(masked_seq, coords))

        results_list = await asyncio.gather(*tasks)

        # Map Results
        out_df = subset_df.copy()
        out_df['esm3_score'] = np.nan
        out_df['esm3_logits_entropy'] = np.nan

        for pos, logits in zip(unique_positions, results_list):
            if logits is None: continue
            
            token_idx = pos # 0-based index + 1 for BOS = pos
            wt_aa = wt_seq[pos-1]
            wt_id = self.vocab.get(wt_aa)
            
            if wt_id is None: continue
            wt_score = logits[token_idx, wt_id].item()

            for mut_aa, df_idx in tasks_map[pos]:
                mut_id = self.vocab.get(mut_aa)
                if mut_id is not None:
                    mut_score = logits[token_idx, mut_id].item()
                    out_df.at[df_idx, 'esm3_score'] = mut_score - wt_score
                    probs = softmax(logits[token_idx, :].float().cpu().numpy())
                    out_df.at[df_idx, 'esm3_logits_entropy'] = entropy(probs, base=2)
        
        return out_df

    async def _infer_doubles_async(self, pdb_path, chain_id, subset_df, backbone_mutation):
        # 1. Setup
        chain_obj = ProteinChain.from_pdb(pdb_path, chain_id)
        wt_seq = chain_obj.sequence
        coords = torch.tensor(chain_obj.atom37_positions)

        if backbone_mutation:
            wt = backbone_mutation[0]
            pos = int(backbone_mutation[1:-1])
            mut = backbone_mutation[-1]
            assert wt_seq[pos-1] == wt
            wt_seq = list(wt_seq)
            wt_seq[pos-1] = mut
            wt_seq = ''.join(wt_seq)

        # 2. Identify Unique Queries Needed
        # We need:
        #  - Singles: Mask at pos1 (WT bkg), Mask at pos2 (WT bkg)
        #  - Contexts: Mask at pos2 (mut1 bkg), Mask at pos1 (mut2 bkg)
        
        singles_queries = set() # Set of 1-based positions
        context_queries = set() # Set of (ctx_pos, ctx_mut, target_pos)

        for _, row in subset_df.iterrows():
            p1, m1 = int(row['pos1']), row['mut1']
            p2, m2 = int(row['pos2']), row['mut2']
            
            # Singles needed
            singles_queries.add(p1)
            singles_queries.add(p2)
            
            # Contexts needed
            # "Mask p2 given m1 at p1"
            context_queries.add((p1, m1, p2))
            # "Mask p1 given m2 at p2"
            context_queries.add((p2, m2, p1))

        # 3. Run Single Mutant Queries
        sorted_singles = sorted(list(singles_queries))
        print(f"Doubles: Queueing {len(sorted_singles)} single-mutant mask requests...")
        
        single_tasks = []
        for p in sorted_singles:
            masked_seq = wt_seq[:p-1] + "_" + wt_seq[p:]
            single_tasks.append(self._get_logits_for_sequence(masked_seq, coords))
        
        single_results = await asyncio.gather(*single_tasks)
        
        # Cache Singles: {pos: logits}
        singles_cache = {p: res for p, res in zip(sorted_singles, single_results) if res is not None}

        # 4. Run Context Mutant Queries
        sorted_contexts = sorted(list(context_queries))
        print(f"Doubles: Queueing {len(sorted_contexts)} context-mutant mask requests...")
        
        context_tasks = []
        for (ctx_p, ctx_m, tgt_p) in sorted_contexts:
            # Construct sequence: WT with ctx_m at ctx_p, and _ at tgt_p
            # 0-based conversion
            seq_list = list(wt_seq)
            seq_list[ctx_p-1] = ctx_m
            seq_list[tgt_p-1] = "_"
            masked_seq = "".join(seq_list)
            context_tasks.append(self._get_logits_for_sequence(masked_seq, coords))

        context_results = await asyncio.gather(*context_tasks)

        # Cache Contexts: {(ctx_p, ctx_m, tgt_p): logits}
        context_cache = {k: res for k, res in zip(sorted_contexts, context_results) if res is not None}

        # 5. Compute Scores
        out_df = subset_df.copy()
        out_df['pred_additive'] = np.nan
        out_df['esm3_score'] = np.nan

        for idx, row in out_df.iterrows():
            p1, m1, wt1 = int(row['pos1']), row['mut1'], row['wt1']
            p2, m2, wt2 = int(row['pos2']), row['mut2'], row['wt2']
            
            # Helper to extract delta from logits
            def get_delta(logits, pos, wt, mut):
                if logits is None: return np.nan
                t_idx = pos
                wt_id = self.vocab.get(wt)
                mut_id = self.vocab.get(mut)
                if wt_id is None or mut_id is None: return np.nan
                return logits[t_idx, mut_id].item() - logits[t_idx, wt_id].item()

            # Retrieve Logits
            logits_single_1 = singles_cache.get(p1)
            logits_single_2 = singles_cache.get(p2)
            logits_ctx_2_given_1 = context_cache.get((p1, m1, p2))
            logits_ctx_1_given_2 = context_cache.get((p2, m2, p1))

            # Calculate Deltas
            # d1: M1 on WT
            d1 = get_delta(logits_single_1, p1, wt1, m1)
            # d2: M2 on WT
            d2 = get_delta(logits_single_2, p2, wt2, m2)
            
            # d2_1: M2 on (M1 background) -> WT at p2 is still wt2
            d2_1 = get_delta(logits_ctx_2_given_1, p2, wt2, m2)
            # d1_2: M1 on (M2 background) -> WT at p1 is still wt1
            d1_2 = get_delta(logits_ctx_1_given_2, p1, wt1, m1)

            if np.isnan([d1, d2]).any():
                continue

            # Additive
            out_df.at[idx, 'pred_additive'] = d1 + d2

            # Chain Rule
            if not np.isnan([d2_1, d1_2]).any():
                path1 = d1 + d2_1
                path2 = d2 + d1_2
                out_df.at[idx, 'esm3_score'] = 0.5 * (path1 + path2)

            out_df['esm3_epistasis_score'] = out_df['esm3_score'] - out_df['pred_additive']
        
        return out_df
    
    def infer_multimutants_sampled(
            self,
            pdb_path: str,
            subset_df: pd.DataFrame,
            chain: str = "A",
            K_paths: int = 4,
            return_path_summaries: bool = False
        ):
            """
            Estimate Δ for multi-mutants by sampling K random single-mutation paths.
            Geometry is held fixed to WT coordinates for all steps.

            Args:
                pdb_path: Path to PDB file
                subset_df: DataFrame with columns wt1,pos1,mut1 ... wt10,pos10,mut10
                chain: Chain ID
                K_paths: Number of random paths to sample
                return_path_summaries: Include list of individual path scores
            """
            import random

            # Helper to parse columns
            def _get_muts_from_row(row):
                muts = []
                for k in range(1, 11):
                    wt_c, pos_c, mut_c = f"wt{k}", f"pos{k}", f"mut{k}"
                    if wt_c in row and pos_c in row and mut_c in row:
                        wt, pos, mut = row[wt_c], row[pos_c], row[mut_c]
                        if pd.notna(wt) and pd.notna(pos) and pd.notna(mut):
                            muts.append((wt, int(pos), mut))
                return muts

            def _canonical_mut_string(muts):
                return ':'.join([f"{wt}{pos}{mut}" for (wt,pos,mut) in muts])

            # --- 1. Load Structure ---
            # Note: We ensure coords are standard (N, CA, C) for the client if possible
            chain_obj = ProteinChain.from_pdb(pdb_path, chain)
            wt_seq = chain_obj.sequence
            
            # Handle coordinates: ensure we have (L, 3, 3) if using atom37
            coords_raw = torch.tensor(chain_obj.atom37_positions)
            if coords_raw.shape[1] == 37:
                coords = coords_raw[:, :3, :]
            else:
                coords = coords_raw
            
            # --- 2. Build Work Items ---
            
            # Part A: Path Sampling Items (Context Dependent)
            path_work_items = [] # (row_idx, path_idx, step_idx, masked_seq, target_pos, target_mut, background_aa)
            
            # Part B: Additive Baseline Items (WT Dependent)
            # We track unique positions to avoid redundant API calls for the baseline
            unique_additive_positions = set()
            
            row_results = [] # To store final metadata

            for row_idx, row in subset_df.iterrows():
                muts = _get_muts_from_row(row)
                if not muts: continue
                
                N = len(muts)
                
                # Setup metadata for this row
                row_results.append({
                    "pdb": os.path.basename(pdb_path),
                    "chain_id": chain,
                    "N": N,
                    "K_paths": K_paths,
                    "mut_type": _canonical_mut_string(muts),
                    "path_scores": [0.0] * K_paths,
                    "mutations_list": muts, # Store for additive calc later
                    "pred_additive": 0.0
                })
                result_idx = len(row_results) - 1

                # 2a. Register positions needed for additive baseline
                for fr, pos, to in muts:
                    unique_additive_positions.add(pos)

                # 2b. Generate K random paths
                for k in range(K_paths):
                    path_order = muts.copy()
                    if N > 1:
                        random.shuffle(path_order)
                    
                    current_seq_list = list(wt_seq)
                    
                    for step_idx, (fr, pos, to) in enumerate(path_order):
                        p_idx = pos - 1
                        
                        # Prepare Masked Sequence (mask target in CURRENT context)
                        masked_seq_list = current_seq_list.copy()
                        masked_seq_list[p_idx] = "_"
                        masked_seq_str = "".join(masked_seq_list)
                        
                        bg_aa = current_seq_list[p_idx]
                        
                        path_work_items.append({
                            "row_idx": result_idx,
                            "path_idx": k,
                            "masked_seq": masked_seq_str,
                            "pos_1based": pos,
                            "target_mut": to,
                            "background_aa": bg_aa
                        })
                        
                        # Update context
                        current_seq_list[p_idx] = to

            # --- 3. Execute Async Requests ---
            
            # Prepare Additive Tasks (One per unique position)
            additive_tasks_list = sorted(list(unique_additive_positions))
            additive_work_items = []
            
            for pos in additive_tasks_list:
                p_idx = pos - 1
                masked_wt_list = list(wt_seq)
                masked_wt_list[p_idx] = "_"
                masked_wt_str = "".join(masked_wt_list)
                additive_work_items.append(masked_wt_str)

            async def process_all():
                tasks = []
                
                # Queue Additive Tasks
                for seq in additive_work_items:
                    tasks.append(self._get_logits_for_sequence(seq, coords))
                
                # Queue Path Tasks
                for item in path_work_items:
                    tasks.append(self._get_logits_for_sequence(item['masked_seq'], coords))
                
                print(f"Sampling: Queueing {len(tasks)} requests ({len(additive_work_items)} additive + {len(path_work_items)} path steps)...")
                return await asyncio.gather(*tasks)

            all_results = self._run_async(process_all())

            # Split results
            num_additive = len(additive_work_items)
            additive_logits = all_results[:num_additive]
            path_logits = all_results[num_additive:]

            # --- 4. Process Additive Baseline ---
            # Cache: pos -> logits
            additive_cache = {}
            for pos, logits in zip(additive_tasks_list, additive_logits):
                if logits is not None:
                    additive_cache[pos] = logits

            # Compute additive sum for each row
            for r_res in row_results:
                sum_additive = 0.0
                valid_additive = True
                
                for fr, pos, to in r_res['mutations_list']:
                    if pos not in additive_cache:
                        valid_additive = False; break
                    
                    logits = additive_cache[pos]
                    token_idx = pos
                    
                    mut_id = self.vocab.get(to)
                    wt_id = self.vocab.get(fr) # Should match WT seq
                    
                    if mut_id is None or wt_id is None:
                        valid_additive = False; break
                        
                    # LLR: P_mut - P_wt
                    delta = logits[token_idx, mut_id].item() - logits[token_idx, wt_id].item()
                    sum_additive += delta
                
                if valid_additive:
                    r_res['pred_additive'] = sum_additive
                else:
                    r_res['pred_additive'] = np.nan

            # --- 5. Process Path Scores ---
            for item, logits in zip(path_work_items, path_logits):
                if logits is None: continue
                
                r_idx = item['row_idx']
                p_idx = item['path_idx']
                pos = item['pos_1based']
                token_idx = pos
                
                mut_id = self.vocab.get(item['target_mut'])
                bg_id = self.vocab.get(item['background_aa'])
                
                if mut_id is None or bg_id is None: continue
                
                step_delta = logits[token_idx, mut_id].item() - logits[token_idx, bg_id].item()
                row_results[r_idx]["path_scores"][p_idx] += step_delta

            # --- 6. Format Output ---
            final_rows = []
            for res in row_results:
                scores = res["path_scores"]
                pred_mean = np.mean(scores)
                pred_std = np.std(scores, ddof=1) if len(scores) > 1 else 0.0
                
                out_rec = {
                    "pdb": res["pdb"],
                    "chain_id": res["chain_id"],
                    "N": res["N"],
                    "K_paths": res["K_paths"],
                    "mut_type": res["mut_type"],
                    "pred_additive": res["pred_additive"],
                    "pred_mean": pred_mean,
                    "pred_std": pred_std,
                    "esm3_score": pred_mean
                }
                if return_path_summaries:
                    out_rec["path_sums"] = scores
                final_rows.append(out_rec)

            return pd.DataFrame(final_rows)


def main_(args):

    os.makedirs('tmp', exist_ok=True)

    CHECKPOINT = f"zero_shot/{args.model_name}"

    torch.set_float32_matmul_precision('high')

    tokenizer = EsmSequenceTokenizer("cpu")
    client = ESM3ForgeInferenceClient(token = "44JOUybuKQzm92Svf4UqTa", model = args.model_name)
    predictor = ESM3ForgePredictor(client=client, tokenizer=tokenizer)
        
    if not args.skip_external:
        external_test_dataloaders_names = ['ptmul', 'q3421', 's669', 'ssym', 'k3822'] #'s461', 'k2369',
        stats_parallel = pd.DataFrame()
        stats_masked = pd.DataFrame()

        for name in external_test_dataloaders_names:
            print(name)
            df_true = pd.read_csv(DATA_DIR / f'{name}_mapped_new.csv')

            if os.path.exists(f'./predictions/{name if name!= "ptmul" else "PTMUL"}/{CHECKPOINT}_masked.csv'):
                res_masked = pd.read_csv(f'./predictions/{name if name!= "ptmul" else "PTMUL"}/{CHECKPOINT}_masked.csv', index_col=0)
            else:
                res_parallel = []
                res_masked = []

                time_parallel = 0
                time_masked = 0    

                if name in ['s669', 's461', 'ssym', 'q3421', 'k3822', 'k2369']:
                    df_true = df_true.reset_index()
                    df_true['position_pdb'] = df_true['position']
                    df_true['position'] = df_true['seq_pos']
                    df_true['mut_type'] = df_true['wild_type'] + df_true['position'].astype(int).astype(str) + df_true['mutation']
                    df_true['id'] = df_true['code'] + df_true['chain'] + '_' + df_true['mut_type']
                    df_true = df_true.set_index('id')

                else:
                    df_true = df_true.reset_index()
                    df_true = utils.sort_mutations_by_position(df_true, 'mut_info_seq_pos', 'mut_type')
                    df_true['id'] = df_true['code'] + df_true['chain'] + '_' + df_true['mut_type']
                    df_true = df_true.set_index('id')
                    df_true = utils.parse_multimutant_column(df_true, 'mut_type', max_mutations=10)

                for (pdb, code, chain), data in tqdm(df_true.groupby(['pdb_file', 'code', 'chain'])):
                    
                    backbone_mutation = None

                    singles = data.loc[data['mut_type'].str.count(':')==0]
                    doubles = data.loc[data['mut_type'].str.count(':')==1]
                    multi = data.loc[data['mut_type'].str.count(':')>=2]

                    has_singles = len(singles)
                    has_doubles = len(doubles)
                    has_multi = len(multi)
                    #print(pdb, code, chain, has_singles, has_doubles, has_multi)

                    #pred_combined_parallel = pd.DataFrame()
                    pred_combined_masked = pd.DataFrame()

                    if has_singles:
                        #pred_singles_parallel, t_parallel = timed_call(model.infer_all_singles, pdb, chain=chain, strategy='parallel', subset_df=singles, backbone_mutation=backbone_mutation, quiet=True)
                        #pred_singles_parallel['id'] = code + chain + ('_' + backbone_mutation if backbone_mutation else '') + '_' + pred_singles_parallel['wt'] + pred_singles_parallel['pos'].astype(int).astype(str) + pred_singles_parallel['mut']
                        #pred_combined_parallel = pd.concat([pred_combined_parallel, pred_singles_parallel])

                        pred_singles_masked, t_masked = timed_call(predictor.infer_all_singles, pdb_path=pdb, chain=chain, subset_df=singles)
                        pred_singles_masked['id'] = code + chain + ('_' + backbone_mutation if backbone_mutation else '') + '_' + pred_singles_masked['wild_type'] + pred_singles_masked['seq_pos'].astype(int).astype(str) + pred_singles_masked['mutation']
                        pred_combined_masked = pd.concat([pred_combined_masked, pred_singles_masked])

                    if has_doubles:
                        #pred_doubles_parallel, t_parallel = timed_call(model.infer_all_doubles, pdb, chain=chain, strategy='parallel', subset_df=doubles, backbone_mutation=backbone_mutation, quiet=True)
                        #pred_doubles_parallel['id'] = code + chain + ('_' + backbone_mutation if backbone_mutation else '') + '_' + pred_doubles_parallel['wt_i'] + pred_doubles_parallel['i'].astype(int).astype(str) + pred_doubles_parallel['mut_i'] + ':' + pred_doubles_parallel['wt_j'] + pred_doubles_parallel['j'].astype(int).astype(str) + pred_doubles_parallel['mut_j']
                        #pred_combined_parallel = pd.concat([pred_combined_parallel, pred_doubles_parallel])

                        pred_doubles_masked, t_masked = timed_call(predictor.infer_all_doubles, pdb_path=pdb, chain=chain, subset_df=doubles)
                        pred_doubles_masked['id'] = code + chain + ('_' + backbone_mutation if backbone_mutation else '') + '_' + pred_doubles_masked['wt1'] + pred_doubles_masked['pos1'].astype(int).astype(str) + pred_doubles_masked['mut1'] + ':' + pred_doubles_masked['wt2'] + pred_doubles_masked['pos2'].astype(int).astype(str) + pred_doubles_masked['mut2']
                        pred_combined_masked = pd.concat([pred_combined_masked, pred_doubles_masked])

                    if has_multi:
                    #    pred_multi_parallel, t_parallel = timed_call(model.infer_multimutants_sampled, pdb, chain=chain, strategy='parallel', subset_df=multi)
                    #    pred_multi_parallel['id'] = code + chain + ('_' + backbone_mutation if backbone_mutation else '') + '_' + pred_multi_parallel['mut_type']
                    #    pred_multi_parallel['esm3_score'] = pred_multi_parallel['pred_mean']
                    #    pred_combined_parallel = pd.concat([pred_combined_parallel, pred_multi_parallel.drop('mut_type', axis=1)])

                        pred_multi_masked, t_masked = timed_call(predictor.infer_multimutants_sampled, pdb_path=pdb, chain=chain, subset_df=multi)
                        pred_multi_masked['id'] = code + chain + ('_' + backbone_mutation if backbone_mutation else '') + '_' + pred_multi_masked['mut_type']
                        pred_combined_masked = pd.concat([pred_combined_masked, pred_multi_masked.drop('mut_type', axis=1)])
                    
                    #pred_combined_parallel = pred_combined_parallel.set_index('id')
                    res_partial_masked = pred_combined_masked.set_index('id')

                    #overlap_cols = list(set(data.columns).intersection(set(pred_combined_parallel.columns)))
                    #res_partial_parallel = data.join(pred_combined_parallel.drop(overlap_cols, axis=1))
                    #res_partial_masked = data.join(pred_combined_masked.drop(overlap_cols, axis=1))

                    #res_parallel.append(res_partial_parallel)
                    res_masked.append(res_partial_masked)

                    #time_parallel += t_parallel
                    time_masked += t_masked

                #res_parallel = pd.concat(res_parallel)
                res_masked = pd.concat(res_masked)

                os.makedirs(os.path.dirname(f'./predictions/{name if name!= "ptmul" else "PTMUL"}/{CHECKPOINT}'), exist_ok=True)
                #res_parallel.to_csv(f'./predictions/{name if name!= "ptmul" else "PTMUL"}/{CHECKPOINT}_parallel.csv')
                res_masked.to_csv(f'./predictions/{name if name!= "ptmul" else "PTMUL"}/{CHECKPOINT}_masked.csv')

            #stats_parallel.at[name, 'spearman'] = res_parallel[['ddG', 'esm3_score']].corr('spearman').iloc[0,1]
            #stats_parallel.at[name, 'ndcg@30'] = utils.compute_ndcg_flexible(res_parallel, 'esm3_score', 'ddG', top_n=30)
            #stats_parallel.at[name, 'ndcg>0'] = utils.compute_ndcg_flexible(res_parallel, 'esm3_score', 'ddG', threshold=0)

            stats_masked.at[name, 'spearman'] = res_masked[['ddG', 'esm3_score']].corr('spearman').iloc[0,1]
            stats_masked.at[name, 'ndcg@30'] = utils.compute_ndcg_flexible(res_masked, 'esm3_score', 'ddG', top_n=30)
            stats_masked.at[name, 'ndcg>0'] = utils.compute_ndcg_flexible(res_masked, 'esm3_score', 'ddG', threshold=0)

            #stats_parallel.at[name, 'time'] = time_parallel
            #stats_parallel.at[name, 'time'] = time_parallel
            #stats_masked.at[name, 'time'] = time_masked
            #stats_masked.at[name, 'time'] = time_masked

            if 'ptmul' not in name:
                assert len(df_true) == len(res_masked)
            else:
                print(len(df_true), len(res_masked))

            os.makedirs(os.path.dirname(f'./stats/external/{CHECKPOINT}'), exist_ok=True)
            #stats_parallel.to_csv(f'./stats/external/{CHECKPOINT}_parallel.csv')
            stats_masked.to_csv(f'./stats/external/{CHECKPOINT}_masked.csv')

    ############## REPEAT WITH SPECIFIC SPLITS ################

    if args.split is not None and not args.skip_tsuboyama:
        splits = pd.read_csv(args.split, index_col=0)
        split_name = args.split.split('/')[-1].split('splits_')[1].split('.csv')[0]

        for scaffold in ['validation', 'testing']:

            if not os.path.exists(f'./predictions/{split_name}-{scaffold.replace("testing", "test")}/{CHECKPOINT}_masked.csv'):

                #results_parallel = []
                #results_parallel_ctx = []
                results_masked = []
                #results_masked_ctx = []

                #stats_parallel = pd.DataFrame()
                #stats_parallel_ctx = pd.DataFrame()
                stats_masked = pd.DataFrame()
                #stats_masked_ctx = pd.DataFrame()     

                scaffold_ = scaffold.replace('testing', 'test')
                test_list = eval(splits.loc['stability', scaffold])
                #test_list = test_list[:3]
                #test_list = [str(t).replace('|', '+') for t in test_list]

                tsu = pd.read_csv('~/PSLMs/data/preprocessed/tsuboyama_all_subs_corrected.csv')
                tsu = tsu.loc[~tsu['mut_type'].apply(utils.is_fake_mutation)]
                tsu = tsu.loc[~tsu['uid'].apply(utils.is_improper_mutation)]
                
                tsu = utils.parse_multimutant_column(tsu, 'mut_type')
                tsu['id'] = tsu['code'] + '_' + tsu['mut_type']
                tsu = tsu.sort_values('id')
                print(tsu.head())

                for code in tqdm(test_list):

                    df_true = tsu.loc[tsu['code'].str.contains(code, regex=False)]
                    if not code.startswith('v2_'):
                        df_true = df_true.loc[~df_true['code'].str.startswith('v2_')]
                    pdb = df_true['pdb_file'].head(1).item()
                    df_true['mut_structure'] = df_true['mut_structure'].fillna('-')

                    #res_parallel = []
                    #res_parallel_ctx = []
                    res_masked = []
                    #res_masked_ctx = []

                    #time_parallel = 0
                    #time_parallel_ctx = 0
                    time_masked = 0
                    #time_masked_ctx = 0       

                    for mut_structure, data in df_true.groupby('mut_structure'):
                        
                        if mut_structure != '-':
                            backbone_mutation = mut_structure
                        else:
                            backbone_mutation = None

                        data = data.set_index('id')
                        data = utils.sum_individual_mutation_scores(data, 'ddG_ML', new_score_column='ddG_additive_ML')
                        data['dddG_ML'] = data['ddG_ML'] - data['ddG_additive_ML']

                        #has_singles = len(data.loc[~data['mut_type'].str.contains(':')]) > 0
                        has_doubles = len(data.loc[data['mut_type'].str.contains(':')]) > 0
                        print(code, has_doubles)
                        
                        singles = data.loc[~data['mut_type'].str.contains(':')]
                        doubles = data.loc[data['mut_type'].str.contains(':')]

                        #if has_singles:
                        #pred_singles_parallel, t_parallel = timed_call(predictor.infer_all_singles, pdb, strategy='parallel', backbone_mutation=backbone_mutation, quiet=True)
                        #pred_singles_parallel['id'] = pred_singles_parallel['pdb'].apply(lambda x: x.split('.')[0]) + ('_' + backbone_mutation if backbone_mutation else '') + '_' + pred_singles_parallel['wt'] + pred_singles_parallel['pos'].astype(int).astype(str) + pred_singles_parallel['mut']
                        
                        singles['wild_type'] = singles['mut_type'].str[0]
                        singles['seq_pos'] = singles['mut_type'].str[1:-1]
                        singles['mutation'] = singles['mut_type'].str[-1]
                        pred_singles_masked, t_masked = timed_call(predictor.infer_all_singles, pdb, chain='A', subset_df=singles, backbone_mutation=backbone_mutation) #strategy='masked', quiet=True)
                        pred_singles_masked['id'] = pred_singles_masked['code'] + ('_' + backbone_mutation if backbone_mutation else '') + '_' + pred_singles_masked['wild_type'] + pred_singles_masked['seq_pos'].astype(int).astype(str) + pred_singles_masked['mutation']

                        #t_parallel_ctx = t_parallel
                        #t_masked_ctx = t_masked

                        if has_doubles:
                            #pred_doubles_parallel, t_parallel = timed_call(predictor.infer_all_doubles, pdb, strategy='parallel', subset_df=data, backbone_mutation=backbone_mutation, quiet=True)
                            #pred_doubles_parallel['id'] = pred_doubles_parallel['pdb'].apply(lambda x: x.split('.')[0]) + ('_' + backbone_mutation if backbone_mutation else '') + '_' + pred_doubles_parallel['wt_i'] + pred_doubles_parallel['i'].astype(int).astype(str) + pred_doubles_parallel['mut_i'] + ':' + pred_doubles_parallel['wt_j'] + pred_doubles_parallel['j'].astype(int).astype(str) + pred_doubles_parallel['mut_j']
                            #pred_combined_parallel = pd.concat([pred_singles_parallel, pred_doubles_parallel])

                            #pred_doubles_parallel_ctx, t_parallel_ctx = timed_call(predictor.infer_all_doubles, pdb, strategy='parallel', subset_df=data, backbone_mutation=backbone_mutation, use_predictored_context_structs=True, mut_structs_root='/home/sareeves/PSLMs/data/lora/FINAL_results', quiet=True)
                            #pred_doubles_parallel_ctx['id'] = pred_doubles_parallel_ctx['pdb'].apply(lambda x: x.split('.')[0]) + ('_' + backbone_mutation if backbone_mutation else '') + '_' + pred_doubles_parallel_ctx['wt_i'] + pred_doubles_parallel_ctx['i'].astype(int).astype(str) + pred_doubles_parallel_ctx['mut_i'] + ':' + pred_doubles_parallel_ctx['wt_j'] + pred_doubles_parallel_ctx['j'].astype(int).astype(str) + pred_doubles_parallel_ctx['mut_j']
                            #pred_combined_parallel_ctx = pd.concat([pred_singles_parallel, pred_doubles_parallel_ctx])

                            pred_doubles_masked, t_masked = timed_call(predictor.infer_all_doubles, pdb, chain='A', subset_df=doubles, backbone_mutation=backbone_mutation) #strategy='masked', , quiet=True
                            pred_doubles_masked['id'] = pred_singles_masked['code'] + ('_' + backbone_mutation if backbone_mutation else '') + '_' + pred_doubles_masked['wt1'] + pred_doubles_masked['pos1'].astype(int).astype(str) + pred_doubles_masked['mut1'] + ':' + pred_doubles_masked['wt2'] + pred_doubles_masked['pos2'].astype(int).astype(str) + pred_doubles_masked['mut2']
                            pred_combined_masked = pd.concat([pred_singles_masked, pred_doubles_masked])

                            #pred_doubles_masked_ctx, t_masked_ctx = timed_call(predictor.infer_all_doubles, pdb, strategy='masked', subset_df=data, backbone_mutation=backbone_mutation, use_modeled_context_structs=True, mut_structs_root='/home/sareeves/PSLMs/data/lora/FINAL_results', quiet=True)
                            #pred_doubles_masked_ctx['id'] = pred_doubles_masked_ctx['pdb'].apply(lambda x: x.split('.')[0]) + ('_' + backbone_mutation if backbone_mutation else '') + '_' + pred_doubles_masked_ctx['wt_i'] + pred_doubles_masked_ctx['i'].astype(int).astype(str) + pred_doubles_masked_ctx['mut_i'] + ':' + pred_doubles_masked_ctx['wt_j'] + pred_doubles_masked_ctx['j'].astype(int).astype(str) + pred_doubles_masked_ctx['mut_j']
                            #pred_combined_masked_ctx = pd.concat([pred_singles_masked, pred_doubles_masked_ctx])
                        else:
                            #pred_combined_parallel = pred_singles_parallel
                            #pred_combined_parallel_ctx = pred_singles_parallel
                            pred_combined_masked = pred_singles_masked
                            #pred_combined_masked_ctx = pred_singles_masked
                    
                        #pred_combined_parallel = pred_combined_parallel.set_index('id')
                        #pred_combined_parallel_ctx = pred_combined_parallel_ctx.set_index('id')
                        res_partial_masked = pred_combined_masked.set_index('id')
                        #pred_combined_masked_ctx = pred_combined_masked_ctx.set_index('id')

                        #res_partial_parallel = data.join(pred_combined_parallel)
                        #res_partial_parallel_ctx = data.join(pred_combined_parallel_ctx)
                        #res_partial_masked = data.join(pred_combined_masked)
                        #res_partial_masked_ctx = data.join(pred_combined_masked_ctx)

                        #res_parallel.append(res_partial_parallel)
                        #res_parallel_ctx.append(res_partial_parallel_ctx)
                        res_masked.append(res_partial_masked)
                        #res_masked_ctx.append(res_partial_masked_ctx)

                        #time_parallel += t_parallel
                        #time_parallel_ctx += t_parallel_ctx
                        time_masked += t_masked
                        #time_masked_ctx += t_masked_ctx

                    #res_parallel = pd.concat(res_parallel)
                    #res_parallel_ctx = pd.concat(res_parallel_ctx)
                    res_masked = pd.concat(res_masked)
                    #res_masked_ctx = pd.concat(res_masked_ctx)

                    #stats_parallel.at[code, 'spearman'] = res_parallel[['ddG_ML', 'esm3_score']].corr('spearman').iloc[0,1]
                    #try:
                    #    stats_parallel.at[code, 'spearman_epi'] = res_parallel[['dddG_ML', 'esm3_epistasis_score']].dropna().corr('spearman').iloc[0,1]
                    #except Exception as e:
                        #print(e)
                    #    stats_parallel.at[code, 'spearman_epi'] = float('nan')
                    #stats_parallel.at[code, 'ndcg@30'] = utils.compute_ndcg_flexible(res_parallel, 'esm3_score', 'ddG_ML', top_n=30)
                    #stats_parallel.at[code, 'ndcg>0'] = utils.compute_ndcg_flexible(res_parallel, 'esm3_score', 'ddG_ML', threshold=0)
                    #stats_parallel_ctx.at[code, 'spearman'] = res_parallel_ctx[['ddG_ML', 'esm3_score']].corr('spearman').iloc[0,1]
                    #try:
                    #    stats_parallel_ctx.at[code, 'spearman_epi'] = res_parallel_ctx[['dddG_ML', 'esm3_epistasis_score']].dropna().corr('spearman').iloc[0,1]
                    #except:
                    #    stats_parallel_ctx.at[code, 'spearman_epi'] = float('nan')
                    #stats_parallel_ctx.at[code, 'ndcg@30'] = utils.compute_ndcg_flexible(res_parallel_ctx, 'esm3_score', 'ddG_ML', top_n=30)
                    #stats_parallel_ctx.at[code, 'ndcg>0'] = utils.compute_ndcg_flexible(res_parallel_ctx, 'esm3_score', 'ddG_ML', threshold=0)

                    stats_masked.at[code, 'spearman'] = res_masked[['ddG_ML', 'esm3_score']].corr('spearman').iloc[0,1]
                    try:
                        stats_masked.at[code, 'spearman_epi'] = res_masked[['dddG_ML', 'esm3_epistasis_score']].dropna().corr('spearman').iloc[0,1]
                    except:
                        stats_masked.at[code, 'spearman_epi'] = float('nan')
                    stats_masked.at[code, 'ndcg@30'] = utils.compute_ndcg_flexible(res_masked, 'esm3_score', 'ddG_ML', top_n=30)
                    stats_masked.at[code, 'ndcg>0'] = utils.compute_ndcg_flexible(res_masked, 'esm3_score', 'ddG_ML', threshold=0)
                    
                    #stats_masked_ctx.at[code, 'spearman'] = res_masked_ctx[['ddG_ML', 'esm3_score']].corr('spearman').iloc[0,1]
                    #try:
                    #    stats_masked_ctx.at[code, 'spearman_epi'] = res_masked_ctx[['dddG_ML', 'esm3_epistasis_score']].dropna().corr('spearman').iloc[0,1]
                    #except:
                    #    stats_masked_ctx.at[code, 'spearman_epi'] = float('nan')
                    #stats_masked_ctx.at[code, 'ndcg@30'] = utils.compute_ndcg_flexible(res_masked_ctx, 'esm3_score', 'ddG_ML', top_n=30)
                    #stats_masked_ctx.at[code, 'ndcg>0'] = utils.compute_ndcg_flexible(res_masked_ctx, 'esm3_score', 'ddG_ML', threshold=0)

                    #stats_parallel.at[code, 'time'] = time_parallel
                    #stats_parallel_ctx.at[code, 'time'] = time_parallel_ctx
                    stats_masked.at[code, 'time'] = time_masked
                    #stats_masked_ctx.at[code, 'time'] = time_masked_ctx

                    #assert len(df_true) == len(res_parallel)
                    #assert len(df_true) == len(res_parallel_ctx)
                    assert len(df_true) == len(res_masked)
                    #assert len(df_true) == len(res_masked_ctx)

                    #results_parallel.append(res_parallel.reset_index(drop=True).set_index('uid'))
                    #results_parallel_ctx.append(res_parallel_ctx.reset_index(drop=True).set_index('uid'))
                    results_masked.append(res_masked.reset_index(drop=True).set_index('uid'))
                    #results_masked_ctx.append(res_masked_ctx.reset_index(drop=True).set_index('uid'))

                #results_parallel = pd.concat(results_parallel, axis=0)
                #results_parallel_ctx = pd.concat(results_parallel_ctx, axis=0)
                results_masked = pd.concat(results_masked, axis=0)
                #results_masked_ctx = pd.concat(results_masked_ctx, axis=0)

                #print(stats_parallel.mean(axis=0))
                #print(stats_parallel_ctx.mean(axis=0))
                print(stats_masked.mean(axis=0))
                #print(stats_masked_ctx.mean(axis=0))

                os.makedirs(os.path.dirname((f'./predictions/{split_name}-{scaffold_}/{CHECKPOINT}')), exist_ok=True)
                os.makedirs(os.path.dirname((f'./stats/{split_name}-{scaffold_}/{CHECKPOINT}')), exist_ok=True)

                #stats_parallel.mean(axis=0).to_csv(f'./stats/{split_name}-{scaffold_}/{CHECKPOINT}_parallel_avg.csv')
                #stats_parallel_ctx.mean(axis=0).to_csv(f'./stats/{split_name}-{scaffold_}/{CHECKPOINT}_mut_ctx_parallel_avg.csv')
                stats_masked.mean(axis=0).to_csv(f'./stats/{split_name}-{scaffold_}/{CHECKPOINT}_masked_avg.csv')
                #stats_masked_ctx.mean(axis=0).to_csv(f'./stats/{split_name}-{scaffold_}/{CHECKPOINT}_mut_ctx_masked_avg.csv')

                #results_parallel.to_csv(f'./predictions/{split_name}-{scaffold_}/{CHECKPOINT}_parallel.csv')
                #results_parallel_ctx.to_csv(f'./predictions/{split_name}-{scaffold_}/{CHECKPOINT}_mut_ctx_parallel.csv')
                results_masked.to_csv(f'./predictions/{split_name}-{scaffold_}/{CHECKPOINT}_masked.csv')
                #results_masked_ctx.to_csv(f'./predictions/{split_name}-{scaffold_}/{CHECKPOINT}_mut_ctx_masked.csv')

                #stats_parallel.to_csv(f'./stats/{split_name}-{scaffold_}/{CHECKPOINT}_parallel.csv')
                #stats_parallel_ctx.to_csv(f'./stats/{split_name}-{scaffold_}/{CHECKPOINT}_mut_ctx_parallel.csv')
                stats_masked.to_csv(f'./stats/{split_name}-{scaffold_}/{CHECKPOINT}_masked.csv')
                #stats_masked_ctx.to_csv(f'./stats/{split_name}-{scaffold_}/{CHECKPOINT}_mut_ctx_masked.csv')

                torch.cuda.empty_cache()

    ######################################

    if not args.skip_dms:

        prots = ['DLG4_HUMAN_Faure_2021_abundance_domain', 'DLG4_HUMAN_Faure_2021_binding_domain', 'GRB2_HUMAN_Faure_2021_abundance_domain', 'GRB2_HUMAN_Faure_2021_binding_domain', 'MYO_HUMAN_Kung_2025_display', 'ESTA_BACSU_Nutschel_2020_dTm', 'GB1_Wu_2016_binding_domain'] #, 'A4_HUMAN_Seuma_2022'] # 'GB1_Wu_2016_binding_domain','A4_HUMAN_Seuma_2022', 
        #stats_parallel = pd.DataFrame()
        stats_masked = pd.DataFrame()

        #results_parallel = []
        results_masked = []
    
        for prot in prots:

            if os.path.exists(f'./predictions/{prot}/{CHECKPOINT}_masked.csv'):
                res_masked = pd.read_csv((f'./predictions/{prot}/{CHECKPOINT}_masked.csv'), index_col=0)
            else:          

                df_true = pd.read_csv(DATA_DIR / f'{prot}.csv')
                df_true['id'] = df_true['code'] + '_' + df_true['mut_info']
                df_true = df_true.set_index('id')

                has_singles = len(df_true.loc[~df_true['mut_info'].str.contains(':')]) > 0
                has_doubles = len(df_true.loc[df_true['mut_info'].str.contains(':')]) > 0
                if has_doubles:
                    df_true = utils.sum_individual_mutation_scores(df_true, 'ddG_ML', new_score_column='ddG_additive_ML')
                    df_true['dddG_ML'] = df_true['ddG_ML'] - df_true['ddG_additive_ML']

                print(prot, has_doubles)

                singles = df_true.loc[~df_true['mut_info'].str.contains(':')]
                doubles = df_true.loc[df_true['mut_info'].str.contains(':')]
                doubles = utils.parse_multimutant_column(doubles, 'mut_type')

                prot_name = '_'.join(prot.split('_')[:2])
                if prot_name == 'GB1_Wu':
                    prot_name = 'GB1'

                if has_singles:

                    singles['wild_type'] = singles['mut_type'].str[0]
                    singles['seq_pos'] = singles['mut_type'].str[1:-1]
                    singles['mutation'] = singles['mut_type'].str[-1]

                    #pred_singles_parallel, t_parallel = timed_call(predictor.infer_all_singles, f'/home/sareeves/PSLMs/structures/{prot_name}.pdb', strategy='parallel', quiet=True)
                    #pred_singles_parallel['id'] = pred_singles_parallel['pdb'].apply(lambda x: x.split('.')[0]) + '_' + pred_singles_parallel['wt'] + pred_singles_parallel['pos'].astype(int).astype(str) + pred_singles_parallel['mut']

                    pred_singles_masked, t_masked = timed_call(predictor.infer_all_singles, f'/home/sareeves/PSLMs/structures/{prot_name}.pdb', subset_df=singles) #, strategy='masked', quiet=True)
                    pred_singles_masked['id'] = pred_singles_masked['code'] + '_' + pred_singles_masked['mut_info']

                if has_doubles:
                    #pred_doubles_parallel, t_parallel = timed_call(predictor.infer_all_doubles, f'/home/sareeves/PSLMs/structures/{prot_namecode', strategy='parallel', subset_df=df_true, quiet=True)
                    #pred_doubles_parallel['id'] = pred_doubles_parallel['pdb'].apply(lambda x: x.split('.')[0]) + '_' + pred_doubles_parallel['mut_info']
                    #pred_combined_parallel = pd.concat([pred_singles_parallel, pred_doubles_parallel])
    
                    pred_doubles_masked, t_masked = timed_call(predictor.infer_all_doubles, f'/home/sareeves/PSLMs/structures/{prot_name}.pdb', subset_df=doubles) #strategy='masked', subset_df=df_true, quiet=True)
                    pred_doubles_masked['id'] = pred_doubles_masked['code'] + '_' + pred_doubles_masked['mut_info']
                    pred_combined_masked = pd.concat([pred_singles_masked, pred_doubles_masked])

                else:
                    #pred_combined_parallel = pred_singles_parallel
                    pred_combined_masked = pred_singles_masked
            
                #pred_combined_parallel = pred_combined_parallel.set_index('id')
                res_masked = pred_combined_masked.set_index('id')

                #res_parallel = df_true.join(pred_combined_parallel.drop('pos', axis=1))
                #res_masked = df_true.join(pred_combined_masked.drop('pos', axis=1))

                #assert len(df_true) == len(res_parallel)
                assert len(df_true) == len(res_masked)

                os.makedirs(os.path.dirname((f'./predictions/{prot}/{CHECKPOINT}')), exist_ok=True)
                #res_parallel.to_csv(f'./predictions/{prot}/{CHECKPOINT}_parallel.csv')
                res_masked.to_csv(f'./predictions/{prot}/{CHECKPOINT}_masked.csv')

            #results_parallel.append(res_parallel)
            results_masked.append(res_masked)

            #stats_parallel.at[prot, 'spearman'] = res_parallel[['ddG_ML', 'esm3_score']].corr('spearman').iloc[0,1]
            #try:
            #    stats_parallel.at[prot, 'spearman_epi'] = res_parallel[['dddG_ML', 'esm3_epistasis_score']].dropna().corr('spearman').iloc[0,1]
            #except Exception:
            #    stats_parallel.at[prot, 'spearman_epi'] = float('nan')
            #stats_parallel.at[prot, 'ndcg@30'] = utils.compute_ndcg_flexible(res_parallel, 'esm3_score', 'ddG_ML', top_n=30)
            #stats_parallel.at[prot, 'ndcg>0'] = utils.compute_ndcg_flexible(res_parallel, 'esm3_score', 'ddG_ML', threshold=0)
            
            stats_masked.at[prot, 'spearman'] = res_masked[['ddG_ML', 'esm3_score']].corr('spearman').iloc[0,1]
            try:
                stats_masked.at[prot, 'spearman_epi'] = res_masked[['dddG_ML', 'esm3_epistasis_score']].dropna().corr('spearman').iloc[0,1]
            except Exception:
                stats_masked.at[prot, 'spearman_epi'] = float('nan')
            stats_masked.at[prot, 'ndcg@30'] = utils.compute_ndcg_flexible(res_masked, 'esm3_score', 'ddG_ML', top_n=30)
            stats_masked.at[prot, 'ndcg>0'] = utils.compute_ndcg_flexible(res_masked, 'esm3_score', 'ddG_ML', threshold=0)
            #stats_parallel.at[prot, 'time'] = t_parallel
            #stats_masked.at[prot, 'time'] = t_masked

        #print(stats_parallel)
        print(stats_masked)

        os.makedirs(os.path.dirname((f'./stats/DMS/{CHECKPOINT}')), exist_ok=True)

        #results_parallel = pd.concat(results_parallel, axis=0)
        results_masked = pd.concat(results_masked, axis=0)

        #stats_parallel.to_csv(f'./stats/DMS/{CHECKPOINT}_parallel.csv')
        stats_masked.to_csv(f'./stats/DMS/{CHECKPOINT}_masked.csv')

        torch.cuda.empty_cache()

        #os.makedirs(os.path.dirname(f'./stats/DMS/{CHECKPOINT}'), exist_ok=True)
        #prot_stats.to_csv(f'/home/{"sareeves" if not args.local_cluster else "sreeves"}/software/esm-msr/analysis_notebooks/stats/DMS/{CHECKPOINT}.csv')

    #########################################

    if not args.skip_domainome:

        path = f'/home/{"sareeves" if not args.local_cluster else "sreeves"}/PSLMs/data/domainome1/domainome_mapped_new.csv'
        df = pd.read_csv(path)
        df['code'] = df['domain_ID'].apply(lambda x: x.replace('/', '_'))
        df['ddG_ML'] = df['scaled_fitness']
        df = df.dropna(subset='pdb_file')
        df = df[['code', 'mut_seq', 'mut_info', 'uniprot_ID', 'domain_ID', 'pdb_file', 'ddG_ML']]
        #results_parallel = []
        results_masked = []
        #stats_parallel = pd.DataFrame()
        stats_masked = pd.DataFrame()

        for prot in tqdm(df['code'].unique()):

            strategy = 'masked'

            df_true = df.loc[df['code']==prot]
            df_true['id'] = df_true['domain_ID'] + '_' + df_true['mut_info']
            df_true = df_true.set_index('id')

            df_true['wild_type'] = df_true['mut_info'].str[0]
            df_true['seq_pos'] = df_true['mut_info'].str[1:-1]
            df_true['mutation'] = df_true['mut_info'].str[-1]

            pdb = df_true['pdb_file'].head(1).item()

            #pred_singles_parallel, t_parallel = timed_call(predictor.infer_all_singles, pdb) #, strategy='parallel', quiet=True)
            #pred_singles_parallel['id'] = pred_singles_parallel['pdb'].apply(lambda x: x.split('_')[0]) + '_' + pred_singles_parallel['wt'] + pred_singles_parallel['pos'].astype(int).astype(str) + pred_singles_parallel['mut']

            pred_singles_masked, t_masked = timed_call(predictor.infer_all_singles, pdb, subset_df=df_true) #, strategy='masked', quiet=True)
            pred_singles_masked['id'] = pred_singles_masked['domain_ID'] + '_' + pred_singles_masked['mut_info']
            res_masked = pred_singles_masked

            #res_parallel = df_true.join(pred_singles_parallel.set_index('id').drop('pos', axis=1))
            #res_masked = df_true.join(pred_singles_masked.set_index('id').drop('pos', axis=1))

            #assert len(df_true) == len(res_parallel)
            assert len(df_true) == len(res_masked)

            #stats_parallel.at[prot, 'spearman'] = res_parallel[['ddG_ML', 'esm3_score']].corr('spearman').iloc[0,1]
            #stats_parallel.at[prot, 'ndcg@30'] = utils.compute_ndcg_flexible(res_parallel, 'esm3_score', 'ddG_ML', top_n=30)
            #stats_parallel.at[prot, 'ndcg>0'] = utils.compute_ndcg_flexible(res_parallel, 'esm3_score', 'ddG_ML', threshold=0)

            stats_masked.at[prot, 'spearman'] = res_masked[['ddG_ML', 'esm3_score']].corr('spearman').iloc[0,1]
            stats_masked.at[prot, 'ndcg@30'] = utils.compute_ndcg_flexible(res_masked, 'esm3_score', 'ddG_ML', top_n=30)
            stats_masked.at[prot, 'ndcg>0'] = utils.compute_ndcg_flexible(res_masked, 'esm3_score', 'ddG_ML', threshold=0)

            #stats_parallel.at[prot, 'time'] = t_parallel
            stats_masked.at[prot, 'time'] = t_masked

            #results_parallel.append(res_parallel)
            results_masked.append(res_masked)

        #results_parallel_out = pd.concat(results_parallel, axis=0)
        results_masked_out = pd.concat(results_masked, axis=0)

        #print(stats_parallel.mean(axis=0))
        print(stats_masked.mean(axis=0))

        os.makedirs(os.path.dirname(f'./predictions/domainome/{CHECKPOINT}'), exist_ok=True)
        os.makedirs(os.path.dirname(f'./stats/domainome/{CHECKPOINT}'), exist_ok=True)

        #stats_parallel.mean(axis=0).to_csv(f'./stats/domainome/{CHECKPOINT}_parallel_avg.csv')
        stats_masked.mean(axis=0).to_csv(f'./stats/domainome/{CHECKPOINT}_masked_avg.csv')

        #results_parallel_out.to_csv(f'./predictions/domainome/{CHECKPOINT}_parallel.csv')
        results_masked_out.to_csv(f'./predictions/domainome/{CHECKPOINT}_masked.csv')

        #stats_parallel.to_csv(f'./stats/domainome/{CHECKPOINT}_parallel.csv')
        stats_masked.to_csv(f'./stats/domainome/{CHECKPOINT}_masked.csv')

        torch.cuda.empty_cache()

    ##########################################

    if not args.skip_functional:

        test_list_DMS = ['D7PM05_CLYGR', 'GFP_AEQVI', 'HIS7_YEAST', 'Q6WV12_9MAXI', 'Q8WTC7_9CNID', 'RASK_HUMAN']

        for mem_size, prot in zip([8,8,8,8,8,8], test_list_DMS):

            strategy = 'masked'

            df_true = pd.read_csv(f'/home/{"sareeves" if not args.local_cluster else "sreeves"}/PSLMs/data/lora/DMS/csv_formatted/{prot}.csv')
            df_true['mut_type'] = df_true['MUTS'].apply(lambda x: x.replace(';', ':'))
            df_true['id'] = df_true['code'] + '_' + df_true['mut_type']
            df_true = df_true.set_index('id')
            df_true = utils.parse_multimutant_column(df_true, mut_column='mut_type')
            has_doubles = len(df_true['mut_type'].str.contains(':')) > 0
            print(prot, has_doubles)
            print(df_true)

            pdb = df_true['pdb_file'].head(1).item()

            pred_singles = predictor.infer_all_singles(pdb, subset_df=df_true)
            pred_singles['id'] = pred_singles['pdb'].apply(lambda x: x.split('.')[0]) + '_' + pred_singles['wt'] + pred_singles['pos'].astype(int).astype(str) + pred_singles['mut']

            if has_doubles:
                pred_doubles = predictor.infer_all_doubles(pdb, subset_df=df_true)
                pred_doubles['id'] = pred_doubles['pdb'].apply(lambda x: x.split('.')[0]) + '_' + pred_doubles['wt_i'] + pred_doubles['i'].astype(int).astype(str) + pred_doubles['mut_i'] + ':' + pred_doubles['wt_j'] + pred_doubles['j'].astype(int).astype(str) + pred_doubles['mut_j']
                pred_combined = pd.concat([pred_singles, pred_doubles])
            else:
                pred_combined = pred_singles
        
            pred_combined = pred_combined.set_index('id')

            res = df_true.join(pred_combined)
            assert len(df_true) == len(res)
            print(res[['ddG_dir', 'ddg_pred']].corr('spearman').iloc[0,1])

            os.makedirs(os.path.dirname(f'./predictions/{prot}/{CHECKPOINT}'), exist_ok=True)
            res.to_csv(f'./predictions/{prot}/{CHECKPOINT}{"_alpha"+str(args.lora_alpha)}chain_rule_avg.csv')
            torch.cuda.empty_cache()

if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_name', type=str, default='esm3-small-2024-08')
        parser.add_argument('--split', type=str, default=DATA_DIR / 'splits_feb19_tsuboyama.csv')
        parser.add_argument('--loc', type=str, default='inference_scripts')
        parser.add_argument('--local_cluster', action='store_true')
        parser.add_argument('--mask_sequence_pos', type=bool, default=True)
        parser.add_argument('--mask_structure_pos', action='store_true')
        parser.add_argument('--mask_coords_pos', action='store_true')
        parser.add_argument('--mask_coords', action='store_true')
        parser.add_argument('--regenerate_results', action='store_true')
        parser.add_argument('--skip_external', action='store_true')
        parser.add_argument('--skip_tsuboyama', action='store_true')
        parser.add_argument('--skip_dms', action='store_true')
        parser.add_argument('--skip_functional', action='store_true')
        parser.add_argument('--skip_domainome', action='store_true')

        # Parse known args for main parser
        args, remaining_argv = parser.parse_known_args()

        # Initialize conditional args namespaces
        # Use temporary namespaces to avoid conflicts if args are defined in multiple subparsers
        ranking_ns = argparse.Namespace()
        regression_ns = argparse.Namespace()

        # Keep track of remaining args after each parse
        current_remaining_argv = list(remaining_argv) # Make a mutable copy

        # Check if any arguments were truly unrecognized by any relevant parser
        if current_remaining_argv:
            parser.error(f"unrecognized arguments: {' '.join(current_remaining_argv)}")

        # Assign the parsed namespaces to the main args object
        args.ranking_config = ranking_ns

        # Convert sub-parser namespaces to dictionaries for easier access in LightningModule
        args.ranking_config = vars(args.ranking_config)

        if args.skip_external:
            print('Skipping benchmark datasets!')
        if args.skip_tsuboyama:
            print('Skipping MegaScale validation and testing datasets!')
        if args.skip_functional:
            print('Skipping double mutant DMS assays!')
        if args.skip_domainome:
            print('Skipping domainome VAMP assays!')
        if args.mask_structure_pos or args.mask_coords_pos:
            print('Masking one or more inputs!')
        if not args.split:
            print('Warning! Not using any specific split file!')

        main_(args)