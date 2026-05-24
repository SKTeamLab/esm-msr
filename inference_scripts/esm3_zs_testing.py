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
import re

from scipy.stats import entropy
from scipy.special import softmax

from esm_msr import utils, preprocessing

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

def calculate_and_save_stats(eval_dfs, checkpoint_name, category_name, args):
    """
    Applies the SPURS stats tracking logic to a dictionary of evaluated datasets.
    Calculates global stats and per-protein grouped stats.
    """
    stats_df = pd.DataFrame()
    for name, df in eval_dfs.items():
        if df.empty: continue
        
        print(f"Calculating stats for {name} ({category_name})")
        
        # Map true column dynamically based on dataset
        true_col = 'ddG_ML' if 'ddG_ML' in df.columns else 'ddG'
        if true_col not in df.columns and 'ddG_true' in df.columns:
            true_col = 'ddG_true'
        elif true_col not in df.columns and 'ddG_dir' in df.columns:
            true_col = 'ddG_dir'
        
        pred_col = 'esm3_score'

        # Standardize n_muts
        mut_col = 'mut_type' if 'mut_type' in df.columns else 'mut_info'
        if mut_col in df.columns:
            df['n_muts'] = df[mut_col].astype(str).apply(lambda x: len(re.split(r'[:;]', x)))
        else:
            df['n_muts'] = 1 # Fallback for strictly single datasets lacking mut col
        
        # FLAW: Destructive slicing that corrupts non-PDB identifiers
        code_col = 'code' if 'code' in df.columns else 'pdb_id'
        if code_col not in df.columns: df['code'] = name
        df['prot_group'] = df[code_col].astype(str).str[:-1]

        # --- GLOBAL STATS ---
        stats_df.at[name, 'spearman_all'] = df[[true_col, pred_col]].corr('spearman').iloc[0, 1]
        
        singles_mask = (df['n_muts'] == 1) & (~df[pred_col].isna())
        stats_df.at[name, 'n_singles'] = singles_mask.sum()
        if singles_mask.sum() > 0:
            stats_df.at[name, 'spearman_singles'] = df.loc[df['n_muts'] == 1, [true_col, pred_col]].corr('spearman').iloc[0, 1]
        
        doubles_mask = (df['n_muts'] > 1) & (~df[pred_col].isna())
        stats_df.at[name, 'n_doubles'] = doubles_mask.sum() if doubles_mask.sum() > 0 else float('nan')
        if doubles_mask.sum() > 0:
            stats_df.at[name, 'spearman_doubles'] = df.loc[df['n_muts'] > 1, [true_col, pred_col]].corr('spearman').iloc[0, 1]

        if 'dddG_ML' in df.columns and 'esm3_epistasis_score' in df.columns:
            stats_df.at[name, 'spearman_doubles_epi'] = df[['dddG_ML', 'esm3_epistasis_score']].dropna().corr('spearman').iloc[0, 1]
        
        try:
            stats_df.at[name, 'ndcg@30'] = utils.compute_ndcg_flexible(df, pred_col, true_col, top_n=30)
            stats_df.at[name, 'ndcg>0'] = utils.compute_ndcg_flexible(df, pred_col, true_col, threshold=0)
        except Exception:
            pass

        # --- PER-PROTEIN STATS ---
        stats_per_prot = pd.DataFrame()
        for prot, group_df in df.groupby('prot_group'):
            stats_per_prot.at[prot, 'spearman_all'] = group_df[[true_col, pred_col]].corr('spearman').iloc[0, 1]
            
            sg_mask = (group_df['n_muts'] == 1) & (~group_df[pred_col].isna())
            stats_per_prot.at[prot, 'n_singles'] = sg_mask.sum()
            if sg_mask.sum() > 0:
                stats_per_prot.at[prot, 'spearman_singles'] = group_df.loc[group_df['n_muts'] == 1, [true_col, pred_col]].corr('spearman').iloc[0, 1]
            
            db_mask = (group_df['n_muts'] > 1) & (~group_df[pred_col].isna())
            stats_per_prot.at[prot, 'n_doubles'] = db_mask.sum() if db_mask.sum() > 0 else float('nan')
            if db_mask.sum() > 0:
                stats_per_prot.at[prot, 'spearman_doubles'] = group_df.loc[group_df['n_muts'] > 1, [true_col, pred_col]].corr('spearman').iloc[0, 1]

            if 'dddG_ML' in group_df.columns and 'esm3_epistasis_score' in group_df.columns:
                stats_per_prot.at[prot, 'spearman_doubles_epi'] = group_df[['dddG_ML', 'esm3_epistasis_score']].dropna().corr('spearman').iloc[0, 1]

            try:
                stats_per_prot.at[prot, 'ndcg@30'] = utils.compute_ndcg_flexible(group_df, pred_col, true_col, top_n=30)
                stats_per_prot.at[prot, 'ndcg>0'] = utils.compute_ndcg_flexible(group_df, pred_col, true_col, threshold=0)
            except Exception:
                pass
        
        safe_name = name.replace('/', '_')
        prot_out_dir = REPO_ROOT / f'analysis_notebooks/stats/{safe_name}/esm3_forge/{checkpoint_name}'
        os.makedirs(prot_out_dir, exist_ok=True)
        stats_per_prot.to_csv(prot_out_dir / 'per_protein_stats.csv')
        
        # FLAW: Statistically invalid arithmetic mean of Spearman correlations
        if args.split and str(args.split) in name:
            stats_per_prot.mean(axis=0).to_csv(prot_out_dir / 'avg_stats.csv')

    out_stats_dir = REPO_ROOT / f'analysis_notebooks/stats/esm3_forge/{checkpoint_name}'
    os.makedirs(out_stats_dir, exist_ok=True)
    stats_df.to_csv(out_stats_dir / f'{category_name}_global_stats.csv')
    print(f"\n--- Global Stats ({category_name}) ---")
    print(stats_df)


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
        self.logits_cache = {}
        self.cache_locks = {}

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
        if sequence in self.logits_cache:
            return self.logits_cache[sequence]
        
        if sequence not in self.cache_locks:
            self.cache_locks[sequence] = asyncio.Lock()
            
        async with self.cache_locks[sequence]:
            if sequence in self.logits_cache:
                return self.logits_cache[sequence]
                
            protein_input = ESMProtein(sequence=sequence, coordinates=coords)
            
            encoded_tensor = await self._safe_api_call(self.client.encode, protein_input)
            if encoded_tensor is None: return None

            config = LogitsConfig(sequence=True, structure=False)
            logits_output = await self._safe_api_call(self.client.logits, encoded_tensor, config)
            if logits_output is None: return None

            seq_logits = logits_output.logits.sequence
            if seq_logits.ndim == 3:
                seq_logits = seq_logits[0]
                
            seq_logits = seq_logits.cpu()
            self.logits_cache[sequence] = seq_logits
            return seq_logits

    def infer_all_singles(self, pdb_path: str, subset_df: pd.DataFrame, chain: str = "A", backbone_mutation = None, use_masks: bool = False):
        req_cols = {'wild_type', 'seq_pos', 'mutation'}
        if not req_cols.issubset(subset_df.columns):
            raise ValueError(f"subset_df missing columns: {req_cols}")

        return self._run_async(self._infer_singles_async(pdb_path, chain, subset_df, backbone_mutation, use_masks))

    def infer_all_doubles(self, pdb_path: str, subset_df: pd.DataFrame, chain: str = "A", backbone_mutation = None, use_masks: bool = False):
        req_cols = {'wt1', 'pos1', 'mut1', 'wt2', 'pos2', 'mut2'}
        if not req_cols.issubset(subset_df.columns):
            raise ValueError(f"subset_df missing columns: {req_cols}")

        return self._run_async(self._infer_doubles_async(pdb_path, chain, subset_df, backbone_mutation, use_masks))

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

    async def _infer_singles_async(self, pdb_path, chain_id, subset_df, backbone_mutation, use_masks):
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

        if not use_masks:
            wt_logits = await self._get_logits_for_sequence(wt_seq, coords)
            tasks, df_indices, valid_rows = [], [], []
            
            for idx, row in tqdm(subset_df.iterrows()):
                p = int(row['seq_pos'])
                if 1 <= p <= len(wt_seq):
                    mut_seq = list(wt_seq)
                    mut_seq[p-1] = row['mutation']
                    tasks.append(self._get_logits_for_sequence("".join(mut_seq), coords))
                    valid_rows.append(row)
                    df_indices.append(idx)
                    
            results_list = await asyncio.gather(*tasks)
            
            out_df = subset_df.copy()
            out_df['esm3_score_wt_seq'] = np.nan
            out_df['esm3_score_mut_seq'] = np.nan
            out_df['esm3_score'] = np.nan
            
            for idx, row, mut_logits in zip(df_indices, valid_rows, results_list):
                p = int(row['seq_pos'])
                w = wt_seq[p-1]
                m = row['mutation']
                
                w_id, m_id = self.vocab.get(w), self.vocab.get(m)
                if wt_logits is not None and mut_logits is not None and w_id is not None and m_id is not None:
                    score_wt = wt_logits[p, m_id].item() - wt_logits[p, w_id].item()
                    score_mut = mut_logits[p, m_id].item() - mut_logits[p, w_id].item()
                    out_df.at[idx, 'esm3_score_wt_seq'] = score_wt
                    out_df.at[idx, 'esm3_score_mut_seq'] = score_mut
                    out_df.at[idx, 'esm3_score'] = 0.5 * (score_wt + score_mut)
            
            return out_df

        # --- Masked approach ---
        tasks_map = defaultdict(list)
        for idx, row in subset_df.iterrows():
            p = int(row['seq_pos'])
            if 1 <= p <= len(wt_seq):
                tasks_map[p].append((row['mutation'], idx))

        unique_positions = list(tasks_map.keys())
        tasks = []
        print(f"Singles: Queueing {len(unique_positions)} masked requests...")
        
        for p in unique_positions:
            masked_seq = wt_seq[:p-1] + "_" + wt_seq[p:]
            tasks.append(self._get_logits_for_sequence(masked_seq, coords))

        results_list = await asyncio.gather(*tasks)

        out_df = subset_df.copy()
        out_df['esm3_score'] = np.nan
        out_df['esm3_logits_entropy'] = np.nan

        for pos, logits in zip(unique_positions, results_list):
            if logits is None: continue
            
            token_idx = pos 
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

    async def _infer_doubles_async(self, pdb_path, chain_id, subset_df, backbone_mutation, use_masks):
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

        if not use_masks:
            wt_logits = await self._get_logits_for_sequence(wt_seq, coords)
            tasks, df_indices, valid_rows = [], [], []
            
            for idx, row in tqdm(subset_df.iterrows()):
                p1, p2 = int(row['pos1']), int(row['pos2'])
                if 1 <= p1 <= len(wt_seq) and 1 <= p2 <= len(wt_seq):
                    mut_seq = list(wt_seq)
                    mut_seq[p1-1] = row['mut1']
                    mut_seq[p2-1] = row['mut2']
                    tasks.append(self._get_logits_for_sequence("".join(mut_seq), coords))
                    valid_rows.append(row)
                    df_indices.append(idx)
                    
            results_list = await asyncio.gather(*tasks)
            
            out_df = subset_df.copy()
            out_df['esm3_score_wt_seq'] = np.nan
            out_df['esm3_score_mut_seq'] = np.nan
            out_df['pred_additive'] = np.nan
            out_df['esm3_score'] = np.nan
            out_df['esm3_epistasis_score'] = np.nan
            
            for idx, row, mut_logits in zip(df_indices, valid_rows, results_list):
                p1, m1, w1 = int(row['pos1']), row['mut1'], row['wt1']
                p2, m2, w2 = int(row['pos2']), row['mut2'], row['wt2']
                
                def get_delta(logits, p, w, m):
                    w_id, m_id = self.vocab.get(w), self.vocab.get(m)
                    if logits is None or w_id is None or m_id is None: return np.nan
                    return logits[p, m_id].item() - logits[p, w_id].item()
                
                score_wt = get_delta(wt_logits, p1, w1, m1) + get_delta(wt_logits, p2, w2, m2)
                score_mut = get_delta(mut_logits, p1, w1, m1) + get_delta(mut_logits, p2, w2, m2)
                
                out_df.at[idx, 'esm3_score_wt_seq'] = score_wt
                out_df.at[idx, 'esm3_score_mut_seq'] = score_mut
                if not np.isnan(score_wt) and not np.isnan(score_mut):
                    out_df.at[idx, 'pred_additive'] = score_wt
                    out_df.at[idx, 'esm3_score'] = 0.5 * (score_wt + score_mut)
                    out_df.at[idx, 'esm3_epistasis_score'] = out_df.at[idx, 'esm3_score'] - score_wt
                    
            return out_df

        # --- Masked approach ---
        singles_queries = set() 
        context_queries = set() 

        for _, row in subset_df.iterrows():
            p1, m1 = int(row['pos1']), row['mut1']
            p2, m2 = int(row['pos2']), row['mut2']
            
            singles_queries.add(p1)
            singles_queries.add(p2)
            
            context_queries.add((p1, m1, p2))
            context_queries.add((p2, m2, p1))

        sorted_singles = sorted(list(singles_queries))
        print(f"Doubles: Queueing {len(sorted_singles)} single-mutant mask requests...")
        
        single_tasks = []
        for p in sorted_singles:
            masked_seq = wt_seq[:p-1] + "_" + wt_seq[p:]
            single_tasks.append(self._get_logits_for_sequence(masked_seq, coords))
        
        single_results = await asyncio.gather(*single_tasks)
        singles_cache = {p: res for p, res in zip(sorted_singles, single_results) if res is not None}

        sorted_contexts = sorted(list(context_queries))
        print(f"Doubles: Queueing {len(sorted_contexts)} context-mutant mask requests...")
        
        context_tasks = []
        for (ctx_p, ctx_m, tgt_p) in sorted_contexts:
            seq_list = list(wt_seq)
            seq_list[ctx_p-1] = ctx_m
            seq_list[tgt_p-1] = "_"
            masked_seq = "".join(seq_list)
            context_tasks.append(self._get_logits_for_sequence(masked_seq, coords))

        context_results = await asyncio.gather(*context_tasks)
        context_cache = {k: res for k, res in zip(sorted_contexts, context_results) if res is not None}

        out_df = subset_df.copy()
        out_df['pred_additive'] = np.nan
        out_df['esm3_score'] = np.nan

        for idx, row in out_df.iterrows():
            p1, m1, wt1 = int(row['pos1']), row['mut1'], row['wt1']
            p2, m2, wt2 = int(row['pos2']), row['mut2'], row['wt2']
            
            def get_delta(logits, pos, wt, mut):
                if logits is None: return np.nan
                t_idx = pos
                wt_id = self.vocab.get(wt)
                mut_id = self.vocab.get(mut)
                if wt_id is None or mut_id is None: return np.nan
                return logits[t_idx, mut_id].item() - logits[t_idx, wt_id].item()

            logits_single_1 = singles_cache.get(p1)
            logits_single_2 = singles_cache.get(p2)
            logits_ctx_2_given_1 = context_cache.get((p1, m1, p2))
            logits_ctx_1_given_2 = context_cache.get((p2, m2, p1))

            d1 = get_delta(logits_single_1, p1, wt1, m1)
            d2 = get_delta(logits_single_2, p2, wt2, m2)
            
            d2_1 = get_delta(logits_ctx_2_given_1, p2, wt2, m2)
            d1_2 = get_delta(logits_ctx_1_given_2, p1, wt1, m1)

            if np.isnan([d1, d2]).any():
                continue

            out_df.at[idx, 'pred_additive'] = d1 + d2

            if not np.isnan([d2_1, d1_2]).any():
                path1 = d1 + d2_1
                path2 = d2 + d1_2
                out_df.at[idx, 'esm3_score'] = 0.5 * (path1 + path2)

            out_df['esm3_epistasis_score'] = out_df['esm3_score'] - out_df['pred_additive']
        
        return out_df
    
    async def _infer_multimutants_unmasked_async(self, pdb_path, chain, subset_df):
        chain_obj = ProteinChain.from_pdb(pdb_path, chain)
        wt_seq = chain_obj.sequence
        coords = torch.tensor(chain_obj.atom37_positions)

        def _get_muts_from_row(row):
            muts = []
            for k in range(1, 11):
                wt_c, pos_c, mut_c = f"wt{k}", f"pos{k}", f"mut{k}"
                if wt_c in row and pos_c in row and mut_c in row:
                    wt, pos, mut = row[wt_c], row[pos_c], row[mut_c]
                    if pd.notna(wt) and pd.notna(pos) and pd.notna(mut):
                        muts.append((wt, int(pos), mut))
            return muts

        wt_logits = await self._get_logits_for_sequence(wt_seq, coords)

        tasks, df_indices, muts_list = [], [], []
        for row_idx, row in tqdm(subset_df.iterrows()):
            muts = _get_muts_from_row(row)
            if not muts: continue
            
            valid = True
            mut_seq = list(wt_seq)
            for fr, pos, to in muts:
                if not (1 <= pos <= len(wt_seq)):
                    valid = False
                    break
                mut_seq[pos-1] = to
                
            if valid:
                tasks.append(self._get_logits_for_sequence("".join(mut_seq), coords))
                df_indices.append(row_idx)
                muts_list.append(muts)

        mut_logits_list = await asyncio.gather(*tasks)

        out_df = subset_df.copy()
        out_df['esm3_score_wt_seq'] = np.nan
        out_df['esm3_score_mut_seq'] = np.nan
        out_df['pred_additive'] = np.nan
        out_df['esm3_score'] = np.nan
        out_df['esm3_epistasis_score'] = np.nan
        out_df['N'] = np.nan

        for idx, muts, mut_logits in zip(df_indices, muts_list, mut_logits_list):
            def get_delta(logits, p, w, m):
                w_id, m_id = self.vocab.get(w), self.vocab.get(m)
                if logits is None or w_id is None or m_id is None: return np.nan
                return logits[p, m_id].item() - logits[p, w_id].item()

            score_wt = sum(get_delta(wt_logits, pos, fr, to) for fr, pos, to in muts)
            score_mut = sum(get_delta(mut_logits, pos, fr, to) for fr, pos, to in muts)

            out_df.at[idx, 'esm3_score_wt_seq'] = score_wt
            out_df.at[idx, 'esm3_score_mut_seq'] = score_mut
            if not np.isnan(score_wt) and not np.isnan(score_mut):
                out_df.at[idx, 'pred_additive'] = score_wt
                out_df.at[idx, 'esm3_score'] = 0.5 * (score_wt + score_mut)
                out_df.at[idx, 'esm3_epistasis_score'] = out_df.at[idx, 'esm3_score'] - score_wt

            out_df.at[idx, 'N'] = len(muts)

        if "original_index" in out_df.columns:
            out_df.index = out_df["original_index"]

        return out_df

    def infer_multimutants_sampled(
            self,
            pdb_path: str,
            subset_df: pd.DataFrame,
            chain: str = "A",
            K_paths: int = 4,
            return_path_summaries: bool = False,
            use_masks: bool = False
        ):
            """
            Estimate Δ for multi-mutants by sampling K random single-mutation paths 
            or using unmasked WT/MUT evaluation.
            """
            if not use_masks:
                return self._run_async(self._infer_multimutants_unmasked_async(pdb_path, chain, subset_df))

            # --- Masked approach ---
            import random

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

            chain_obj = ProteinChain.from_pdb(pdb_path, chain)
            wt_seq = chain_obj.sequence
            coords = torch.tensor(chain_obj.atom37_positions)
            
            path_work_items = []
            unique_additive_positions = set()
            row_results = [] 

            for row_idx, row in subset_df.iterrows():
                muts = _get_muts_from_row(row)
                if not muts: continue
                
                N = len(muts)
                
                # FLAW FIX: Track the original index to prevent index obliteration
                row_results.append({
                    "original_index": row_idx,
                    "pdb": os.path.basename(pdb_path),
                    "chain_id": chain,
                    "N": N,
                    "K_paths": K_paths,
                    "mut_type": _canonical_mut_string(muts),
                    "path_scores": [0.0] * K_paths,
                    "mutations_list": muts,
                    "pred_additive": 0.0
                })
                result_idx = len(row_results) - 1

                for fr, pos, to in muts:
                    unique_additive_positions.add(pos)

                for k in range(K_paths):
                    path_order = muts.copy()
                    if N > 1:
                        random.shuffle(path_order)
                    
                    current_seq_list = list(wt_seq)
                    
                    for step_idx, (fr, pos, to) in enumerate(path_order):
                        p_idx = pos - 1
                        
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
                        
                        current_seq_list[p_idx] = to

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
                
                for seq in additive_work_items:
                    tasks.append(self._get_logits_for_sequence(seq, coords))
                
                for item in path_work_items:
                    tasks.append(self._get_logits_for_sequence(item['masked_seq'], coords))
                
                print(f"Sampling: Queueing {len(tasks)} requests ({len(additive_work_items)} additive + {len(path_work_items)} path steps)...")
                return await asyncio.gather(*tasks)

            all_results = self._run_async(process_all())

            num_additive = len(additive_work_items)
            additive_logits = all_results[:num_additive]
            path_logits = all_results[num_additive:]

            additive_cache = {}
            for pos, logits in zip(additive_tasks_list, additive_logits):
                if logits is not None:
                    additive_cache[pos] = logits

            for r_res in row_results:
                sum_additive = 0.0
                valid_additive = True
                
                for fr, pos, to in r_res['mutations_list']:
                    if pos not in additive_cache:
                        valid_additive = False; break
                    
                    logits = additive_cache[pos]
                    token_idx = pos
                    
                    mut_id = self.vocab.get(to)
                    wt_id = self.vocab.get(fr)
                    
                    if mut_id is None or wt_id is None:
                        valid_additive = False; break
                        
                    delta = logits[token_idx, mut_id].item() - logits[token_idx, wt_id].item()
                    sum_additive += delta
                
                if valid_additive:
                    r_res['pred_additive'] = sum_additive
                else:
                    r_res['pred_additive'] = np.nan

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

            final_rows = []
            for res in row_results:
                scores = res["path_scores"]
                pred_mean = np.mean(scores)
                pred_std = np.std(scores, ddof=1) if len(scores) > 1 else 0.0
                
                out_rec = {
                    "original_index": res["original_index"],
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

            out_df = pd.DataFrame(final_rows)
            # Restore the true dataframe index so pd.concat works correctly later
            if "original_index" in out_df.columns:
                out_df.index = out_df["original_index"]
            return out_df

def main_(args):
    os.makedirs('tmp', exist_ok=True)

    CHECKPOINT = f"zero_shot/{args.model_name}"

    torch.set_float32_matmul_precision('high')

    tokenizer = EsmSequenceTokenizer("cpu")
    client = ESM3ForgeInferenceClient(token = "44JOUybuKQzm92Svf4UqTa", model = args.model_name)
    predictor = ESM3ForgePredictor(client=client, tokenizer=tokenizer)
        
    if not args.skip_external:
        external_test_dataloaders_names = ['ptmul', 'q3421', 's669', 'ssym', 'k3822', 's571', 's783', 's8754', 's2648']
        external_dfs = {}

        for name in external_test_dataloaders_names:
            print(name)
            df_true = pd.read_csv(DATA_DIR / f'{name}_mapped.csv')

            if os.path.exists(f'./predictions/{name if name!= "ptmul" else "PTMUL"}/{CHECKPOINT}_masked.csv'):
                res_masked = pd.read_csv(f'./predictions/{name if name!= "ptmul" else "PTMUL"}/{CHECKPOINT}_masked.csv', index_col=0)
            else:
                if name in ['q3421', 's669', 'ssym', 'k3822', 's571', 's783', 's8754', 's2648']:
                    df_true = df_true.reset_index()
                    df_true['position_pdb'] = df_true['position']
                    df_true['position'] = df_true['seq_pos']
                    df_true['mut_type'] = df_true['wild_type'] + df_true['position'].astype(int).astype(str) + df_true['mutation']
                    df_true['id'] = df_true['code'] + df_true['chain'] + '_' + df_true['mut_type']
                    df_true = df_true.set_index('id')
                    if name == 's571':
                        df_true['ddG'] = df_true['dTm']
                else:
                    df_true = df_true.reset_index()
                    df_true = utils.sort_mutations_by_position(df_true, 'mut_info_seq_pos', 'mut_type')
                    df_true['id'] = df_true['code'] + df_true['chain'] + '_' + df_true['mut_type']
                    df_true = df_true.set_index('id')
                    df_true = utils.parse_multimutant_column(df_true, 'mut_type', max_mutations=10)

                res_masked_list = []
                for (pdb, code, chain), data in tqdm(df_true.groupby(['pdb_file', 'code', 'chain'])):
                    backbone_mutation = None
                    singles = data.loc[data['mut_type'].str.count(':')==0].copy()
                    doubles = data.loc[data['mut_type'].str.count(':')==1].copy()
                    multi = data.loc[data['mut_type'].str.count(':')>=2].copy()

                    pred_combined_masked = pd.DataFrame()

                    if not singles.empty:
                        singles['wild_type'] = singles['mut_type'].str[0]
                        singles['seq_pos'] = singles['mut_type'].str[1:-1]
                        singles['mutation'] = singles['mut_type'].str[-1]
                        pred_singles_masked, _ = timed_call(predictor.infer_all_singles, pdb, chain=chain, subset_df=singles, backbone_mutation=backbone_mutation, use_masks=args.use_masks)
                        pred_singles_masked['id'] = pred_singles_masked['code'] + '_' + pred_singles_masked['wild_type'] + pred_singles_masked['seq_pos'].astype(int).astype(str) + pred_singles_masked['mutation'] #+ ('_' + backbone_mutation if backbone_mutation else '')
                        pred_combined_masked = pd.concat([pred_combined_masked, pred_singles_masked])

                    if not doubles.empty:
                        pred_doubles_masked, _ = timed_call(predictor.infer_all_doubles, pdb, chain=chain, subset_df=doubles, backbone_mutation=backbone_mutation, use_masks=args.use_masks)
                        pred_doubles_masked['id'] = pred_doubles_masked['code'] + '_' + pred_doubles_masked['wt1'] + pred_doubles_masked['pos1'].astype(int).astype(str) + pred_doubles_masked['mut1'] + ':' + pred_doubles_masked['wt2'] + pred_doubles_masked['pos2'].astype(int).astype(str) + pred_doubles_masked['mut2']
                        pred_combined_masked = pd.concat([pred_combined_masked, pred_doubles_masked])

                    if not multi.empty:
                        pred_multi_masked, _ = timed_call(predictor.infer_multimutants_sampled, pdb_path=pdb, chain=chain, subset_df=multi, use_masks=args.use_masks)
                        pred_multi_masked['id'] = pred_multi_masked.index # id is preserved if original_index was restored
                        pred_combined_masked = pd.concat([pred_combined_masked, pred_multi_masked])
                    
                    if 'id' in pred_combined_masked.columns:
                        res_partial_masked = pred_combined_masked.set_index('id')
                    else:
                        res_partial_masked = pred_combined_masked
                    
                    res_masked_list.append(res_partial_masked)

                res_masked = pd.concat(res_masked_list, axis=0)
                
                out_dir = REPO_ROOT / 'analysis_notebooks' / f'predictions/{name if name!= "ptmul" else "PTMUL"}/esm3_forge/{CHECKPOINT}'
                os.makedirs(out_dir, exist_ok=True)
                res_masked.to_csv(out_dir / f'predictions{"_masked" if args.use_masks else "_unmasked"}.csv')

            external_dfs[name] = res_masked
        
        calculate_and_save_stats(external_dfs, CHECKPOINT, 'external', args)

    # ================= REPEAT WITH SPECIFIC SPLITS =================
    if args.split is not None and not args.skip_tsuboyama:
        split_file = REPO_ROOT / "data" / f"{args.split}.pkl"
        split_name = args.split

        ds = preprocessing.MegaScaleDatasetPreprocessor(
            data_file = str(REPO_ROOT / 'data/tsuboyama/Tsuboyama2023_Dataset2_Dataset3_20230416.csv'),
            af_model_folder = str(REPO_ROOT / 'data/tsuboyama/AlphaFold_model_PDBs'))

        splits = ds.create_training_splits(str(split_file), -1)

        for scaffold in ['validation', 'testing']:
            scaffold_ = {'validation': 'val', 'testing': 'test'}[scaffold]
            tsu_dfs = {}
            
            out_dir = REPO_ROOT / 'analysis_notebooks' / f'predictions/{split_name}-{scaffold_}/esm3_forge/{CHECKPOINT}'
            if not os.path.exists(out_dir / f'predictions{"_masked" if args.use_masks else "_unmasked"}.csv'):
                results_masked = []

                data_scaffold = ds.split_dfs[scaffold_]
                
                data_scaffold = utils.parse_multimutant_column(data_scaffold, 'mut_type')
                data_scaffold['id'] = data_scaffold['code'] + '_' + data_scaffold['mut_type']
                data_scaffold = data_scaffold.sort_values('id')

                for code in tqdm(data_scaffold['code_wt'].unique()):

                    df_true = data_scaffold.loc[data_scaffold['code_wt']==code]
                    assert len(df_true) > 0
                    df_true['mut_structure'] = df_true['mut_structure'].fillna('-')

                    res_masked_list = []
                    for mut_structure, data in df_true.groupby('mut_structure'):
                        backbone_mutation = mut_structure if mut_structure != '-' else None

                        data = data.set_index('id')
                        data = utils.sum_individual_mutation_scores(data, 'ddG_ML', new_score_column='ddG_additive_ML')
                        data['dddG_ML'] = data['ddG_ML'] - data['ddG_additive_ML']
                        pdb = data['pdb_file'].head(1).item()

                        has_doubles = len(data.loc[data['mut_type'].str.contains(':')]) > 0
                        singles = data.loc[~data['mut_type'].str.contains(':')].copy()
                        doubles = data.loc[data['mut_type'].str.contains(':')].copy()

                        pred_combined_masked = pd.DataFrame()
                        
                        if not singles.empty:
                            singles['wild_type'] = singles['mut_type'].str[0]
                            singles['seq_pos'] = singles['mut_type'].str[1:-1]
                            singles['mutation'] = singles['mut_type'].str[-1]
                            pred_singles_masked, _ = timed_call(predictor.infer_all_singles, pdb, chain='A', subset_df=singles, backbone_mutation=backbone_mutation, use_masks=args.use_masks) 
                            pred_singles_masked['id'] = pred_singles_masked['code'] + ('_' + backbone_mutation if backbone_mutation else '') + '_' + pred_singles_masked['wild_type'] + pred_singles_masked['seq_pos'].astype(int).astype(str) + pred_singles_masked['mutation']
                            pred_combined_masked = pd.concat([pred_combined_masked, pred_singles_masked])

                        if not doubles.empty:
                            pred_doubles_masked, _ = timed_call(predictor.infer_all_doubles, pdb, chain='A', subset_df=doubles, backbone_mutation=backbone_mutation, use_masks=args.use_masks) 
                            pred_doubles_masked['id'] = pred_doubles_masked['code'] + ('_' + backbone_mutation if backbone_mutation else '') + '_' + pred_doubles_masked['wt1'] + pred_doubles_masked['pos1'].astype(int).astype(str) + pred_doubles_masked['mut1'] + ':' + pred_doubles_masked['wt2'] + pred_doubles_masked['pos2'].astype(int).astype(str) + pred_doubles_masked['mut2']
                            pred_combined_masked = pd.concat([pred_combined_masked, pred_doubles_masked])

                        if 'id' in pred_combined_masked.columns:
                            res_partial_masked = pred_combined_masked.set_index('id')
                        else:
                            res_partial_masked = pred_combined_masked

                        res_masked_list.append(res_partial_masked)
                    
                    if res_masked_list:
                        results_masked.append(pd.concat(res_masked_list))

                results_masked_df = pd.concat(results_masked, axis=0)
                os.makedirs(out_dir, exist_ok=True)
                results_masked_df.to_csv(out_dir / f'predictions{"_masked" if args.use_masks else "_unmasked"}.csv')
                tsu_dfs[f"{split_name}-{scaffold_}"] = results_masked_df
            else:
                tsu_dfs[f"{split_name}-{scaffold_}"] = pd.read_csv(out_dir / f'predictions{"_masked" if args.use_masks else "_unmasked"}.csv', index_col=0)
            
            calculate_and_save_stats(tsu_dfs, CHECKPOINT, f'{split_name}_{scaffold_}', args)

    # ================= DMS PROCESSING =================
    if not args.skip_dms:
        prots = ['GB1_Wu_2016_binding_domain'] #'DLG4_HUMAN_Faure_2021_abundance_domain', 'DLG4_HUMAN_Faure_2021_binding_domain', 'GRB2_HUMAN_Faure_2021_abundance_domain', 'GRB2_HUMAN_Faure_2021_binding_domain', 'MYO_HUMAN_Kung_2025_display', 'ESTA_BACSU_Nutschel_2020_dTm', 
        dms_dfs = {}
    
        for prot in prots:
            out_dir = REPO_ROOT / 'analysis_notebooks' / f'predictions/{prot}/esm3_forge/{CHECKPOINT}'
            if os.path.exists(out_dir / f'predictions{"_masked" if args.use_masks else "_unmasked"}.csv'):
                res_masked = pd.read_csv(out_dir / f'predictions{"_masked" if args.use_masks else "_unmasked"}.csv', index_col=0)
            else:          
                df_true = pd.read_csv(DATA_DIR / f'{prot}.csv')
                df_true['id'] = df_true['code'] + '_' + df_true['mut_info']
                df_true = df_true.set_index('id')

                has_doubles = len(df_true.loc[df_true['mut_info'].str.contains(':')]) > 0
                if has_doubles:
                    df_true = utils.sum_individual_mutation_scores(df_true, 'ddG_ML', new_score_column='ddG_additive_ML')
                    df_true['dddG_ML'] = df_true['ddG_ML'] - df_true['ddG_additive_ML']

                singles = df_true.loc[~df_true['mut_info'].str.contains(':')].copy()
                doubles = df_true.loc[df_true['mut_info'].str.count(':') == 1].copy()
                multi = df_true.loc[df_true['mut_info'].str.count(':') >= 2].copy()

                prot_name = '_'.join(prot.split('_')[:2])
                if prot_name == 'GB1_Wu':
                    prot_name = 'GB1'

                pred_combined_masked = pd.DataFrame()

                if not singles.empty:
                    singles['wild_type'] = singles['mut_info'].str[0]
                    singles['seq_pos'] = singles['mut_info'].str[1:-1]
                    singles['mutation'] = singles['mut_info'].str[-1]
                    pred_singles_masked, _ = timed_call(predictor.infer_all_singles, REPO_ROOT / f'data/structures/{prot_name}.pdb', subset_df=singles, use_masks=args.use_masks)
                    pred_singles_masked['id'] = pred_singles_masked['code'] + '_' + pred_singles_masked['mut_info']
                    pred_combined_masked = pd.concat([pred_combined_masked, pred_singles_masked])
                    #pred_singles_masked.set_index('id')
                    #pred_singles_masked.to_csv(out_dir / f'predictions{"_masked" if args.use_masks else "_unmasked"}_singles.csv')
                    #continue

                if not doubles.empty:
                    doubles = utils.parse_multimutant_column(doubles, 'mut_type')
                    pred_doubles_masked, _ = timed_call(predictor.infer_all_doubles, REPO_ROOT / f'data/structures/{prot_name}.pdb', subset_df=doubles, use_masks=args.use_masks) 
                    pred_doubles_masked['id'] = pred_doubles_masked['code'] + '_' + pred_doubles_masked['mut_info']
                    pred_combined_masked = pd.concat([pred_combined_masked, pred_doubles_masked])
                
                if not multi.empty:
                    multi = utils.parse_multimutant_column(multi, 'mut_info')
                    pred_multi_masked, _ = timed_call(predictor.infer_multimutants_sampled, REPO_ROOT / f'data/structures/{prot_name}.pdb', subset_df=multi, use_masks=args.use_masks)
                    if "original_index" in pred_multi_masked.columns:
                        pred_multi_masked.index = pred_multi_masked["original_index"]
                    pred_multi_masked['id'] = pred_multi_masked.index
                    pred_combined_masked = pd.concat([pred_combined_masked, pred_multi_masked])

                res_masked = pred_combined_masked.set_index('id') if 'id' in pred_combined_masked.columns else pred_combined_masked
                
                os.makedirs(out_dir, exist_ok=True)
                res_masked.to_csv(out_dir / f'predictions{"_masked" if args.use_masks else "_unmasked"}.csv')

            dms_dfs[prot] = res_masked
            
        calculate_and_save_stats(dms_dfs, CHECKPOINT, 'dms', args)

    # ================= DOMAINOME PROCESSING =================
    if not args.skip_domainome:
        path = REPO_ROOT / f'data/domainome1/domainome_mapped_2026.csv'
        df = pd.read_csv(path)
        df['code'] = df['domain_ID'].apply(lambda x: x.replace('/', '_'))
        df['ddG_ML'] = df['scaled_fitness']
        df['wild_type'] = df['mut_type'].str[0]
        df['seq_pos'] = df['mut_type'].str[1:-1]
        df['mutation'] = df['mut_type'].str[-1]
        df = df.dropna(subset=['pdb_file', 'seq_pos'])
        df = df[['code', 'mut_type', 'wild_type', 'seq_pos', 'mutation', 'uniprot_ID', 'domain_ID', 'pdb_file', 'ddG_ML']]
        df['chain'] = 'A'
        
        dom_dfs = {}
        out_dir = REPO_ROOT / 'analysis_notebooks' / f'predictions/domainome/esm3_forge/{CHECKPOINT}'
        
        if os.path.exists(out_dir / f'predictions{"_masked" if args.use_masks else "_unmasked"}.csv'):
            results_masked_out = pd.read_csv(out_dir / f'predictions{"_masked" if args.use_masks else "_unmasked"}.csv', index_col=0)
        else:
            results_masked = []
            for prot in tqdm(df['code'].unique()):
                df_true = df.loc[df['code']==prot].copy()
                df_true['id'] = df_true['domain_ID'] + '_' + df_true['mut_type']
                df_true = df_true.set_index('id')

                pdb = df_true['pdb_file'].head(1).item()

                pred_singles_masked, _ = timed_call(predictor.infer_all_singles, pdb, subset_df=df_true, use_masks=args.use_masks)
                pred_singles_masked['id'] = pred_singles_masked['domain_ID'] + '_' + pred_singles_masked['mut_type']
                res_masked = pred_singles_masked

                results_masked.append(res_masked)

            results_masked_out = pd.concat(results_masked, axis=0)
            os.makedirs(out_dir, exist_ok=True)
            results_masked_out.to_csv(out_dir / f'predictions{"_masked" if args.use_masks else "_unmasked"}.csv')
            
        dom_dfs['DOMAINOME'] = results_masked_out
        calculate_and_save_stats(dom_dfs, CHECKPOINT, 'Domainome', args)

    # ================= FUNCTIONAL (DMS ASSAYS) =================
    if not args.skip_functional:
        test_list_DMS = ['D7PM05_CLYGR', 'GFP_AEQVI', 'HIS7_YEAST', 'Q6WV12_9MAXI', 'Q8WTC7_9CNID', 'RASK_HUMAN']
        func_dfs = {}

        for mem_size, prot in zip([8,8,8,8,8,8], test_list_DMS):
            out_dir = REPO_ROOT / 'analysis_notebooks' / f'predictions/{prot}/esm3_forge/{CHECKPOINT}'
            if os.path.exists(out_dir / f'predictions{"_masked" if args.use_masks else "_unmasked"}.csv'):
                res = pd.read_csv(out_dir / f'predictions{"_masked" if args.use_masks else "_unmasked"}.csv', index_col=0)
            else:
                df_true = pd.read_csv(f'/home/{"sareeves" if not args.local_cluster else "sreeves"}/PSLMs/data/lora/DMS/csv_formatted/{prot}.csv')
                df_true['mut_type'] = df_true['MUTS'].apply(lambda x: x.replace(';', ':'))
                df_true['id'] = df_true['code'] + '_' + df_true['mut_type']
                df_true = df_true.set_index('id')
                df_true = utils.parse_multimutant_column(df_true, mut_column='mut_type')
                
                pdb = df_true['pdb_file'].head(1).item()

                singles = df_true.loc[~df_true['mut_type'].str.contains(':')].copy()
                doubles = df_true.loc[df_true['mut_type'].str.count(':') == 1].copy()
                multi = df_true.loc[df_true['mut_type'].str.count(':') >= 2].copy()
                
                pred_combined = pd.DataFrame()
                
                if not singles.empty:
                    # FLAW FIX: Map wt1 -> wild_type to prevent KeyError inside infer_all_singles
                    singles['wild_type'] = singles['wt1']
                    singles['seq_pos'] = singles['pos1']
                    singles['mutation'] = singles['mut1']
                    pred_singles = predictor.infer_all_singles(pdb, subset_df=singles, use_masks=args.use_masks)
                    pred_singles['id'] = pred_singles.index
                    pred_combined = pd.concat([pred_combined, pred_singles])

                if not doubles.empty:
                    pred_doubles = predictor.infer_all_doubles(pdb, subset_df=doubles, use_masks=args.use_masks)
                    pred_doubles['id'] = pred_doubles.index
                    pred_combined = pd.concat([pred_combined, pred_doubles])
                    
                if not multi.empty:
                    pred_multi = predictor.infer_multimutants_sampled(pdb, subset_df=multi, use_masks=args.use_masks)
                    if "original_index" in pred_multi.columns:
                        pred_multi.index = pred_multi["original_index"]
                    pred_multi['id'] = pred_multi.index
                    pred_combined = pd.concat([pred_combined, pred_multi])
            
                pred_combined = pred_combined.set_index('id') if 'id' in pred_combined.columns else pred_combined
                
                # FLAW FIX: Safe join that drops conflicting overlapping columns
                cols_to_drop = [c for c in df_true.columns if c in pred_combined.columns and c != 'id']
                res = df_true.join(pred_combined.drop(columns=cols_to_drop, errors='ignore'))
                
                os.makedirs(out_dir, exist_ok=True)
                res.to_csv(out_dir / f'predictions{"_masked" if args.use_masks else "_unmasked"}.csv')
                
            func_dfs[prot] = res
            
        calculate_and_save_stats(func_dfs, CHECKPOINT, 'Functional', args)


if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_name', type=str, default='esm3-small-2024-08')
        parser.add_argument('--split', type=str, default='hyperopt_splits')
        parser.add_argument('--local_cluster', action='store_true')
        parser.add_argument('--use_masks', action='store_true', default=False, help='Use masked trajectories instead of WT/MUT unmasked inference')
        #parser.add_argument('--mask_sequence_pos', type=bool, default=True)
        #parser.add_argument('--mask_structure_pos', action='store_true')
        #parser.add_argument('--mask_coords_pos', action='store_true')
        #parser.add_argument('--mask_coords', action='store_true')
        #parser.add_argument('--regenerate_results', action='store_true')
        parser.add_argument('--skip_external', action='store_true')
        parser.add_argument('--skip_tsuboyama', action='store_true')
        parser.add_argument('--skip_dms', action='store_true')
        parser.add_argument('--skip_functional', action='store_true')
        parser.add_argument('--skip_domainome', action='store_true')

        args, remaining_argv = parser.parse_known_args()

        ranking_ns = argparse.Namespace()
        regression_ns = argparse.Namespace()
        current_remaining_argv = list(remaining_argv) 

        if current_remaining_argv:
            parser.error(f"unrecognized arguments: {' '.join(current_remaining_argv)}")

        args.ranking_config = ranking_ns
        args.ranking_config = vars(args.ranking_config)

        if args.skip_external:
            print('Skipping benchmark datasets!')
        if args.skip_tsuboyama:
            print('Skipping MegaScale validation and testing datasets!')
        if args.skip_functional:
            print('Skipping double mutant DMS assays!')
        if args.skip_domainome:
            print('Skipping domainome VAMP assays!')
        #if args.mask_structure_pos or args.mask_coords_pos:
        #    print('Masking one or more inputs!')
        if not args.split:
            print('Warning! Not using any specific split file!')

        main_(args)