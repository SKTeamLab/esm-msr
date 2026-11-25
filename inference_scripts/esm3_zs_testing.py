import os
import re
import sys
import json
import time
import math
import base64
import random
import asyncio
import logging
import argparse
from pathlib import Path
from collections import deque, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import httpx
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import entropy, spearmanr
from scipy.special import softmax

from esm_msr import utils, preprocess_megascale
from esm.utils.structure.protein_chain import ProteinChain
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
from esm.utils.misc import maybe_list, deserialize_tensors

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "preprocessed"
FORGE_DEFAULT_URL = "https://forge.evolutionaryscale.ai"

_CAP_RE = re.compile(r"usage cap of\s+(\d+)\s+tokens", re.IGNORECASE)


def resolve_forge_token(
    cli_token: Optional[str] = None,
    token_file: Optional[str] = None,
) -> str:
    """Resolve Forge API token from CLI, file, or environment (never hard-coded)."""
    if cli_token and cli_token.strip():
        return cli_token.strip()
    if token_file:
        p = Path(token_file).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"Forge token file not found: {p}")
        tok = p.read_text().strip()
        if tok:
            return tok
    for env_var in ("ESM_API_KEY", "FORGE_API_KEY", "ESM_FORGE_TOKEN"):
        val = os.environ.get(env_var, "").strip()
        if val:
            return val
    raise RuntimeError(
        "No Forge API token supplied. Pass --forge_token / --token_file or set ESM_API_KEY."
    )


def resolve_pdb_path(pdb_path: str, structures_dir: Optional[str] = None, base_dir: Optional[Path] = None) -> str:
    """Resolve a PDB path across local/cluster home-dir prefixes and fallback structure folders."""
    p = str(pdb_path).strip()
    if os.path.isfile(p):
        return p
    if base_dir is not None and not os.path.isabs(p):
        cand = base_dir / p
        if cand.is_file():
            return str(cand.resolve())
    for src, dst in (("/home/sareeves/", "/home/sreeves/"), ("/home/sreeves/", "/home/sareeves/")):
        if src in p:
            cand = p.replace(src, dst, 1)
            if os.path.isfile(cand):
                return cand
    bn = os.path.basename(p)
    fallbacks = []
    if structures_dir:
        fallbacks.append(Path(structures_dir) / bn)
    fallbacks.extend([
        REPO_ROOT / "data" / "structures" / bn,
        Path.home() / "software" / "esm-msr" / "data" / "structures" / bn,
        REPO_ROOT / "data" / "tsuboyama" / "AlphaFold_model_PDBs" / bn,
        REPO_ROOT / "data" / "domainome1" / "spurs_structures" / bn,
    ])
    for cand in fallbacks:
        if cand.is_file():
            return str(cand)
    return p


def parse_mutation_string(mut_str: str) -> List[Tuple[str, int, str]]:
    """Parse 'A12C:D15E' or 'A12C;D15E' into [(wt, pos_1based, mut), ...]."""
    if pd.isna(mut_str):
        return []
    s = str(mut_str).strip()
    if not s or s.upper() == "WT":
        return []
    out = []
    for part in re.split(r"[:;]", s):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^([A-Za-z])(\d+)([A-Za-z])$", part)
        if not m:
            return []
        out.append((m.group(1).upper(), int(m.group(2)), m.group(3).upper()))
    return out


def timed_call(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


class AdaptiveTokenLimiter:
    """Sliding-window token-bucket rate limiter that auto-tunes to Forge 429 caps."""

    def __init__(self, tokens_per_min: Optional[int] = None, window_sec: float = 60.5):
        self.window_sec = float(window_sec)
        self.tokens_per_min: Optional[int] = int(tokens_per_min) if tokens_per_min and tokens_per_min > 0 else None
        self._events: deque = deque()  # (monotonic_ts, n_tokens)
        self._window_tokens: int = 0
        self._lock = asyncio.Lock()
        self._cooldown_until: float = 0.0

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_sec
        while self._events and self._events[0][0] < cutoff:
            _, n = self._events.popleft()
            self._window_tokens -= n

    async def acquire(self, n_tokens: int) -> None:
        n_tokens = max(1, int(n_tokens))
        while True:
            sleep_for = 0.0
            async with self._lock:
                now = time.monotonic()
                if now < self._cooldown_until:
                    sleep_for = self._cooldown_until - now
                else:
                    self._prune(now)
                    cap = self.tokens_per_min
                    if cap is None or self._window_tokens + n_tokens <= cap or not self._events:
                        self._events.append((now, n_tokens))
                        self._window_tokens += n_tokens
                        return
                    # Wait until the oldest event drops out of the sliding window.
                    oldest_ts = self._events[0][0]
                    sleep_for = max(0.05, (oldest_ts + self.window_sec) - now)
            await asyncio.sleep(min(sleep_for, 2.0))

    async def note_rate_limit(self, err_text: str, retry_after: Optional[float] = None) -> None:
        m = _CAP_RE.search(err_text or "")
        async with self._lock:
            now = time.monotonic()
            if m:
                detected_cap = int(m.group(1))
                target = max(1000, int(detected_cap * 0.94))
                if self.tokens_per_min is None or target < self.tokens_per_min:
                    self.tokens_per_min = target
                    print(
                        f"[FORGE-LIMIT] Auto-detected server cap={detected_cap:,} tok/min -> pacing at {target:,} tok/min",
                        flush=True,
                    )
            wait = float(retry_after) if retry_after is not None else 1.5
            self._cooldown_until = max(self._cooldown_until, now + wait)


class ESM3ForgePredictor:
    """High-throughput async Forge client using cached WT /encode and independent masking."""

    def __init__(
        self,
        token: str,
        model_name: str,
        tokenizer=None,
        url: str = FORGE_DEFAULT_URL,
        max_concurrency: int = 48,
        tokens_per_min: Optional[int] = None,
        max_retries: int = 15,
        request_timeout: float = 120.0,
    ):
        self.token = token
        self.model_name = model_name
        self.url = url.rstrip("/")
        self.tokenizer = tokenizer or EsmSequenceTokenizer("cpu")
        self.max_concurrency = max(1, int(max_concurrency))
        self.max_retries = max(1, int(max_retries))
        self.request_timeout = float(request_timeout)
        self.tokens_per_min = tokens_per_min

        if hasattr(self.tokenizer, "get_vocab"):
            self.vocab = self.tokenizer.get_vocab()
        elif hasattr(self.tokenizer, "vocab"):
            self.vocab = self.tokenizer.vocab
        else:
            raise ValueError("Tokenizer must have .get_vocab() or .vocab attribute")
        self.mask_token_id = int(self.vocab.get("<mask>", self.vocab.get("_", 32)))

        # Runtime stats
        self.total_encode_calls = 0
        self.total_logits_calls = 0
        self.total_tokens_sent = 0
        self.total_429s = 0
        self.start_wall = time.perf_counter()

        # Per-protein WT encode cache: (pdb_path, chain_id, backbone_mutation) -> dict
        self._encode_cache: Dict[Tuple[str, str, Optional[str]], dict] = {}
        # Persistent limiter across calls in same process
        self._limiter = AdaptiveTokenLimiter(tokens_per_min=tokens_per_min)

    def _headers(self, return_bytes: bool = False) -> Dict[str, str]:
        h = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        if return_bytes:
            h["return-bytes"] = "true"
        return h

    async def _post_json(
        self,
        http_client: httpx.AsyncClient,
        sem: asyncio.Semaphore,
        endpoint: str,
        payload: dict,
        n_tokens: int,
        return_bytes: bool = False,
    ) -> Optional[dict]:
        full_url = f"{self.url}/api/v1/{endpoint}"
        headers = self._headers(return_bytes=return_bytes)
        payload["potential_sequence_of_concern"] = False

        for attempt in range(self.max_retries):
            await self._limiter.acquire(n_tokens)
            async with sem:
                try:
                    resp = await http_client.post(
                        full_url,
                        json=payload,
                        headers=headers,
                        timeout=self.request_timeout,
                    )
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    backoff = min(30.0, (1.5 ** attempt) + random.uniform(0.1, 0.8))
                    if attempt == self.max_retries - 1:
                        print(f"[FORGE-ERR] {endpoint} transport failed after {self.max_retries} tries: {exc}", flush=True)
                        return None
                    await asyncio.sleep(backoff)
                    continue

            if resp.status_code == 200:
                data = resp.json()
                if endpoint == "encode":
                    self.total_encode_calls += 1
                    if "outputs" not in data and "data" in data:
                        data = data["data"]
                else:
                    self.total_logits_calls += 1
                    self.total_tokens_sent += n_tokens
                    if "logits" not in data and "data" in data:
                        data = data["data"]
                return data

            text = resp.text[:400]
            if resp.status_code == 429:
                self.total_429s += 1
                ra = resp.headers.get("Retry-After")
                retry_after = float(ra) if ra and ra.replace(".", "", 1).isdigit() else None
                await self._limiter.note_rate_limit(text, retry_after=retry_after)
                backoff = min(30.0, (retry_after or (1.2 + 0.5 * attempt)) + random.uniform(0.1, 0.9))
                await asyncio.sleep(backoff)
                continue

            if resp.status_code in (500, 502, 503, 504, 408):
                backoff = min(30.0, (1.6 ** attempt) + random.uniform(0.1, 0.8))
                if attempt == self.max_retries - 1:
                    print(f"[FORGE-ERR] {endpoint} HTTP {resp.status_code}: {text}", flush=True)
                    return None
                await asyncio.sleep(backoff)
                continue

            # Permanent 4xx error (e.g., sequence length limit, invalid input)
            print(f"[FORGE-PERM-ERR] {endpoint} HTTP {resp.status_code}: {text}", flush=True)
            return None

        return None

    async def _encode_wt(
        self,
        http_client: httpx.AsyncClient,
        sem: asyncio.Semaphore,
        pdb_path: str,
        chain_id: str,
        backbone_mutation: Optional[str] = None,
    ) -> Optional[dict]:
        key = (pdb_path, chain_id, backbone_mutation)
        if key in self._encode_cache:
            return self._encode_cache[key]

        chain_obj = ProteinChain.from_pdb(pdb_path, chain_id)
        wt_seq = chain_obj.sequence
        coords = torch.tensor(chain_obj.atom37_positions)

        if backbone_mutation:
            wt = backbone_mutation[0]
            pos = int(backbone_mutation[1:-1])
            mut = backbone_mutation[-1]
            if wt_seq[pos - 1] != wt:
                raise AssertionError(f"Backbone mutation mismatch on {pdb_path}: expected {wt} at {pos}, got {wt_seq[pos - 1]}")
            seq_l = list(wt_seq)
            seq_l[pos - 1] = mut
            wt_seq = "".join(seq_l)

        payload = {
            "model": self.model_name,
            "inputs": {
                "sequence": wt_seq,
                "secondary_structure": None,
                "sasa": None,
                "coordinates": maybe_list(coords, convert_nan_to_none=True),
            },
        }
        data = await self._post_json(http_client, sem, "encode", payload, n_tokens=len(wt_seq) + 2, return_bytes=False)
        if not data or "outputs" not in data:
            return None

        outputs = data["outputs"]
        encoded = {
            "wt_seq": wt_seq,
            "sequence": list(outputs["sequence"]),
            "structure": outputs["structure"],
            "coordinates": outputs["coordinates"],
            "n_tokens": len(outputs["sequence"]),
        }
        self._encode_cache[key] = encoded
        return encoded

    async def _fetch_state_positions(
        self,
        http_client: httpx.AsyncClient,
        sem: asyncio.Semaphore,
        encoded_wt: dict,
        overrides: Tuple[Tuple[int, int], ...],
        positions_needed: Sequence[int],
    ) -> Optional[Dict[int, torch.Tensor]]:
        """Query /api/v1/logits once for a masked sequence state and extract only requested positions."""
        seq_tokens = list(encoded_wt["sequence"])
        for pos_1based, tok_id in overrides:
            seq_tokens[pos_1based] = tok_id

        payload = {
            "model": self.model_name,
            "inputs": {
                "sequence": seq_tokens,
                "structure": encoded_wt["structure"],
                "secondary_structure": None,
                "sasa": None,
                "function": None,
                "coordinates": encoded_wt["coordinates"],
                "residue_annotation": None,
            },
            "logits_config": {
                "sequence": True,
                "structure": False,
                "secondary_structure": False,
                "sasa": False,
                "function": False,
                "residue_annotations": False,
                "return_embeddings": False,
                "return_hidden_states": False,
                "ith_hidden_layer": -1,
            },
        }
        data = await self._post_json(
            http_client, sem, "logits", payload, n_tokens=encoded_wt["n_tokens"], return_bytes=True
        )
        if not data or data.get("logits") is None:
            return None

        logits_raw = data["logits"]
        if isinstance(logits_raw, str):
            logits_dict = deserialize_tensors(base64.b64decode(logits_raw))
            seq_logits = logits_dict["sequence"]
        else:
            seq_logits = torch.tensor(logits_raw["sequence"])
        if seq_logits.ndim == 3:
            seq_logits = seq_logits[0]
        seq_logits = seq_logits.detach().cpu().float()

        return {p: seq_logits[p].clone() for p in positions_needed}

    async def _infer_independent_async(
        self,
        pdb_path: str,
        subset_df: pd.DataFrame,
        chain_id: str = "A",
        backbone_mutation: Optional[str] = None,
        mut_col: Optional[str] = None,
        skip_reverse: bool = False,
        compute_entropy: bool = True,
        quiet: bool = False,
        http_client: Optional[httpx.AsyncClient] = None,
        sem: Optional[asyncio.Semaphore] = None,
    ) -> pd.DataFrame:
        """Score arbitrary K>=1 mutants on a single structure via deduplicated independent masking.

        Formulation (exact match to src/esm_msr/models.py `mask_strategy='independent'`):
          - WT pass (`wt_lora_pred` / `pred_additive`):
              For each mutated site k, mask p_k in WT -> sum_k [logit(m_k) - logit(w_k)].
          - MT pass (`mt_lora_pred`, skipped if `skip_reverse=True`):
              For each mutated site k, set all j != k to m_j and mask p_k -> sum_k [logit(m_k) - logit(w_k)].
          - Combined (`combined_pred` / `esm3_score`):
              0.5 * (wt + mt) when K >= 2 and not skip_reverse; else wt.
          - Epistasis (`esm3_epistasis_score` / `epi_pred`):
              combined - wt (identically 0.0 for K == 1 or skip_reverse).
        For K == 1 and K == 2, this is algebraically identical to the previous Forge singles/doubles scorer.
        """
        out_df = subset_df.copy()
        for col in (
            "wt_lora_pred",
            "mt_lora_pred",
            "combined_pred",
            "pred_additive",
            "esm3_score",
            "esm3_epistasis_score",
            "epi_pred",
            "esm3_logits_entropy",
        ):
            out_df[col] = np.nan

        if subset_df.empty:
            return out_df

        # Determine mutation tuples per row
        if mut_col is None:
            for cand in ("mut_type_renumbered", "mut_type", "mut_info_seq_pos", "mut_info", "mutant"):
                if cand in subset_df.columns:
                    mut_col = cand
                    break
        if mut_col is None:
            raise ValueError(f"Could not find a mutation column in columns={list(subset_df.columns)}")

        if http_client is None or sem is None:
            limits = httpx.Limits(
                max_connections=max(64, self.max_concurrency * 2),
                max_keepalive_connections=max(32, self.max_concurrency),
            )
            local_sem = asyncio.Semaphore(self.max_concurrency)
            async with httpx.AsyncClient(limits=limits) as local_client:
                return await self._infer_independent_async(
                    pdb_path=pdb_path,
                    subset_df=subset_df,
                    chain_id=chain_id,
                    backbone_mutation=backbone_mutation,
                    mut_col=mut_col,
                    skip_reverse=skip_reverse,
                    compute_entropy=compute_entropy,
                    quiet=quiet,
                    http_client=local_client,
                    sem=local_sem,
                )

        encoded_wt = await self._encode_wt(http_client, sem, pdb_path, chain_id, backbone_mutation)
        if encoded_wt is None:
            return out_df
        wt_seq = encoded_wt["wt_seq"]
        L = len(wt_seq)

        # Phase A: deduplicate all unique masked states across singles, doubles, and multis
        state_positions: Dict[Tuple[Tuple[int, int], ...], set] = defaultdict(set)
        row_specs = []  # (df_idx, muts, wt_keys, mt_keys)

        for df_idx, row in subset_df.iterrows():
            muts = parse_mutation_string(row[mut_col])
            if not muts:
                continue
            valid = True
            for wt_aa, pos, mut_aa in muts:
                if not (1 <= pos <= L):
                    valid = False
                    break
                if self.vocab.get(wt_aa) is None or self.vocab.get(mut_aa) is None:
                    valid = False
                    break
            if not valid:
                continue

            wt_keys = []
            for _, pos, _ in muts:
                st = ((pos, self.mask_token_id),)
                wt_keys.append(st)
                state_positions[st].add(pos)

            mt_keys = None
            if not skip_reverse:
                if len(muts) == 1:
                    mt_keys = wt_keys
                else:
                    mt_keys = []
                    for i, (_, pos_i, _) in enumerate(muts):
                        ov = {pos_j: int(self.vocab[mut_j]) for j, (_, pos_j, mut_j) in enumerate(muts) if j != i}
                        ov[pos_i] = self.mask_token_id
                        st = tuple(sorted(ov.items()))
                        mt_keys.append(st)
                        state_positions[st].add(pos_i)

            row_specs.append((df_idx, muts, wt_keys, mt_keys))

        unique_states = list(state_positions.items())
        if not quiet:
            print(
                f"  [FORGE] {os.path.basename(pdb_path)} L={L} rows={len(row_specs)} "
                f"unique_states={len(unique_states)} (~{len(unique_states) * encoded_wt['n_tokens']:,} tok)",
                flush=True,
            )

        # Phase B: execute all deduplicated /logits calls concurrently on shared client
        state_results: Dict[Tuple[Tuple[int, int], ...], Optional[Dict[int, torch.Tensor]]] = {}

        async def _run_one(st, pos_set):
            res = await self._fetch_state_positions(http_client, sem, encoded_wt, st, tuple(sorted(pos_set)))
            state_results[st] = res

        chunk_sz = max(200, self.max_concurrency * 16)
        for start in range(0, len(unique_states), chunk_sz):
            batch = unique_states[start : start + chunk_sz]
            await asyncio.gather(*(_run_one(st, pset) for st, pset in batch))

        # Phase C: assemble per-row scores
        ent_cache: Dict[Tuple[Tuple[Tuple[int, int], ...], int], float] = {}
        for df_idx, muts, wt_keys, mt_keys in row_specs:
            wt_sum = 0.0
            wt_ok = True
            for (wt_aa, pos, mut_aa), st in zip(muts, wt_keys):
                pos_map = state_results.get(st)
                if not pos_map or pos not in pos_map:
                    wt_ok = False
                    break
                logits_p = pos_map[pos]
                wt_sum += (logits_p[self.vocab[mut_aa]] - logits_p[self.vocab[wt_aa]]).item()
            if not wt_ok:
                continue

            out_df.at[df_idx, "wt_lora_pred"] = wt_sum
            out_df.at[df_idx, "pred_additive"] = wt_sum

            if compute_entropy and len(muts) == 1:
                st0, pos0 = wt_keys[0], muts[0][1]
                ek = (st0, pos0)
                if ek not in ent_cache:
                    probs = softmax(state_results[st0][pos0].numpy())
                    ent_cache[ek] = float(entropy(probs, base=2))
                out_df.at[df_idx, "esm3_logits_entropy"] = ent_cache[ek]

            if skip_reverse:
                if len(muts) == 1:
                    out_df.at[df_idx, "mt_lora_pred"] = wt_sum
                out_df.at[df_idx, "combined_pred"] = wt_sum
                out_df.at[df_idx, "esm3_score"] = wt_sum
                out_df.at[df_idx, "esm3_epistasis_score"] = 0.0
                out_df.at[df_idx, "epi_pred"] = 0.0
                continue

            if len(muts) == 1:
                out_df.at[df_idx, "mt_lora_pred"] = wt_sum
                out_df.at[df_idx, "combined_pred"] = wt_sum
                out_df.at[df_idx, "esm3_score"] = wt_sum
                out_df.at[df_idx, "esm3_epistasis_score"] = 0.0
                out_df.at[df_idx, "epi_pred"] = 0.0
            else:
                mt_sum = 0.0
                mt_ok = True
                for (wt_aa, pos, mut_aa), st in zip(muts, mt_keys):
                    pos_map = state_results.get(st)
                    if not pos_map or pos not in pos_map:
                        mt_ok = False
                        break
                    logits_p = pos_map[pos]
                    mt_sum += (logits_p[self.vocab[mut_aa]] - logits_p[self.vocab[wt_aa]]).item()
                if mt_ok:
                    comb = 0.5 * (wt_sum + mt_sum)
                    epi = comb - wt_sum
                    out_df.at[df_idx, "mt_lora_pred"] = mt_sum
                    out_df.at[df_idx, "combined_pred"] = comb
                    out_df.at[df_idx, "esm3_score"] = comb
                    out_df.at[df_idx, "esm3_epistasis_score"] = epi
                    out_df.at[df_idx, "epi_pred"] = epi

        return out_df

    def _run_async(self, coroutine):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
        return loop.run_until_complete(coroutine)

    def infer_mutants_independent(
        self,
        pdb_path: str,
        subset_df: pd.DataFrame,
        chain: str = "A",
        backbone_mutation: Optional[str] = None,
        mut_col: Optional[str] = None,
        skip_reverse: bool = False,
        compute_entropy: bool = True,
        quiet: bool = False,
    ) -> pd.DataFrame:
        return self._run_async(
            self._infer_independent_async(
                pdb_path=pdb_path,
                subset_df=subset_df,
                chain_id=chain,
                backbone_mutation=backbone_mutation,
                mut_col=mut_col,
                skip_reverse=skip_reverse,
                compute_entropy=compute_entropy,
                quiet=quiet,
            )
        )

    def infer_many_proteins_independent(
        self,
        jobs: Sequence[dict],
        protein_concurrency: int = 16,
        on_done=None,
    ) -> List[pd.DataFrame]:
        """Run many protein-level jobs over ONE persistent HTTP keep-alive client to keep concurrency saturated."""
        async def _run_all():
            limits = httpx.Limits(
                max_connections=max(64, self.max_concurrency * 2),
                max_keepalive_connections=max(32, self.max_concurrency),
            )
            req_sem = asyncio.Semaphore(self.max_concurrency)
            prot_sem = asyncio.Semaphore(max(1, int(protein_concurrency)))
            results: List[Optional[pd.DataFrame]] = [None] * len(jobs)

            async with httpx.AsyncClient(limits=limits) as http_client:
                async def _worker(idx: int, job: dict):
                    async with prot_sem:
                        res = await self._infer_independent_async(
                            pdb_path=job["pdb_path"],
                            subset_df=job["subset_df"],
                            chain_id=job.get("chain", "A"),
                            backbone_mutation=job.get("backbone_mutation"),
                            mut_col=job.get("mut_col"),
                            skip_reverse=job.get("skip_reverse", False),
                            compute_entropy=job.get("compute_entropy", True),
                            quiet=job.get("quiet", True),
                            http_client=http_client,
                            sem=req_sem,
                        )
                        results[idx] = res
                        if on_done is not None:
                            on_done(idx, job, res)

                await asyncio.gather(*(_worker(i, j) for i, j in enumerate(jobs)))
            return results

        return self._run_async(_run_all())

    def log_throughput(self, prefix: str = "") -> None:
        dt = max(1e-6, time.perf_counter() - self.start_wall)
        rpm = self.total_logits_calls / dt * 60.0
        tpm = self.total_tokens_sent / dt * 60.0
        print(
            f"{prefix}[FORGE-STATS] elapsed={dt/60:.1f}m encodes={self.total_encode_calls:,} "
            f"logits={self.total_logits_calls:,} ({rpm:.0f} req/min, {tpm:,.0f} tok/min) "
            f"429s={self.total_429s} pace_cap={self._limiter.tokens_per_min}",
            flush=True,
        )


def _load_reuse_domain_cache(reuse_paths: Sequence[str]) -> Dict[str, pd.DataFrame]:
    """Load already-scored Tsuboyama/hyperopt predictions keyed by code_wt (or code) to avoid re-querying."""
    cache: Dict[str, pd.DataFrame] = {}
    for rp in reuse_paths:
        if not rp or not os.path.isfile(rp):
            continue
        df = pd.read_csv(rp, low_memory=False)
        if "esm3_score" not in df.columns:
            continue
        code_col = "code_wt" if "code_wt" in df.columns else ("code" if "code" in df.columns else None)
        if not code_col:
            continue
        for c, sub in df.groupby(code_col):
            if sub["esm3_score"].notna().all():
                cache[str(c)] = sub
    return cache


def run_ptmul(predictor: ESM3ForgePredictor, args, out_root: Path):
    """Run independent-masked Forge inference on PTMUL (all rows or k>=3 update)."""
    out_dir = out_root / "ptmul"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "ptmul.csv"
    if out_csv.is_file() and not args.overwrite:
        print(f"[PTMUL] Already complete: {out_csv}", flush=True)
        return

    df_true = pd.read_csv(DATA_DIR / "ptmul_mapped.csv")
    df_true = df_true.reset_index(drop=True)
    df_true = utils.sort_mutations_by_position(df_true, "mut_info_seq_pos", "mut_type")
    df_true["id"] = df_true["code"].astype(str) + df_true["chain"].astype(str) + "_" + df_true["mut_type"].astype(str)
    df_true = df_true.set_index("id")

    parts_dir = out_dir / "per_pdb"
    parts_dir.mkdir(exist_ok=True)
    dfs = []
    for (pdb, code, chain), data in tqdm(df_true.groupby(["pdb_file", "code", "chain"]), desc="PTMUL"):
        safe = f"{code}_{chain}".replace("/", "_")
        part_csv = parts_dir / f"{safe}.csv"
        if part_csv.is_file() and not args.overwrite:
            dfs.append(pd.read_csv(part_csv, index_col=0))
            continue
        resolved_pdb = resolve_pdb_path(pdb, args.structures_dir)
        res = predictor.infer_mutants_independent(
            resolved_pdb, data, chain=chain, mut_col="mut_type", skip_reverse=args.skip_reverse
        )
        res.to_csv(part_csv)
        dfs.append(res)

    full = pd.concat(dfs, axis=0)
    full.to_csv(out_csv)
    valid = full[["ddG", "esm3_score"]].dropna()
    rho = spearmanr(valid["ddG"], valid["esm3_score"]).statistic if len(valid) > 1 else float("nan")
    print(f"[PTMUL] DONE -> {out_csv} rows={len(full)} ok={full['esm3_score'].notna().sum()} rho={rho:.4f}", flush=True)
    predictor.log_throughput("[PTMUL] ")


def run_domainome(predictor: ESM3ForgePredictor, args, out_root: Path):
    """Run independent-masked Forge inference on all 522 Domainome domains (resumable per-domain)."""
    out_dir = out_root / "domainome"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "domainome.csv"
    if out_csv.is_file() and not args.overwrite:
        print(f"[DOMAINOME] Already complete: {out_csv}", flush=True)
        return

    path = REPO_ROOT / "data" / "domainome1" / "domainome_mapped_2026.csv"
    df = pd.read_csv(path)
    df["code"] = df["domain_ID"].apply(lambda x: str(x).replace("/", "_"))
    df["ddG_ML"] = df["scaled_fitness"]
    df["wild_type"] = df["mut_type"].str[0]
    df["seq_pos"] = df["mut_type"].str[1:-1]
    df["mutation"] = df["mut_type"].str[-1]
    df = df.dropna(subset=["pdb_file", "seq_pos"])
    df = df[["code", "mut_type", "wild_type", "seq_pos", "mutation", "uniprot_ID", "domain_ID", "pdb_file", "ddG_ML"]].copy()
    df["chain"] = "A"

    parts_dir = out_dir / "per_domain"
    parts_dir.mkdir(exist_ok=True)
    codes = list(df["code"].unique())
    if args.max_proteins > 0:
        codes = codes[: args.max_proteins]

    dfs = []
    pending_jobs = []
    for prot in codes:
        part_csv = parts_dir / f"{prot}.csv"
        if part_csv.is_file() and not args.overwrite:
            dfs.append(pd.read_csv(part_csv, index_col=0))
            continue
        sub = df.loc[df["code"] == prot].copy()
        sub["id"] = sub["domain_ID"].astype(str) + "_" + sub["mut_type"].astype(str)
        sub = sub.set_index("id")
        pdb = resolve_pdb_path(sub["pdb_file"].iloc[0], args.structures_dir)
        pending_jobs.append({
            "pdb_path": pdb,
            "subset_df": sub,
            "chain": "A",
            "mut_col": "mut_type",
            "skip_reverse": True,
            "compute_entropy": True,
            "quiet": True,
            "part_csv": part_csv,
            "prot": prot,
        })

    if pending_jobs:
        done_count = [0]
        total_pending = len(pending_jobs)
        def _save_dom(_idx, job, res):
            res.to_csv(job["part_csv"])
            done_count[0] += 1
            if done_count[0] % 25 == 0 or done_count[0] == total_pending:
                predictor.log_throughput(f"[DOMAINOME {done_count[0]}/{total_pending}] ")

        new_dfs = predictor.infer_many_proteins_independent(pending_jobs, protein_concurrency=8, on_done=_save_dom)
        dfs.extend(new_dfs)

    full = pd.concat(dfs, axis=0)
    full.to_csv(out_csv)
    print(f"[DOMAINOME] DONE -> {out_csv} rows={len(full)} ok={full['esm3_score'].notna().sum()}", flush=True)
    predictor.log_throughput("[DOMAINOME] ")


def run_tsuboyama_split(predictor: ESM3ForgePredictor, args, out_root: Path, split_name: str):
    """Run independent-masked Forge inference on a MegaScale split (val + test, resumable per-domain)."""
    split_file = REPO_ROOT / "data" / f"{split_name}.pkl"
    ds = preprocess_megascale.MegaScaleDatasetPreprocessor(
        data_file=str(REPO_ROOT / "data/tsuboyama/Tsuboyama2023_Dataset2_Dataset3_20230416.csv"),
        af_model_folder=str(REPO_ROOT / "data/tsuboyama/AlphaFold_model_PDBs"),
    )
    ds.create_training_splits(str(split_file), -1)

    reuse_cache = _load_reuse_domain_cache(args.reuse_tsuboyama or [])
    if reuse_cache:
        print(f"[TSU:{split_name}] Loaded {len(reuse_cache)} reusable domains from --reuse_tsuboyama", flush=True)

    for scaffold_ in ("val", "test"):
        out_dir = out_root / f"{split_name}-{scaffold_}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / f"{split_name}-{scaffold_}.csv"
        if out_csv.is_file() and not args.overwrite:
            print(f"[TSU:{split_name}-{scaffold_}] Already complete: {out_csv}", flush=True)
            continue

        data_scaffold = ds.split_dfs[scaffold_].copy()
        data_scaffold = utils.parse_multimutant_column(data_scaffold, "mut_type")
        data_scaffold["id"] = data_scaffold["code"].astype(str) + "_" + data_scaffold["mut_type"].astype(str)
        data_scaffold = data_scaffold.sort_values("id")

        parts_dir = out_dir / "per_domain"
        parts_dir.mkdir(exist_ok=True)
        codes = list(data_scaffold["code_wt"].unique())
        if args.max_proteins > 0:
            codes = codes[: args.max_proteins]

        dfs = []
        reused = 0
        for idx, code in enumerate(codes, 1):
            safe = str(code).replace("/", "_")
            part_csv = parts_dir / f"{safe}.csv"
            if part_csv.is_file() and not args.overwrite:
                dfs.append(pd.read_csv(part_csv, index_col=0))
                continue

            df_true = data_scaffold.loc[data_scaffold["code_wt"] == code].copy()
            # Check reuse cache (exact ID match & 100% non-null)
            if str(code) in reuse_cache and not args.overwrite:
                cached = reuse_cache[str(code)]
                if "id" in cached.columns:
                    cached = cached.set_index("id")
                elif cached.index.name != "id" and "Unnamed: 0" in cached.columns:
                    cached = cached.set_index("Unnamed: 0")
                want_ids = set(df_true["id"])
                if want_ids.issubset(set(cached.index)):
                    matched = cached.loc[df_true["id"]].copy()
                    if matched["esm3_score"].notna().all():
                        if "wt_lora_pred" not in matched.columns:
                            matched["wt_lora_pred"] = matched["pred_additive"].where(
                                matched.get("pred_additive", pd.Series(index=matched.index)).notna(),
                                matched["esm3_score"],
                            )
                        if "combined_pred" not in matched.columns:
                            matched["combined_pred"] = matched["esm3_score"]
                        if "mt_lora_pred" not in matched.columns:
                            matched["mt_lora_pred"] = 2.0 * matched["combined_pred"] - matched["wt_lora_pred"]
                        matched.to_csv(part_csv)
                        dfs.append(matched)
                        reused += 1
                        continue

            df_true["mut_structure"] = df_true["mut_structure"].fillna("-")
            sub_parts = []
            for mut_structure, data in df_true.groupby("mut_structure"):
                backbone_mutation = mut_structure if mut_structure != "-" else None
                data = data.set_index("id")
                data = utils.sum_individual_mutation_scores(data, "ddG_ML", new_score_column="ddG_additive_ML")
                data["dddG_ML"] = data["ddG_ML"] - data["ddG_additive_ML"]
                pdb = resolve_pdb_path(data["pdb_file"].iloc[0], args.structures_dir)
                res = predictor.infer_mutants_independent(
                    pdb, data, chain="A", backbone_mutation=backbone_mutation,
                    mut_col="mut_type", skip_reverse=args.skip_reverse, quiet=True
                )
                sub_parts.append(res)
            dom_df = pd.concat(sub_parts, axis=0)
            dom_df.to_csv(part_csv)
            dfs.append(dom_df)
            if idx % 5 == 0 or idx == len(codes):
                predictor.log_throughput(f"[TSU:{split_name}-{scaffold_} {idx}/{len(codes)} reused={reused}] ")

        full = pd.concat(dfs, axis=0)
        full.to_csv(out_csv)
        # Add newly scored domains to reuse_cache so mega_splits can reuse splits_oct06_capped!
        for c, sub in full.groupby("code_wt"):
            if sub["esm3_score"].notna().all():
                reuse_cache[str(c)] = sub.reset_index()
        print(
            f"[TSU:{split_name}-{scaffold_}] DONE -> {out_csv} rows={len(full)} "
            f"ok={full['esm3_score'].notna().sum()} reused_domains={reused}",
            flush=True,
        )


def run_protein_gym(predictor: ESM3ForgePredictor, args, out_root: Path):
    """Run independent-masked Forge inference across all 217 ProteinGym assays (resumable per-assay)."""
    pgym_dir = Path(args.pgym_dir).resolve()
    mode_tag = "pgym_wt_independent" if args.skip_reverse else "pgym_full_independent"
    out_dir = Path(args.pgym_out).resolve() if args.pgym_out else (out_root / mode_tag)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(pgym_dir / "manifest.csv")
    dms_ids = list(manifest["DMS_id"])
    if args.pgym_dms and args.pgym_dms.lower() != "all":
        want = {x.strip() for x in args.pgym_dms.split(",") if x.strip()}
        dms_ids = [d for d in dms_ids if d in want]

    # Optionally restrict to multi-mutant assays and/or sort from fewest API calls to most
    tasks = []
    for dms_id in dms_ids:
        in_csv = pgym_dir / f"{dms_id}.csv"
        if not in_csv.is_file():
            continue
        out_csv = out_dir / f"{dms_id}.csv"
        if out_csv.is_file() and not args.overwrite:
            continue
        if args.pgym_multi_only or args.sort_by_cost:
            df_m = pd.read_csv(in_csv, usecols=["mut_type_renumbered"])
            muts = df_m["mut_type_renumbered"].astype(str)
            has_multi = bool(muts.str.contains(":").any())
            if args.pgym_multi_only and not has_multi:
                continue
            if args.skip_reverse:
                pos_set = {m[1:-1] for ms in muts for m in ms.split(":")}
                est_calls = len(pos_set)
            else:
                pos_set, mt_set = set(), set()
                for ms in muts:
                    ml = ms.split(":")
                    for m in ml:
                        pos_set.add(m[1:-1])
                    if len(ml) >= 2:
                        t = tuple(ml)
                        for i in range(len(t)):
                            mt_set.add((t[:i] + t[i + 1 :], t[i][1:-1]))
                est_calls = len(pos_set) + len(mt_set)
            tasks.append((est_calls, dms_id, in_csv, out_csv))
        else:
            tasks.append((0, dms_id, in_csv, out_csv))

    if args.sort_by_cost:
        tasks.sort(key=lambda x: x[0])
    if args.max_proteins > 0:
        tasks = tasks[: args.max_proteins]

    print(
        f"[PGYM] Running {len(tasks)} pending assays (of {len(dms_ids)} total) -> {out_dir} "
        f"(skip_reverse={args.skip_reverse}, sort_by_cost={args.sort_by_cost})",
        flush=True,
    )

    summary_rows = []
    summary_csv = out_dir / "summary.csv"
    if summary_csv.is_file():
        try:
            summary_rows = pd.read_csv(summary_csv).to_dict("records")
        except Exception:
            summary_rows = []
    seen_summary = {r["DMS_id"] for r in summary_rows if "DMS_id" in r}

    for idx, (est_calls, dms_id, in_csv, out_csv) in enumerate(tasks, 1):
        t0 = time.perf_counter()
        df = pd.read_csv(in_csv)
        pdb = resolve_pdb_path(df["pdb_file"].iloc[0], args.structures_dir)
        chain = str(df["chain"].iloc[0]) if "chain" in df.columns else "A"
        res = predictor.infer_mutants_independent(
            pdb, df, chain=chain, mut_col="mut_type_renumbered",
            skip_reverse=args.skip_reverse, compute_entropy=False, quiet=False
        )
        res.to_csv(out_csv, index=False)
        pred_col = "wt_lora_pred" if args.skip_reverse else "combined_pred"
        valid = res[["DMS_score", pred_col]].dropna()
        rho = float(spearmanr(valid["DMS_score"], valid[pred_col]).statistic) if len(valid) > 1 else float("nan")
        dt = time.perf_counter() - t0
        print(
            f"[PGYM {idx}/{len(tasks)}] DONE {dms_id}: n={len(res)} ok={len(valid)} "
            f"rho={rho:.4f} ({dt:.1f}s)",
            flush=True,
        )
        if dms_id not in seen_summary:
            summary_rows.append({"DMS_id": dms_id, "n": len(res), "n_ok": len(valid), "spearman": rho, "sec": round(dt, 2)})
            seen_summary.add(dms_id)
            pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
        if idx % 10 == 0 or idx == len(tasks):
            predictor.log_throughput(f"[PGYM {idx}/{len(tasks)}] ")


def run_mgnify(predictor: ESM3ForgePredictor, args, out_root: Path):
    """Run independent-masked Forge inference on MGnify / Rocklin de novo in resumable shards."""
    mgn_dir = Path(args.mgnify_dir).resolve()
    table_path = Path(args.mgnify_table).resolve() if args.mgnify_table else (mgn_dir / "mgnify_mapped.csv.gz")
    out_dir = Path(args.mgnify_out).resolve() if args.mgnify_out else (out_root / "mgnify")
    shards_dir = out_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    print(f"[MGNIFY] Loading {table_path} ...", flush=True)
    df = pd.read_csv(table_path, low_memory=False)
    if args.mgnify_sources and args.mgnify_sources.lower() != "all":
        want = {s.strip() for s in args.mgnify_sources.split(",") if s.strip()}
        df = df[df["source"].astype(str).isin(want)].copy()
    if args.mgnify_splits and args.mgnify_splits.lower() != "all":
        want = {s.strip() for s in args.mgnify_splits.split(",") if s.strip()}
        col = "mgnify_split" if "mgnify_split" in df.columns else "split"
        df = df[df[col].astype(str).isin(want)].copy()

    df = df[df["pdb_file"].notna() & (df["pdb_file"].astype(str).str.strip() != "")].copy()
    if "chain" not in df.columns:
        df["chain"] = "A"
    if "backbone_mutation" not in df.columns:
        df["backbone_mutation"] = ""
    df["backbone_mutation"] = df["backbone_mutation"].fillna("").astype(str)
    df["pdb_file"] = df["pdb_file"].astype(str).apply(lambda p: resolve_pdb_path(p, args.structures_dir, base_dir=mgn_dir))
    df = df[df["pdb_file"].apply(os.path.isfile)].copy()

    grouped = list(df.groupby(["code", "pdb_file", "chain", "backbone_mutation"], sort=False, dropna=False))
    if args.max_proteins > 0:
        grouped = grouped[: args.max_proteins]

    shard_size = max(1, int(args.mgnify_shard_size))
    print(f"[MGNIFY] Scoring {len(df):,} rows across {len(grouped):,} scaffolds in shards of {shard_size} -> {out_dir}", flush=True)

    shard_files = []
    for s_start in range(0, len(grouped), shard_size):
        s_idx = s_start // shard_size
        shard_csv = shards_dir / f"shard_{s_idx:05d}.csv.gz"
        shard_files.append(shard_csv)
        if shard_csv.is_file() and not args.overwrite:
            continue
        batch_groups = grouped[s_start : s_start + shard_size]
        jobs = [
            {
                "pdb_path": pdb,
                "subset_df": sub,
                "chain": str(chain),
                "backbone_mutation": (bb if bb else None),
                "mut_col": "mut_type",
                "skip_reverse": args.skip_reverse,
                "compute_entropy": False,
                "quiet": True,
            }
            for (code, pdb, chain, bb), sub in batch_groups
        ]
        parts = predictor.infer_many_proteins_independent(jobs, protein_concurrency=24)
        shard_df = pd.concat(parts, axis=0)
        shard_df.to_csv(shard_csv, index=False, compression="gzip")
        predictor.log_throughput(f"[MGNIFY shard {s_idx} ({min(s_start + shard_size, len(grouped))}/{len(grouped)})] ")

    # Merge shards into predictions.csv.gz once all are present
    all_parts = [pd.read_csv(sf, low_memory=False) for sf in shard_files if sf.is_file()]
    if all_parts:
        full = pd.concat(all_parts, axis=0)
        full.to_csv(out_dir / "predictions.csv.gz", index=False, compression="gzip")
        print(f"[MGNIFY] DONE -> {out_dir / 'predictions.csv.gz'} rows={len(full):,}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="High-throughput ESM3 Forge zero-shot runner (independent masking).")
    parser.add_argument("--model_name", type=str, default="esm3-small-2024-08")
    parser.add_argument("--forge_token", type=str, default=None, help="Forge API token (or set ESM_API_KEY env var)")
    parser.add_argument("--token_file", type=str, default=None, help="Path to file containing Forge API token")
    parser.add_argument("--out_dir", type=str, required=True, help="Output root directory for this model")
    parser.add_argument("--structures_dir", type=str, default=None, help="Fallback directory for PDB structures")
    parser.add_argument("--max_concurrency", type=int, default=48, help="Concurrent Forge /logits requests")
    parser.add_argument("--tokens_per_min", type=int, default=0, help="Explicit token/min cap (0 = auto-tune from 429)")
    parser.add_argument("--skip_reverse", action="store_true", help="WT-pass only (additive masked marginal on multis)")
    parser.add_argument("--overwrite", action="store_true", help="Recompute existing per-protein/assay outputs")
    parser.add_argument("--max_proteins", type=int, default=0, help="Smoke-test cap on domains/assays (0 = all)")

    # Dataset selectors
    parser.add_argument("--run_ptmul", action="store_true")
    parser.add_argument("--run_domainome", action="store_true")
    parser.add_argument("--splits", type=str, default="", help="Comma-separated Tsuboyama splits (e.g. splits_oct06_capped,mega_splits)")
    parser.add_argument("--reuse_tsuboyama", nargs="*", default=[], help="Existing Tsuboyama CSVs whose scored domains can be reused")
    parser.add_argument("--protein_gym", action="store_true")
    parser.add_argument("--pgym_dir", type=str, default=str(REPO_ROOT / "pgym_inputs"))
    parser.add_argument("--pgym_out", type=str, default=None)
    parser.add_argument("--pgym_dms", type=str, default="all")
    parser.add_argument("--pgym_multi_only", action="store_true")
    parser.add_argument("--sort_by_cost", action="store_true", help="Order ProteinGym assays from fewest API calls to most")
    parser.add_argument("--mgnify", action="store_true")
    parser.add_argument("--mgnify_dir", type=str, default=str(REPO_ROOT / "data" / "mgnify"))
    parser.add_argument("--mgnify_table", type=str, default=None)
    parser.add_argument("--mgnify_out", type=str, default=None)
    parser.add_argument("--mgnify_sources", type=str, default="all")
    parser.add_argument("--mgnify_splits", type=str, default="all")
    parser.add_argument("--mgnify_shard_size", type=int, default=200)

    args = parser.parse_args()
    token = resolve_forge_token(args.forge_token, args.token_file)
    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    predictor = ESM3ForgePredictor(
        token=token,
        model_name=args.model_name,
        max_concurrency=args.max_concurrency,
        tokens_per_min=args.tokens_per_min if args.tokens_per_min > 0 else None,
    )

    if args.run_ptmul:
        run_ptmul(predictor, args, out_root)
    if args.run_domainome:
        run_domainome(predictor, args, out_root)
    if args.splits:
        for sp in [s.strip() for s in args.splits.split(",") if s.strip()]:
            run_tsuboyama_split(predictor, args, out_root, sp)
    if args.protein_gym:
        run_protein_gym(predictor, args, out_root)
    if args.mgnify:
        run_mgnify(predictor, args, out_root)

    predictor.log_throughput("[FINAL] ")


if __name__ == "__main__":
    main()
