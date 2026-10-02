"""
auto_batch.py
=============

Self-calibrating, on-the-fly GPU batch sizing for chunked inference workloads.

Design goals
------------
* Start from batch size 1 for each work list (e.g. the deduped masked states
  of one DMS, or the dense mutant rows of one DMS) and double repeatedly
  until the maximum batch size that fits in VRAM is established.
* Do NOT rely on a CUDA OOM as the discovery mechanism. A doubled size is
  attempted only if a linear memory model -- fit on MEASURED peaks of the
  chunks that already ran -- predicts the attempt stays below a conservative
  headroom limit. On devices that cannot overflow (discrete GPUs), OOM is
  thus structurally avoided instead of discovered.
* When the gate REJECTS a doubling, the sizer does not park at the last
  confirmed size (leaving VRAM unused for every remaining chunk). It
  bisects the same linear model for the LARGEST size in (b, 2b] that still
  fits -- which need not be a power of two -- and runs at that size, so the
  steady-state chunks sit as close to the headroom limit as the model
  justifies. The first chunk at the probed size IS the empirical probe;
  the OOM backstop remains the final guard.
* Agnostic to inference method, GPU model, sequence length, and VRAM
  reporting quirks. The memory model is refit from driver-level measured
  peaks (torch.cuda.mem_get_info, sampled in-flight by a background thread
  plus a post-sync read), not from lookup tables or predefined trends.
* Self-correcting. If an OOM is hit anyway (a missed transient spike, or a
  chunk heavier than the model expects), it is caught, the failed range is
  retried at half size, that size is blacklisted for the rest of the work
  list, and a learned safety factor makes every subsequent prediction more
  conservative. Doublings that confirm the model (measured peak within 15%
  of prediction) relax the factor back toward 1.0.

Everything lives in this one file. Other modules only call:

    sizer = AutoBatchSizer('cuda:0', headroom=0.90)
    sizer.run(n_items, chunk_fn, label='...')

where chunk_fn(start, end) performs the real GPU work for items [start:end)
in one batch of size (end - start). The chunk size IS the batch size.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch


# --------------------------------------------------------------------------- #
# Driver-level VRAM reads
# --------------------------------------------------------------------------- #

def _gpu_mem_info(device: torch.device, _tries: int = 40) -> Tuple[int, int]:
    """Return (used_bytes, total_bytes) from the driver, fast in-process.

    Retries a few times: right after lazy CUDA context initialization the
    driver can transiently report inconsistent (free, total) values.
    """
    import time as _time
    for _ in range(_tries):
        free, total = torch.cuda.mem_get_info(device)
        if total > 0 and 0 <= free <= total:
            return total - free, total
        _time.sleep(0.05)
    return total - free, total


# --------------------------------------------------------------------------- #
# Peak sampler
# --------------------------------------------------------------------------- #

class _PeakSampler(threading.Thread):
    """Background thread sampling driver-level used VRAM at high frequency.

    In-flight sampling catches transient spikes inside a forward pass that a
    single post-hoc read would miss, without any dependence on nvidia-smi
    subprocess cadence or its under-reporting while kernels run.
    """

    def __init__(self, device: torch.device, interval: float = 0.02):
        super().__init__(daemon=True, name='auto-batch-peak-sampler')
        self._device = device
        self._interval = max(0.005, interval)
        # NB: must NOT be named _stop (shadows threading.Thread._stop used by join())
        self._halt = threading.Event()
        self._lock = threading.Lock()
        self._peak = 0

    def run(self) -> None:
        while not self._halt.is_set():
            try:
                used, _ = _gpu_mem_info(self._device)
                with self._lock:
                    if used > self._peak:
                        self._peak = used
            except Exception:
                pass  # never let the sampler take the run down
            self._halt.wait(self._interval)

    def arm(self) -> None:
        """Reset the tracked peak (call just before a chunk)."""
        with self._lock:
            self._peak = 0

    def read(self) -> int:
        """Consume the peak since arm(); include one post-sync read."""
        with self._lock:
            peak, self._peak = self._peak, 0
        try:
            torch.cuda.synchronize(self._device)
            used, _ = _gpu_mem_info(self._device)
            peak = max(peak, used)
        except Exception:
            pass
        return peak

    def shutdown(self) -> None:
        if self.is_alive():
            self._halt.set()
            self.join(timeout=max(0.25, 10 * self._interval))
            self._halt.clear()


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

@dataclass
class AutoBatchReport:
    label: str
    n_items: int
    ladder: List[int]          # chunk sizes actually used, in order
    max_batch: int
    peak_bytes: int
    limit_bytes: int
    ooms: int

    def summary(self) -> str:
        gi = 1024 ** 3
        return (f"[AUTO-BATCH {self.label}] items={self.n_items} "
                f"ladder={'+'.join(map(str, self.ladder))} "
                f"max_batch={self.max_batch} peak={self.peak_bytes / gi:.2f}GiB "
                f"limit={self.limit_bytes / gi:.2f}GiB ooms={self.ooms}")


# --------------------------------------------------------------------------- #
# Sizer
# --------------------------------------------------------------------------- #

class AutoBatchSizer:
    """Adaptive batch-size controller for chunked GPU work.

    Parameters
    ----------
    device:
        CUDA device to size against.
    headroom:
        Fraction of VRAM a PREDICTED chunk peak may occupy (default 0.90,
        deliberately below external watchdog kill points). The gate is
        min(total * headroom, baseline + free * headroom), so a GPU shared
        with other processes is handled correctly.
    max_batch:
        Optional hard cap on the batch size.
    sampler_interval:
        Seconds between in-flight VRAM samples (default 0.02).
    log:
        Callable for progress lines (default print).
    """

    def __init__(self,
                 device,
                 headroom: float = 0.90,
                 max_batch: Optional[int] = None,
                 sampler_interval: float = 0.02,
                 log=print):
        self.device = torch.device(device)
        if not (0.0 < headroom <= 1.0):
            raise ValueError(f"headroom must be in (0, 1]; got {headroom}")
        self.headroom = float(headroom)
        self.max_batch = int(max_batch) if max_batch else None
        self.log = log
        self._sampler_interval = float(sampler_interval)
        # Learned across work lists within this process:
        self.safety = 1.0            # >= 1.0 multiplier applied to predictions
        self.total_ooms = 0
        self.reports: List[AutoBatchReport] = []
        self._slope = 0.0            # most recent measured bytes-per-item

    # Driver-level overhead for very large single allocations, measured on
    # the RTX 5090 (32607 MB): an 18.75 GiB tensor costs exactly 18.75 GiB,
    # but 23.4-25.8 GiB tensors cost +1.25..1.5 GiB over the requested size
    # (driver reporting, not torch-internal). Bisection probes land between
    # measured points near the gate limit, where this overhead is real, so
    # probe searches reserve this margin to keep the probe's TRUE peak
    # within ~0.5 GiB of the limit. Steady-state/doubling sizes are gated
    # on the full limit: their worst-case true peak (limit + 1.5 GiB) still
    # sits ~0.85 GiB below the external 95% watchdog kill on this card.
    _PROBE_RESERVE_BYTES = 1 * 1024**3

    # ------------------------------------------------------------------ #
    # Memory model
    # ------------------------------------------------------------------ #
    def _predict(self, points: List[Tuple[int, int]],
                 b_target: int, baseline: int) -> int:
        """Predict the total-VRAM peak for a chunk of b_target items.

        With >= 2 measured points: linear extrapolation through the last two
        (batched-forward memory is linear in batch size above the fixed cost).
        With a single point (the first doubling, 1 -> 2): full-doubling of the
        measured incremental cost.
        """
        if len(points) >= 2:
            (b1, p1), (b2, p2) = points[-2], points[-1]
            if b2 > b1:
                self._slope = (p2 - p1) / (b2 - b1)
            return int(p2 + (b_target - b2) * self._slope)
        (b1, p1) = points[0]
        return int(p1 + (b_target - b1) * (p1 - baseline))

    def _max_fitting_size(self, points: List[Tuple[int, int]], b: int,
                          nxt: int, limit: float, baseline: int) -> Optional[int]:
        """Largest integer m in (b, nxt] whose predicted peak satisfies the
        SAME gate the doubling decision uses.

        The prediction is (piecewise) linear in m and non-decreasing with a
        positive measured slope, so the predicate is monotone: bisection
        finds the largest fitting size in O(log n) O(1) _predict calls.
        Returns None if not even b+1 is justified by the model.

        The `limit` argument is the gate the caller wants enforced; the
        run() caller passes `limit - _PROBE_RESERVE_BYTES` here, because a
        bisection probe is an unmeasured size between model points and the
        driver adds 1.25-1.5 GiB to >19 GiB single allocations (see
        _PROBE_RESERVE_BYTES).

        Note the range top is `nxt` (already clamped to the remaining items
        and to max_batch), so the result can never exceed either cap, and
        the caller only invokes this when the OOM blacklist does not cover
        the range — the result can never re-attempt a failed size.
        """
        if nxt <= b + 1:
            return None
        if self._predict(points, b + 1, baseline) * self.safety > limit:
            return None
        lo, hi = b + 1, nxt
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._predict(points, mid, baseline) * self.safety <= limit:
                lo = mid
            else:
                hi = mid - 1
        return lo

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #
    def run(self, n_items: int,
            chunk_fn: Callable[[int, int], None],
            label: str = "") -> AutoBatchReport:
        """Process n_items with adaptively-doubled batch sizes.

        chunk_fn(start, end) must do the real work for items [start:end) in
        one GPU batch of size (end - start). Every item is processed exactly
        once. The batch ladder starts at 1 and doubles; each doubling is
        gated on the measured memory model.
        """
        if n_items <= 0:
            return AutoBatchReport(label, 0, [], 0, 0, 0, 0)

        # Unmap stale caching-allocator segments left behind by previous work
        # (the prior DMS, the prior adapter pass) BEFORE measuring the
        # baseline. If the baseline absorbs stale cache, the new work reuses
        # that cache for "free": measured peaks stay flat, the linear fit
        # sees a ~zero slope, and the gate doubles past the true cost until
        # the stale cache is exhausted and the chunk overflows. empty_cache()
        # never frees live tensors — only unused cached segments — so it is
        # always safe to call here.
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        baseline, total = _gpu_mem_info(self.device)
        free = total - baseline
        # Gate on the total-usage scale, but bounded by what is actually
        # free right now (other processes / already-cached tensors included).
        limit = min(total * self.headroom, baseline + free * self.headroom)
        self.log(
            f"    [AUTO-BATCH {label}] baseline={baseline / 1024**3:.2f}GiB "
            f"total={total / 1024**3:.2f}GiB limit={limit / 1024**3:.2f}GiB"
        )

        # A fresh sampler per run: threading.Thread objects are one-shot.
        sampler = _PeakSampler(self.device, self._sampler_interval)
        sampler.start()
        ladder: List[int] = []
        points: List[Tuple[int, int]] = []   # (chunk_size, measured peak)
        peak_max = 0
        ooms = 0
        b = 1
        i = 0
        last_size: Optional[int] = None      # size of the previous executed chunk
        last_exec = 0                        # last executed chunk size (for probe step-back)
        failed_size: Optional[int] = None    # OOM'd this run: never re-attempt >= it
        pending_pred: Optional[int] = None   # predicted peak for the in-flight chunk
        floor_logged = False                 # over-limit-at-floor notice emitted once
        last_prog = -1                       # last progress bucket logged (i // 500)
        t0 = time.monotonic()                # run start, for progress elapsed
        try:
            while i < n_items:
                c = min(b, n_items - i)
                if c != last_size:
                    # Size transition: unmap the caching allocator's reserved
                    # (cached) segments of the OLD size before the new chunk
                    # allocates. The driver-level "used" the sampler measures
                    # includes those cached segments, and they accumulate
                    # across the ladder (chunks 1,2,4,...,c leave 1+2+...+c
                    # units of cache). Without the unmap, the peak at a
                    # non-power-of-two size m is B + m*x + (stale cache) —
                    # ABOVE the linear model's B + m*x — so a bisection probe
                    # can overflow the card even though the gate says it fits
                    # (the model is exact only at doubling points). In steady
                    # state (same size as the previous chunk) the allocator
                    # reuses the cached segment of exactly that size, so this
                    # costs one cudaFree/cudaMalloc cycle per size transition
                    # only (~tens of ms).
                    torch.cuda.synchronize(self.device)
                    torch.cuda.empty_cache()
                    last_size = c
                sampler.arm()
                try:
                    chunk_fn(i, i + c)
                except torch.cuda.OutOfMemoryError:
                    ooms += 1
                    self.total_ooms += 1
                    pending_pred = None
                    try:
                        torch.cuda.synchronize(self.device)
                    except Exception:
                        pass
                    torch.cuda.empty_cache()
                    if c <= 1:
                        # A single item does not fit: the workload itself
                        # exceeds the GPU. Nothing to retry; propagate.
                        raise
                    # The model under-predicted (missed transient or a
                    # heavier-than-expected chunk). Blacklist the failed size
                    # for this work list, be conservative from now on, and
                    # retry this same range at half the failed size.
                    failed_size = c
                    self.safety = max(self.safety, 1.5)
                    b = max(1, c // 2)
                    self.log(f"    [AUTO-BATCH {label}] OOM at batch {c}; "
                             f"safety={self.safety:.2f}, retrying range at <= {b}")
                    continue
                peak = sampler.read()
                peak_max = max(peak_max, peak)
                points.append((c, peak))
                ladder.append(c)
                i += c
                # Safety relaxation: did the chunk that just ran land near
                # what the gate predicted for it?
                if pending_pred is not None:
                    if peak <= pending_pred * 1.15:
                        self.safety = max(1.0, self.safety * 0.8)
                    pending_pred = None
                # Probe confirmation: the gate is a model PREDICTION; this
                # measured peak is the empirical check. If a chunk (usually
                # the first at a freshly probed size) overshot the limit by
                # more than 2%, the model under-predicted — fit jitter,
                # non-uniform item costs in the real pipeline, or driver
                # overhead on >19 GiB allocations. The chunk SUCCEEDED, so
                # its items are done; just step the batch back halfway to
                # the previous executed size and stay conservative. The new
                # (c, peak) point already entered the fit above, which
                # self-corrects the slope. The OOM backstop remains the
                # final guard; the external 95% watchdog sits ~2.3 GiB
                # above the 90% limit, well clear of a +2% overshoot.
                if peak > limit * 1.02:
                    nb = max(1, (last_exec + c) // 2)
                    if nb < c:
                        self.log(f"    [AUTO-BATCH {label}] batch {c} measured "
                                 f"{peak / 1024**3:.2f}GiB > limit "
                                 f"{limit / 1024**3:.2f}GiB (+2%); stepping back "
                                 f"to {nb}")
                    elif not floor_logged:
                        # Step-back target == current size: the batch is at its
                        # floor (typically 1) and still measures above the soft
                        # limit, yet below the card. Running over the soft limit
                        # is legitimate here (the measurement itself just
                        # succeeded); log it ONCE, not once per chunk, or a
                        # floor-bound work list spams the log thousands of
                        # identical lines and looks like a hang.
                        self.log(f"    [AUTO-BATCH {label}] batch {c} measured "
                                 f"{peak / 1024**3:.2f}GiB > limit "
                                 f"{limit / 1024**3:.2f}GiB; at floor, holding "
                                 f"b={c} (fits under the card; OOM backstop "
                                 f"active)")
                        floor_logged = True
                    b = nb
                    self.safety = max(self.safety, 1.2)
                last_exec = c
                # Periodic progress so a long floor-bound or ladder-bound work
                # list is visibly alive (one line per ~500 items).
                if (i // 500) != last_prog:
                    last_prog = i // 500
                    self.log(f"    [AUTO-BATCH {label}] progress {i}/{n_items} "
                             f"b={last_exec} elapsed={time.monotonic() - t0:.0f}s")
                if i >= n_items:
                    break
                # Doubling decision. When the gate rejects the full doubling
                # (VRAM limit, max_batch cap, or OOM blacklist) we do NOT
                # merely park at b: we bisection-search the linear model for
                # the largest size in (b, nxt] the model says fits, and climb
                # to it. (nxt is already clamped to the remaining items and
                # to max_batch, so this also sizes tail remainders — e.g. a
                # 200-item tail at b=128 runs 150+50 instead of 128+72 when
                # the model says 150 fits.) The first chunk at the probed
                # size IS the empirical probe; the OOM backstop remains the
                # final guard, and the blacklist above guarantees (b, nxt]
                # never overlaps a failed size. If the model does not even
                # justify b+1 we park at b — this must NOT break the work
                # loop.
                if (self.max_batch is None or b < self.max_batch) and \
                        not (failed_size is not None and 2 * b >= failed_size):
                    nxt = min(2 * b, n_items - i)
                    if self.max_batch is not None:
                        nxt = min(nxt, self.max_batch)
                    pred = self._predict(points, nxt, baseline)
                    if pred * self.safety <= limit:
                        pending_pred = pred
                        b = nxt
                    else:
                        m = self._max_fitting_size(
                            points, b, nxt,
                            limit - self._PROBE_RESERVE_BYTES, baseline)
                        if m is not None:
                            pending_pred = self._predict(points, m, baseline)
                            b = m
        finally:
            sampler.shutdown()

        report = AutoBatchReport(label, n_items, ladder,
                                 max(ladder) if ladder else 0,
                                 peak_max, int(limit), ooms)
        self.reports.append(report)
        self.log(f"    {report.summary()}")
        return report
