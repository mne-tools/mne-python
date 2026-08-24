# W5 — HDF5 chunk-cache tuning & zarr-v3 local-read verdict

Date: 2026-08-24 · Machine: Apple M4 Max, 36 GB, macOS · h5py 3.16.0, zarr 3.1.5,
numpy 2.4.6 · Data: `benchmarks/io_dl/data_big/` (128 ch × 1800 s @ 512 Hz,
float32; `bench_big.h5` datasets `t10s` chunks=(1,5120), `win` chunks=(128,512),
`full` contiguous; zarr v3 twins with identical chunk shapes).
Script: `bench_cache.py` (this directory); raw numbers in
`cache_results_20260824-205105.json` / `-205253.json`.

## Methodology (drift-proof by construction)

- **Single process, interleaved arms**: all 16 arms alternate every round, order
  re-shuffled per round (seeded), 7 rounds × 300 random 2 s windows (seed 99),
  persistent handles opened once (training-loop pattern). GC disabled during
  timed passes; `perf_counter_ns`; float64 checksum verified identical across
  every arm/pass (pure-read sanity).
- **Warmup pass** per arm before round 1 (fills page cache + arm-local caches);
  no `sudo purge` available → regime is *warm OS page cache*, which is also the
  realistic steady state of a training loop.
- **Control arm** `h5_win_default` interleaved with everything; its per-round
  drift was 88–113 µs/win (max/min 1.22–1.29) across rounds — i.e. machine noise
  of ±10–15 % was present and is exactly what interleaving cancels. Any claimed
  effect below must exceed that band to be real.
- Two independent process runs (A, B); conclusions require agreement in both.
- Cache-fit sizes computed exactly from the seed-99 window set:
  `win`: 527 distinct chunks touched → fit = 138 MB;
  `t10s`: 20,736 chunks (162 time-slabs × 128 ch) → fit = 405 MB.

## Decision table

µs/window, median over 14 interleaved passes (7 rounds × 2 runs); spread =
min..max pooled. "vs CTRL" compares each arm's median against its same-run
control to cancel drift.

| config | µs/win median [min..max] | vs CTRL | verdict |
|---|---|---|---|
| h5_win default cache 1 MB/521 (**CONTROL**) | 100 [88..113] | 1.00 | baseline |
| h5_win fit-to-working-set 138 MB/1061 slots | 91 [86..108] | −5 % | no effect (inside noise band) |
| h5_win oversized 512 MB/65537 | 101 [88..108] | +1 % | no effect |
| h5_win nslots=10103 (1 MB) | 97 [89..104] | −4 % | no effect |
| h5_win nslots=65537 (1 MB) | 100 [88..107] | −1 % | no effect |
| h5_t10s default 1 MB/521 | 479 [443..558] | 1.00 | baseline (slab layout) |
| h5_t10s fit-to-working-set 405 MB/41479 | 466 [443..543] | −2 % | no effect |
| h5_t10s oversized 512 MB/65537 | 473 [455..496] | −1 % | no effect |
| h5_t10s nslots=10103 | 468 [446..605] | −2 % | no effect |
| h5_t10s nslots=65537 | 477 [446..535] | 0 % | no effect |
| h5_t10s driver=sec2 explicit | 476 [442..498] | ≈0 | confirms default driver |
| **h5_t10s driver='core' (RAM)** | **344 [330..376]** | **−26 %** | real, reproducible (−26/−28 % both runs) |
| zarr_win default (`async.concurrency=10`) | 1049 [1025..1279] | — | **10.1–10.6× slower than h5_win** |
| zarr_win concurrency=1 | 1853 [1786..1986] | — | +77 % vs zarr default — do NOT set |
| zarr_win concurrency=32 | 1047 [1020..1269] | — | ≈ zarr default (no gain) |
| zarr_t10s default | 27,994 [27,326..28,806] | — | ~60× slower than h5_t10s |

### Does chunk-cache tuning matter (>20 %)? **NO.**

Largest deviation of any cache arm from control: **−5 % (win_fit), −2 %
(t10s_fit)** — an order of magnitude below the 20 % threshold and not
systematic across the two runs (sign flips between runs). Mechanism:

1. With a warm page cache every chunk read is already served at memcpy speed
   by macOS; HDF5's rdcc only short-circuits *re-reads*, which were cheap
   anyway.
2. Random windows give huge reuse distances: each `t10s` chunk is re-visited
   only ~3× over a whole 300-window pass, separated by thousands of other
   chunk reads — nothing stays "hot" even in a perfectly sized cache.
3. No intra-read re-touch exists (a window's ≤256 chunks are all distinct), so
   oversizing has nothing to exploit.

rdcc tuning should only matter when the working set exceeds RAM (cold reads)
or windows are sampled with tight reuse distance (heavy overlap). Neither
applies to EEG-shaped files (~0.5 GB) on developer/workstation RAM.

## File-driver variants (measured once each, interleaved like other arms)

`driver="core", backing_store=False` reads the whole file into process RAM at
open: **+~0.9 GiB RSS** for this 944 MB file (plus your working arrays), and
buys **−26 % per window on the t10s layout** (344 vs 476 µs) by eliminating
per-chunk `read()` syscalls (≤256 syscalls/window → memcpy). Benefit shrinks
proportionally with chunks-per-window (the `win` layout touches only 2), so
its practical value is for slab/time-major layouts or network filesystems.
Not recommended as default advice; document as an option with the memory cost.

## Zarr v3 local-read verdict

- Identical chunk shape `(128,512)` to `h5_win`, zstd level-0 codecs (chunks
  stored ~9 % smaller): **zarr 1049 µs/win vs h5py 100 µs/win ⇒ 10.1–10.6×
  penalty** (earlier ad-hoc "~20×" was drift-inflated; clean interleaved ratio
  is ~10×). For the slab layout the gap explodes to ~60× (28 ms/win).
- Knobs: `async.concurrency` 32 = no change; `concurrency=1` makes it **worse**
  (+77 %) — the default pool of 10 is already optimal for 2-chunk reads. The
  penalty is architectural: per-chunk Python/async dispatch + buffer creation
  in zarr v3's sync path, not a tunable configuration issue.

## RECOMMENDATIONS

1. **Tell users nothing about rdcc kwargs** — there is no `h5py.open(...)`
   chunk-cache setting worth recommending for EEG-shaped local training data;
   defaults (1 MB/521) measure identically to exact-fit and 512 MB caches.
   Spend the effort on chunk geometry instead: `chunks=(n_ch, window_samples)`
   reproduces the ~50–100 µs/window result and needs zero tuning.
2. Keep zarr flagged **remote/parallel-only** for training loops: ~10×
   per-window penalty locally with identical chunking, unfixable via its
   concurrency settings. Revisit only if zarr v3 gains a C-speed batched read
   path or if reads move to object storage where network latency dominates
   anyway.
3. If someone insists on squeezing the slab-layout case without rewriting
   chunks, `h5py.File(..., driver="core", backing_store=False)` buys ~26 % at
   +1× file-size RAM; cheaper than cache tuning, still worse than rechunking.
4. Do not set `zarr.config.set({"async.concurrency": 1})` anywhere (76 %
   slowdown); leave defaults alone.

## Draft text for RESULTS.md §12 (ready to paste)

```markdown
## 12. Session 5: HDF5 chunk-cache tuning is a dead end; zarr-v3 local verdict finalized

Question: does `rdcc_nbytes`/`rdcc_nslots` change random-window read speed for
EEG-shaped data? Method: single-process interleaved benchmark (16 arms × 7
rounds × 300 windows, seeded shuffle per round, persistent handles, identical
checksums, control arm tracked per-round drift of ±10–15 % which interleave
cancels; two independent runs).

Answer: **no**. Exact-fit caches (computed from the true touched-chunk set:
138 MB for window-chunks, 405 MB for per-channel slabs), a 512 MB oversized
cache, and hash-slot variations (10103/65537 primes) all land within ±5 % of
the 1 MB/521 default — far under the ±15 % machine-noise band, with signs
flipping between runs. Reason: with a warm page cache, chunk re-reads are
already memcpy-speed, and random sampling gives reuse distances so large that
nothing stays hot in any cache size. Chunk *geometry* remains the only lever
that matters (`chunks=(n_ch, win)` ≈ 100 µs/win; slabs ≈ 480 µs/win).

Two side results. (1) `driver="core"` (whole file pinned in RAM, +0.9 GiB
here) cuts slab-layout windows by 26 % (476→344 µs) via syscall elimination —
an option, not a recommendation. (2) The zarr v3 local-read penalty is real
but was overstated by earlier noisy probes: with byte-identical chunk shapes,
zarr reads windows at ~1050 µs vs h5py's ~100 µs ⇒ **≈10× slower** (not ~20×);
for per-channel-slab chunks it is ~60×. Its `async.concurrency` knob neither
helps at 32 nor tolerates 1 (+77 %), so the gap is per-chunk Python dispatch
overhead in v3's sync path. Guidance stands: HDF5/memmap for local training,
zarr reserved for remote/parallel storage.
```
