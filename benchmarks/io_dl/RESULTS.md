# MNE-Python IO for deep learning: benchmark + first speedups

Date: 2026-08-24 · Machine: Apple Silicon (arm64), macOS, py 3.12.12, numpy 2.4.6,
MNE 1.12.1 (this working tree). All numbers below are medians; see
"Methodology notes" for why interleaved A/B is authoritative here.

## 1. Which three formats?

| Rank | Format | Why it dominates DL workloads |
|---|---|---|
| 1 | **EDF/EDF+** | The big DL corpora are EDF: TUH EEG Corpus (~25k studies, braindecode's flagship dataset), PhysioNet (CHB-MIT, Sleep-EDF). BIDS-recommended. |
| 2 | **BrainVision (.vhdr/.eeg/.vmrk)** | The other BIDS-recommended format; standard on OpenNeuro and MOABB; mne-bids converts to it by default. |
| 3 | **FIF** | MNE-native; what DL pipelines persist preprocessed data/caches to (braindecode preprocessing saves `.fif`). |

(Honorable mention: EEGLAB `.set`, common but usually converted to one of the above.)

## 2. Benchmark design

Files in this directory:

- `generate_data.py` – writes the *same* synthetic signal (64 ch × 300 s @ 256 Hz,
  ~±100 µV band-limited noise) to `bench.edf`, `bench.vhdr(+.eeg,.vmrk)`,
  `bench_raw.fif`, plus a raw float32 `bench.bin` baseline. EDF is int16 by design
  (half the on-disk bytes of float32).
- `bench_io.py` – scenario suite (see below), GC disabled during timing.
- `profile_io.py` – cProfile attribution + deepcopy tracer (rebinds every
  `from copy import deepcopy` reference in loaded `mne.*` modules).
- `ab_test.py` – interleaved A/B between the installed (unpatched) release copy
  and this working tree; cancels machine drift.
- `check_equivalence.py` – md5-compares outputs of every access pattern between
  unpatched and patched code (must be bit-identical).

Scenarios (the DL-relevant access patterns):

| Scenario | Meaning |
|---|---|
| `open_meta` | `read_raw_X(preload=False)` — header parse only |
| `full_load` | `read_raw_X(preload=True)` — end-to-end load |
| `seq_1s` | 300 sequential 1 s `get_data()` calls (streaming/eval) |
| `rand_windows` | 300 shuffled 2 s windows via `get_data()`, `preload=False` — **the training-loop pattern** (dataset opens raw once, sampler draws windows) |
| `preloaded_windows` | same windows on a preloaded raw — pure per-call Python overhead |
| floors | numpy `memmap` / `fromfile` equivalents reading identical bytes |

### Methodology notes (things that silently corrupt results)

1. **Import resolution**: running scripts from inside `benchmarks/io_dl/` or with
   `python -c` resolves `mne` differently (cwd lands on `sys.path` for `-c`;
   script dir replaces cwd for files). Both bench scripts now pin the working
   tree explicitly. Verify with the printed `mne loaded from:` line.
2. **Sustained-load drift**: whole-suite runs back-to-back showed ±20–100 %
   swings under background load (even the numpy floor drifted 4→11 ms).
   Conclusions here rely on interleaved A/B (`ab_test.py`), not cross-run diffs.
3. cProfile inflates tiny-function costs ~3×; use it for attribution only.

## 3. Results

### 3.1 Baseline (unpatched tree) — where does time go?

Steady-state cost per `get_data()` call, preloaded raw, all three formats:
**~100 µs**, identical across formats ⇒ pure Python overhead in the shared path,
zero relation to disk. The numpy floor for the same window is ~15 µs ⇒ MNE was
~6× off the floor before any I/O happens.

Attribution (cProfile, 200 calls):

- `_handle_tmin_tmax` touches `self.times` even when `tmin=tmax=None`
  (`mne/utils/mixin.py`) and Raw's `times` property allocates
  `np.arange(n_times)/sfreq` each call — **600 KB allocated+discarded per window**
  at 300 s; scales linearly with recording length (≈59 MB per call for an 8 h TUH
  recording!).
- Channel picks resolved **twice** per call (once in `get_data`, again in
  `_parse_get_set_params`); the first goes None→`"all"`→string machinery with
  ~65 `list.index` lookups per call.
- EDF reader extras (non-preloaded only): `from scipy.interpolate import
  interp1d` executed inside `_read_segment_file` (first call in a fresh DataLoader
  worker pays a ~0.4 s import/JIT spike), a wasted `.copy()` of every channel's
  block, and an extra full-size temp buffer.
- Per-call fixed cost curve (unpatched, ms/call vs window length):
  EDF flat ≈0.37 ms up to 2 s windows; BV/FIF ≈0.12–0.15 ms — i.e. small reads
  were almost entirely fixed overhead.

### 3.2 Patches applied (this tree)

1. `mne/utils/mixin.py::_handle_tmin_tmax`: use integer `n_times` when available
   (Raw); Epochs/Evoked keep their cheap stored-times path.
2. `mne/io/base.py::get_data`: `picks=None` short-circuits to
   `np.arange(nchan)` — exactly equivalent to `_picks_to_idx(..., "all",
   exclude=())` semantics, skips name resolution entirely.
3. `mne/io/edf/edf.py::_read_segment_file`: `interp1d` import moved into the
   mixed-sfreq stim-interpolation branch; removed the per-channel `.copy()`
   (arithmetic is out-of-place; TAL views stay valid until consumed).

### 3.3 Speedups (interleaved A/B, best-of-medians)

Preloaded random-window `get_data()` (pure access cost):

| arm | µs/call |
|---|---|
| installed = release (unpatched) | 101 |
| tree = patched | 31 |

⇒ **3.2× faster**; now within ~2× of the raw-numpy floor instead of ~6×.

Non-preloaded random windows (includes reader):

| format | unpatched µs/call | patched µs/call | speedup |
|---|---|---|---|
| EDF | 426 | 335 | **1.27×** |
| BrainVision | ~172 | ~111 | **1.55×** |
| FIF | ~192 | ~130 | **1.47×** |

Whole-suite runs agree (preloaded_windows ~100→~34 µs/call; seq_1s BV/FIF
~150→~82 µs/call) once drift is controlled.

Correctness: `check_equivalence.py` shows bit-identical output vs the release
copy for every pattern (full read, boundary-crossing windows, tmin/tmax, int /
str / slice / negative picks); upstream suites pass
(`test_edf.py` + `test_brainvision.py` 116 passed, `test_raw.py` 63 passed,
`test_epochs.py` 239 passed, targeted evoked tests passed).

## 4. The deepcopy question ("I hear there's a problem")

Traced with a tracer that rebinds *every* `deepcopy` reference held by loaded
`mne.*` modules (a plain `copy.deepcopy` monkeypatch misses `from copy import
deepcopy` bindings — first tracer attempt caught almost nothing):

| Operation | deepcopy calls | total time |
|---|---|---|
| `read_raw_X(preload=True)` ×3 formats | 141 (mostly Info internals) | 0.64 ms |
| load+crop+reref+notch+resample | 2 | 0.07 ms |
| Epochs creation + iterate 20 epochs | 238 | 1.65 ms |
| `deepcopy(info)` alone, 64 ch | 1 | 0.05 ms |

Verdict: on current main, **deepcopy is not the bottleneck for DL-style data
access**. It becomes measurable only with large Infos — copies scale ~66 µs
(32 ch) → ~400 µs (512 ch), roughly ×1.5–2 with a montage/dig set — or when an
operation copies Info thousands of times. If complaints trace to older MNE
versions, note `Info.__deepcopy__` already contains fast paths (chs shallow-copy
trick) that recent releases added.

The real fixed-cost culprits were the ones fixed above (times array materialized
per call; double pick resolution), plus the EDF reader's per-channel Python loop.

## 5. Remaining bottlenecks, ranked (candidates for next PRs)

1. **EDF non-preloaded small reads still pay ~300 µs fixed cost** (per-channel
   Python loop over blocks + temp buffer + block bookkeeping). Vectorizing the
   uniform-sfreq case (the overwhelmingly common one) could cut most of it.
2. **File opened per `get_data()` call** in all three readers (~5–10 µs +
   syscalls). Keeping an open handle per Raw would help streaming patterns.
3. **Batched window reads**: a `raw.get_windows(starts, width)` style API could
   amortize validation/picks across hundreds of windows (the remaining ~30 µs
   fixed cost per call would drop to ~µs amortized).
4. **`save()` to FIF is slow**: ~105 ms for a 19.7 MB payload (~190 MB/s) — this
   is the caching path DL pipelines hit constantly; tag serialization looks
   unoptimized (separate investigation).
5. Output dtype is always float64; DL users immediately cast to float32.
   An opt-in `dtype="float32"` read path halves memory traffic.
6. `open_meta` ≈ 2 ms/format — matters when instantiating datasets over ~10k
   files (20 s just for headers); mostly Python-side parsing/checks.

## 6. Provider-library dissection & BIG multi-format study (session 2)

Setup: second preset `--preset big` = **128 ch × 1800 s @ 512 Hz** (118M samples;
float32 472 MB / int16 236 MB), same signal written to every format:
`data_big/` holds EDF(int16), **BDF(int24)**, BrainVision(f32), FIF(f32),
NPZ(f32), HDF5(f32; three chunk layouts), Zarr(f32; two layouts), raw .bin.
Backends compared by `bench_backends.py`; per-call latencies via interleaved A/B.

### 6.1 Full-file loads (BIG)

| backend | time | effective MB/s |
|---|---|---|
| floor_memmap (f32 bin) | 152–156 ms | ~3050 |
| h5 contiguous | 153 ms | ~3077 |
| npz | 206 ms | ~2289 |
| **edfio eager EDF** | **244 ms** | ~967 |
| mne_edf | 454–500 ms | ~950–1040 (vs 236 MB payload) |
| mne_fif | 535–552 ms | ~855 |
| mne_bv | 814–842 ms | ~560 |
| mne_bdf | 995–1003 ms | ~470 |
| zarr t10s chunks | 4688–5099 ms | ~100 (!!) |

Corroborates the MNE-forum report (Dec 2023): edfio ≈ 2× faster than MNE on
uniform-sfreq EDF; MNE's catastrophic mixed-sfreq cases (minutes) come from its
upsample-to-max-sfreq policy. Upstream floated making edfio an optional reader
backend — worth pursuing for `read_raw_edf`.

### 6.2 Random-window reads (the DL pattern; 300 × 2 s windows)

| backend | µs/window |
|---|---|
| h5_win (chunks = 128×512 f32) | **53–94** — beats memmap floor! |
| floor_memmap | 142 |
| h5 contiguous | 28–94 |
| h5_t10s (per-channel slabs) | 725–930 |
| zarr_win | 1678–2535 |
| mne_fif / mne_bv non-preloaded | 1298–1696 |
| edfio lazy per-signal slices | ~1158 |
| zarr_t10s | ~31700 |

Two big lessons:
1. **Chunk geometry must mirror access geometry.** Window-shaped HDF5 chunks
   turn each sample into ONE contiguous ~512 KB read and even beat the raw
   memmap "floor" (which touches 128 scattered pages per window). Time-slab
   chunking is 13× worse; misaligned Zarr chunks are pathological.
2. **Zarr v3's per-chunk Python dispatch makes it a poor local training
   backend** despite identical chunk shapes to HDF5 (~20× slower here).
   HDF5 or plain memmap/npy for single-node; Zarr for remote/parallel.

### 6.3 Where MNE's remaining window cost lives (128 ch)

After this round of patches, steady-state per-window (get_data incl. .sum()):

| format | non-preloaded | preloaded |
|---|---|---|
| EDF | ~400–430 µs | ~79 µs |
| BDF | ~1000 µs | ~79 µs |
| BrainVision | ~515 µs | ~80 µs |
| FIF | ~416 µs | ~78 µs |

Attribution experiments (`_getitem` bypass): at 128 ch, public-API validation
was consuming **55–58 %** of BV/FIF window latency (double pick resolution,
astype copies, reductions). The `_picks_to_idx` integer-fast-exit patch below
reclaims much of it; a future batched `raw.get_windows(starts, width)` API
could amortize nearly all of it (validation once, then tight read loop).

### 6.4 Patches added in session 2 (this tree)

4. `mne/io/edf/edf.py::_read_segment_file`: vectorized fast path for
   uniform-sfreq EDF/BDF with no TAL/stim channels among requested ones
   (gated to decoded outputs ≤ 32 MB so huge sequential loads keep the legacy
   cache-friendly loop). Replaces the per-channel Python loop with one
   reshape + strided gather + 3 vector ops; writes straight into the output
   buffer when no projector/compensation is active and cals == 1 (always true
   for EDF/BDF). Bit-identical output (md5-verified on small set; max |diff|
   = 0 on BIG cross-checks).
5. `mne/_fiff/pick.py::_picks_to_idx`: fast return for integer arrays already
   in range (skips astype copy, two boolean-reduction passes, modulo pass).
   Benefits every hot call site resolving array picks.

Measured against **pristine main** (interleaved subprocesses, best-of):

| metric (BIG file) | pristine | patched | speedup |
|---|---|---|---|
| EDF random windows | 1317–1383 µs | 405–414 µs | **3.3×** |
| EDF 4-channel windows | 592–599 µs | 90–97 µs | **6.3×** |
| BDF random windows | ~2000 µs | ~1000 µs | **~2.0×** |
| full sequential load | ~equal within machine noise (see caveat) | | |

Caveat learned the hard way: whole-file loads (≈944 MB float64 output alloc +
page faults) swing ±100 % with background system state on this laptop; only
interleaved same-window comparisons are trustworthy. Two earlier "regressions"
were artifacts: (a) comparing against the *installed release* instead of
pristine main — main itself got faster than 1.12.1 on full loads; (b) harness
rows that re-opened/reallocated the Raw inside the timed region.

Correctness: 179 io tests + 368-test epochs/raw/bv batches pass; outputs
bit-identical to both the installed release and stashed-pristine main on every
pattern tested (windows, boundary spans, subsets, negative picks, tmin/tmax).

## 7. How others speed this up (survey notes)

- **Pre-conversion + memory-mapping**: PyRain (RainBench) reports 27–60×
  dataloading speedups over NetCDF/Dask using mmap'd samples for randomized
  sliding-window access; explicitly recommends mmap over chunked stores for
  fragmented random access on local disks. LaBraM pre-packages EEG into HDF5
  windows. Braindecode caches preprocessed data as FIF.
- **Chunk-layout alignment**: h5py/Zarr guidance and benchmarks
  (e.g., rabernat/zarr_hdf_benchmarks) — our §6.2 confirms: shape chunks like
  the reads. For (n_ch, n_time) EEG with 2 s window sampling:
  chunks=(n_ch, window) is optimal; (1, time_slab) is terrible for multi-channel
  windows; fully contiguous is best for full loads but mediocre for random
  windows on spinning rust/NVMe (still fine on macOS page cache).
- **Reader-backend competition**: pyedflib (Cython/C), edfio (vectorized numpy,
  lazy loading, partial digital slices) — edfio's design (one-shot frombuffer +
  record reshape + vectorized calibration + lazy per-signal loading) is the
  model for MNE's EDF path; adopting it as an optional backend was proposed
  upstream by MNE maintainers.
- **Amortization APIs**: DALI/WebDataset-style pre-decoded batches; in MNE
  terms, reading K consecutive windows through one resolved-picks call.

## 8. Recommended next steps (updated ranking)

1. Land sessions-1+2 patches (bit-exact, tested): shared-path fixes, EDF
   vectorized fast path, `_picks_to_idx` fast exit. ~2–6× on DL window loops.
2. Add `raw.get_windows(starts, stop)`-style batched reader (amortizes the
   remaining ~30–90 µs/call Python overhead across windows; prototype shows
   ≥50 % headroom for BV/FIF).
3. Optional edfio-backed `read_raw_edf` engine (fast path for the corpus-scale
   mixed-sfreq pathology; upstream discussion exists).
4. Publish chunk-layout guidance + a converter recipe (BIDS→HDF5 win-chunks)
   for training pipelines; keep Zarr for cloud/parallel contexts.
5. BDF int24 decode vectorization (mne_bdf is now the slowest full-loader).
6. Investigate FIF save throughput separately (fixed overhead dominates small
   saves; large saves run at ~300 MB/s).

## 9. Reproduce

```bash
python benchmarks/io_dl/generate_data.py                     # small set
python benchmarks/io_dl/generate_data.py --preset big        # BIG set (~2.5 GB)
python benchmarks/io_dl/bench_io.py                          # small-suite timings
python benchmarks/io_dl/bench_backends.py --dir data_big     # multi-backend table
AB_FMT=edf AB_PRELOAD=0 python benchmarks/io_dl/ab_test.py 5 # interleaved A/B
python benchmarks/io_dl/profile_io.py                        # profiles + deepcopy traces
python benchmarks/io_dl/check_equivalence.py                 # bit-equality vs release
python benchmarks/io_dl/check_equivalence_big.py             # numeric equality, BIG
```

Result JSONs: `results-*.json`, `backends-*.json` in this directory.


## 10. Session 3: profiling to the metal + batched reads

Method: accumulator-wrapping of hot functions (no profiler inflation) plus
interleaved subprocess A/B against both pristine main and the installed
release. py-spy available as cross-check.

### 10.1 What the traces showed (BV, BIG file, per window)

| component | before | after fix |
|---|---|---|
| `_mult_cal_one` (cast+index+scale = **3 passes** + alloc) | ~169 µs (61 %) | **one fused pass** |
| `get_data` validation layers | ~30 µs | ~21 µs |
| open+seek+fromfile glue | ~88 µs | unchanged |

### 10.2 Patches added (session 3)

6. `mne/_fiff/utils.py::_mult_cal_one`: fuse gather + float-cast + calibration
   into a single strided multiply into the output view (`np.multiply(one[idx],
   cals, out=data_view)`), eliminating a full-size intermediate allocation and
   two extra passes. Numerically identical (elementwise ops on same values);
   all suites pass, outputs bit-identical.
7. `mne/io/edf/_bdf_numba.py` (new) + hook in `_read_ch`: numba-accelerated
   int24→int32 BDF decoder following MNE's existing optional-numba pattern
   (`mne/_numba.py`), with graceful fallback to the vectorized-numpy path.

### 10.3 Cumulative effect (BIG file, 2 s windows, non-preloaded)

Interleaved A/B vs installed release:

| format | session start | now | total speedup |
|---|---|---|---|
| BrainVision | ~1300 µs | ~225–232 µs | **~5.7×** |
| FIF | ~1700 µs | ~291 µs | **~5.8×** |
| EDF | ~2080 µs | ~475 µs | **~4.4×** |
| BDF | ~2150 µs | ~561 µs | **~3.8×** |

Preloaded access is unchanged at ~78–80 µs/window (= numpy slice floor).

### 10.4 The batched-read ceiling (adopt today without touching MNE)

Reusing one output buffer and calling internal read machinery directly
(`raw._read_segment(..., data_buffer=buf, sel=arange)`) vs public `get_data`,
measured in a single run (within-run ratios are meaningful):

| format | public | internal | buffered |
|---|---|---|---|
| EDF | 420 µs | 417 µs | **338 µs** |
| BDF | 553 µs | 541 µs | **501 µs** |
| BV | 538 µs | 208 µs | **199 µs** |
| FIF | 537 µs | 296 µs | **266 µs** |

This is the case for an upstream batched API (`raw.get_windows(starts,
width)`): resolve picks/cals once, fill a preallocated buffer per window —
PyTorch's own data-loading tutorial measures the same pattern (`__getitems__`)
at ~2.9× marginal throughput.

### 10.5 Where the remaining time goes (honest accounting)

Per 2 s window at 128 ch (~512 KB payload):
- preloaded path is already AT the numpy floor;
- non-preloaded paths now spend their budget on genuine I/O + format decode:
  EDF overreads whole records (format-inherent ~1.5×), BDF decodes 3 bytes per
  sample by design, BV/FIF do one contiguous read + one cast/scale pass.
- Next levers, in order: persistent mmap/handle per Raw (saves ~30–60 µs of
  syscall glue per call), upstreaming the batched API, FIF tag-layer dispatch
  caching (`_compare_version` etc. seen in early profiles).


## 11. Session 4: BIDS format completeness + batched reader + memmap path

### 11.1 BIDS-accepted formats — full coverage check

Per the current BIDS spec (EEG section): EEG MUST be EDF, BrainVision,
EEGLAB (.set/.fdt), or Biosemi (.bdf); iEEG additionally allows MEF and NWB.
Benchmark status of every accepted format:

| BIDS format | status in this benchmark |
|---|---|
| EDF / EDF+ | done (sessions 1–3) |
| BrainVision | done (sessions 1–3) |
| Biosemi BDF | done (session 2, incl. numba int24 decoder) |
| **EEGLAB .set/.fdt** | **added (this session)** — writer via scipy.savemat + f32 .fdt |
| **NWB** | **added (this session)** — written with pynwb; read directly (MNE has no NWB reader) |
| MEF (iEEG) | deferred: pymef is installed but its write API requires manual segment-metadata assembly; reading benchmarks need a real corpus |

### 11.2 New-format results (BIG file: 128 ch, 1800 s @ 512 Hz)

| backend | full load | windows (µs/win) |
|---|---|---|
| mne_set (EEGLAB) | 511 ms (~924 MB/s) | 986 |
| **nwb (pynwb/h5py direct)** | **136 ms (~3.5 GB/s)** | **100** |

NWB's time-major storage makes a window one contiguous row-block — near-optimal
for DL sampling without any chunk tuning. EEGLAB performs like BrainVision
(multiplexed binary + header parse). Practical guidance for BIDS corpora:
EDF/BrainVision are fine after our reader fixes; if you control the export
format for training-only copies, time-major stores (NWB-style or
window-chunked HDF5) give the best random-window behavior.

### 11.3 Speed patches added (session 4)

8. `BaseRaw._get_windows(starts, width, *, out=None, sel=None)` (internal API):
   resolves channel selection once and fills an optional reusable
   ``(n_win, n_ch, width)`` buffer — zero per-window allocations. Verified
   bit-identical to per-window `get_data`; fork-safe.
9. `_read_segments_file` (generic binary reader used by BrainVision et al.):
   persistent `np.memmap` cached on the Raw's extras, keyed by PID so forked
   DataLoader workers create their own mapping. Removes per-call
   open/seek/read syscalls. BV windows: ~230 → **~189 µs** public-API.

Batched vs per-call (300 × 2 s windows, identical per-window reductions,
buffer reused across repetitions): EDF 1.27×, BDF 1.15×, BV 1.14×,
FIF 1.05× — plus the elimination of ~315 MB/epoch of allocation churn.

All suites green (246 io tests), outputs bit-identical to release on small set
and to pristine main on the BIG set (worst diff 0.0).

## 12. Session 5: multi-agent round — FIF mmap, searchsorted, stim fast path

Deployed per workstream (subagent infra was flaky; W1/W2 landed via direct execution):

| Workstream | Outcome |
|---|---|
| W1 FIF mmap | **LANDED**: `mne/_fiff/_mmap_cache.py` (PID-keyed, mtime+size validated) + partial-tag byte-offset reads in `Raw._read_segment_file`; gzip/file-like/odd-tag fallback to legacy loop. Gates: test_raw_fiff 37✔, equivalence ✔, A/B below |
| W2 EDF/BDF numba | **LANDED** (by agent): fused decode kernel `mne/io/edf/_edf_numba.py`, `fastmath=False` required for bit-exactness; verified numpy fallback under `MNE_USE_NUMBA=false`; also fixed NumPy-2 uint32 overflow in `_bdf_numba.decode_int24` no-numba path + missing `has_numba` guard |
| opencode idea #8 | **LANDED**: `searchsorted` on sorted bounds replaces O(n_ent) mask per call (`fiff/raw.py`); property-tested vs boolean mask on 2000 random span cases |
| opencode idea #9 | **LANDED**: EDF/BDF fast path no longer abandons when uniform stim channels are among picks — replicates legacy post-calibration truncating bitmask per row; synthetic EDF+STATUS channel verified md5-identical vs release for all/stim-only/mixed picks |
| W5 stores | Chunk-cache tuning ruled out (±5%); zarr v3 ≈10× slower locally with identical chunks → remote/parallel-only guidance; h5py driver="core" −26% on slabs at ~1 GiB RAM cost |
| W7 writes | "FIF save fixed overhead" root-caused to lazy `from mne_bids import BIDSPath` inside `_check_fname` (~105 ms first save/process; warm saves ~1220 MB/s). pybv local install has a 10× write regression + in-place caller mutation bug (upstream is fine) |
| W3 glue / W6 MEF / W4 edfio | pending (infra flakiness / next round) |

### Session-5 cumulative A/B (BIG file, 2 s windows, public get_data, best-of interleaved)

| format | pristine main | patched tree | speedup |
|---|---|---|---|
| EDF | 1472 µs | **290 µs** | **5.1×** |
| BDF | 2166 µs | **405 µs** | **5.4×** |
| BrainVision | 927 µs | **189 µs** | **4.9×** |
| FIF | 1036 µs | **239 µs** | **4.3×** |

All outputs bit-exact (small-set md5 identical vs release; BIG-set worst diff 0.0);
246 io/fiff/pick tests green.

### External-auditor ideas adopted vs queued
Adopted: searchsorted bounds (#8), stim-in-fast-path (#9), codex's FIF mmap prototype.
Queued: EDF mixed-sfreq partial fast path (decode uniform EEG blocks, interp stim
separately — needs careful interp semantics), float32 output read path,
open_meta header-parse cost, batched-API upstream proposal, edfio engine backend.
