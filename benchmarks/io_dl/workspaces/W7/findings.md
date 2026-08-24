# W7 — Write-path decomposition for DL caching pipelines

Date: 2026-08-24 · Agent W7 · Measurement-only (no library edits).
Machine: Apple Silicon (arm64), macOS 15, py 3.12.12, numpy 2.4.6, MNE 1.13.dev0
(this working tree, incl. sessions 1–4 reader/access patches), edfio 0.4.13,
pybv 0.7.6 (locally patched in site-packages — see §3).

Method: medians over repeated saves to /tmp (deleted after), GC off, floors and
writers measured back-to-back in single-session passes (`bench_w7_consolidated.py`,
3 passes; spread reported). Decomposition via accumulator-wrapping of mne
internals (module-attribute rebinding, no cProfile inflation) + one cProfile run.
Prototypes monkeypatched in-process, outputs md5-verified byte-identical.

## 1. Headline table

| writer | payload | MB/s | % fixed overhead | top hotspot function / file:line |
|---|---|---|---|---|
| **FIF save, small, warm preloaded** | 19.7 MB f32 | **1220** (16.2 ms) | **~19 %** true logic (fit: 2–6 ms/file); ~87 % if first-save-in-process (see §2.1) | payload cast+transpose+`tobytes` copies: `write_float`/`_write` mne/_fiff/write.py:94,27; per-buffer fetch `BaseRaw.__getitem__` mne/io/base.py:973 |
| **FIF save, small, NON-preloaded** | 19.7 MB | 472 (41.7 ms) | ~60 % = per-buffer source reads through EDF reader (~87 µs × 300 buffers) | `_read_segment_file` buffer fetch per 1 s output buffer, called from `_write_raw_data` loop mne/io/base.py:3197 |
| FIF save, big (128 ch × 1800 s) | 471.9 MB f32 | **1478–1600** (295–320 ms) | ≈0 (marginal regime) | same as small; CPU side = `buf / cals` copy + `np.array(">f4")` strided cast + `.tobytes()` extra copy (`_write_raw_buffer` mne/io/base.py:3324 → `_write`) |
| floor: `np.tofile` big f32 | 471.9 MB | 4400–5700 (71–107 ms) | – | – |
| **BrainVision (installed pybv)** | 471.9 MB f32 | **199** (2420 ms) | ~90 % vs its own layout floor | chunked `chunk.T.tofile(fout)` pybv/io.py:827 — non-contiguous transposed write |
| BrainVision (pristine upstream 0.7.6) | 471.9 MB f32 | **958** (493 ms) | ~40 % | 3 internal full-array copies: `data * scales`, `astype(dtype)`, `ravel(order="F")` pybv/io.py:683,799,811 |
| floor: multiplexed layout (`ravel(F)`+tofile) | 471.9 MB | 1929–2406 (197–245 ms) | – | – |
| prototype fix (chunked `ascontiguousarray(chunk.T)`) | 471.9 MB | **2628** (191 ms) | ~0 | one fused transpose-copy per ≤32 MB block |
| **EDF write (edfio), serialization only** | 236.0 MB i16 | **3156–4000** (59–75 ms) | ≈0 | single contiguous `data_record.tofile()` edfio/_edfio.py `Edf.write` |
| EDF end-to-end (construct + int16 convert + write) | 236.0 MB i16 | **772–897** (263–306 ms) | conversion-dominated | f64→i16 digital quantization at `EdfSignal` construction (memory-bound pass, ~330 ms measured standalone) |

Notes on the table:
- "% fixed overhead" for FIF small = two-point fit `t(bytes) = fixed + bytes/rate`
  over {small, big} within-session: fixed = 2.3–5.7 ms/file across 3 passes,
  marginal rate = 1490–1632 MB/s. The earlier "~190 MB/s ⇒ huge fixed cost" flag
  is explained by §2.1/§2.2, not by tag machinery.
- BV "its own layout floor" = writing the same multiplexed (time-major) byte
  layout via `volts.ravel(order="F").tofile()`.
- edfio converts float→digital-int16 at signal construction, so its `write()`
  alone is nearly free; the honest end-to-end number is the last row.

## 2. FIF save decomposition (mission item 1)

Instrumented component breakdown (accumulator wrapping, ms):

| component | small warm preloaded (17.6 ms total) | small NON-preloaded (41.7 ms) | big preloaded (281 ms instr.) |
|---|---|---|---|
| `_check_fname` + `check_fname` regexes | 0.13 (0.7 %) | 0.13 | 0.15 |
| info deepcopy | 0.15 (35 calls) | 0.14 | 0.24 |
| `write_meas_info` (metadata tags) | 0.68 | 0.67 | 1.27 |
| annotations handling | 0.01 (empty) | 0.01 | 0.01 |
| split-size logic (`fid.tell` checks) | <0.5 (in glue) | <0.5 | <1 |
| `_write_raw_buffer` (incl. payload `_write` inside) | 11.54 | 9.94 | 214.31 |
| per-buffer source read (`raw[picks, first:last]` via EDF reader) | 3.44 (11.5 µs/call) | **≈25** (glue row, 60 %) | ~21 (est. from 11.5 µs × 1800) |
| unattributed Python glue (loop, asserts, logger) | ~1 | 24.75 | ~45 |

Tag counts: 319 tags for small (300 payload + 19 metadata), 1819 for big;
metadata bytes = 0.0003 MB. Disk I/O is only 4.6 ms (small) / 71–107 ms (big)
of the totals — everything else is CPU-side array traffic.

### 2.1 The historical "~103 ms" mystery — SOLVED

A fresh process's first `raw.save()` costs **121 ms**; with `mne_bids` already
imported it is **16 ms**. Cause: `_check_fname(..., check_bids_split=True)`
(mne/utils/check.py:270, called from BaseRaw.save at mne/io/base.py:1967 and
Epochs.save at mne/epochs.py:2357) executes `from mne_bids import BIDSPath`
inside a try/except. When mne-bids is installed this lazily imports mne_bids →
mne.viz → jinja2 → scipy.sparse/scipy.special/scipy.io (hundreds of modules;
standalone `from mne_bids import BIDSPath` = 265–279 ms; ≈105 ms after
`import mne`). Every caching pipeline pays this once per process (per DataLoader
worker!), and it was misattributed to tag serialization.

Secondary factor: the old benchmark saved a NON-preloaded raw
(`bench_io.py::bench_micro`, `profile_io.profile_save`), adding per-buffer EDF
reader costs (now 42 ms; was ~100 ms before the session-1 reader patches).

### 2.2 Rejected hypotheses (measured, not guessed)

- **info deepcopy**: NOT executed by default (`proj=False` passes `self.info` by
  reference, base.py:2006). With `proj=True`: +3.9 ms total, almost all
  `setup_proj` SVD work, not the copy; `deepcopy(info)` alone ≈ 0.05 ms @64ch.
  "Skip deepcopy when projs/comps empty" would save ~nothing on DL paths.
- **Tag serialization loop**: 19 metadata tags = 0.68 ms. Negligible.
- **Annotations**: empty case = 0.01 ms.
- **Split-size logic**: two `fid.tell()` + comparisons per buffer; µs-scale.
- **Bigger `buffer_size_sec`**: tested 4/16/60 s on BIG save — all SLOWER than
  default 1 s (400/425/382 vs 297 ms; larger temps blow L2/L3). Reject.

## 3. BrainVision writer (mission item 2)

The site-packages pybv 0.7.6 has been locally patched (vs pristine PyPI wheel:
`data * scales` → in-place `*=`, full `ravel(order="F").tofile()` → chunked
`chunk.T.to_file`). Two defects introduced:

1. **Performance pessimization**: `ndarray.T.tofile()` on a (128, n)-slice view
   writes ~10× slower than necessary (numpy iterates the transposed view with
   poor buffering): microbench same layout — `T.tofile` 2244–2341 ms vs
   `ravel(F).tofile` 224 ms. Result: 199 MB/s installed vs 958 MB/s pristine
   vs 2628 MB/s achievable (§1 prototype B: `np.ascontiguousarray(chunk.T)`
   per ≤32 MB block, then scale+write — bounded memory AND fastest).
2. **Caller-visible mutation**: in-place scaling multiplies the user's array by
   unit/resolution factors every call (verified: input grows ×1e13 per call,
   overflow→inf→ValueError on 3rd call). Upstream returns copies; any pipeline
   re-using a data array across `write_brainvision` calls silently corrupts it.

Cost structure otherwise: scaling passes ≈108 ms of pure memory traffic
(2 multiplies + range check) out of ~2400 ms — i.e., I/O-pattern cost, not
copies, dominate. `raw.export(fmt="brainvision")` routes through pybv, so MNE
inherits both defects.

## 4. EDF writer (mission item 3)

Confirmed and refined the earlier "~590 MB/s": that was end-to-end. edfio's
`Edf.write` itself moves bytes at **~3200–4000 MB/s** (one contiguous
`tofile` of a pre-packed uint8 record matrix); the f64→int16 digital
quantization happens earlier, at `EdfSignal` construction, and costs ~330 ms
for 118 M samples (a pure memory-bound pass; measured standalone). End-to-end
construct+convert+write = 263–306 ms (**772–897 MB/s**, matches the earlier
observation). Headroom: the conversion could stream/fuse with generation, or
accept f32 input directly (halving one pass); disk-side nothing left.

## 5. TOP-3 fix proposals (ranked by expected gain)

### #1 — Fix the BrainVision binary write path (pybv patch or upstream PR)
Replace `chunk.T.tofile(fout)` with
`np.ascontiguousarray(data[:, s:e].T).tofile(fout)` blocks (≤32 MB), folding
the two scaling multiplies into one multiply on each transposed block (also
removes the caller-mutation bug since scaling then happens on the private
block copy). Expected: **2420 → ~200 ms on a 472 MB file (≈12×)**; MNE
`raw.export(fmt="brainvision")` inherits the win. Risk: none to output bytes
(layout identical; md5-verifiable against upstream writer).

### #2 — De-guard the lazy `mne_bids` import in `_check_fname`
Only attempt `from mne_bids import BIDSPath` when `"mne_bids" in sys.modules`
(a real BIDSPath cannot reach these call sites unless mne-bids was imported),
or duck-type on `hasattr(fname, "split")`. Expected: **first save per process
121 → 16 ms (−105 ms)** for every pipeline with mne-bids installed; hits
`Raw.save` and `Epochs.save` (the two DL caching entry points) plus read-path
callers of `_check_fname`. Behavior-identical (import succeeds iff mne_bids
already loaded ⇒ identical validation outcomes). Trivially upstreamable.

### #3 — Fuse the FIF payload buffer write into one pass
In `_write_raw_buffer` (single fmt): divide once into a fresh `(n_t, n_ch)`
`">f4"` C-contig output and `fid.write(out)` directly (buffer protocol),
eliminating the `buf/cals` temp, the separate strided cast, and the redundant
`tobytes()` bytes-object copy (4 payload passes → 2). Prototype verified
byte-identical md5 on the 472 MB save; measured gain modest but real:
**≈5–15 % on warm saves** (big 306→293 ms; small ~16→~14 ms), larger on
memory-constrained machines. Complementary free win: avoid saving from
NON-preloaded raws when possible (document/batch source reads) — that state
costs 42 vs 16 ms per small file (2.7×) today.

Rejected for lack of measured support: skip-info-deepcopy fast path (§2.2),
default `buffer_size_sec` changes (§2.2), split-size/tag-loop micro-opts.

## 6. Repro

```bash
cd /Users/braristimunha/Projects/libraries/mne_python/mne_python_more_io_speed
python benchmarks/io_dl/workspaces/W7/bench_w7_write.py --only floor   # np.tofile floors
python benchmarks/io_dl/workspaces/W7/bench_w7_write.py --only fif     # small+BIG FIF decomp, cProfile
python benchmarks/io_dl/workspaces/W7/bench_w7_write.py --only fifx    # non-preloaded, fetch, cold-process A/B
python benchmarks/io_dl/workspaces/W7/bench_w7_write.py --only bv      # installed vs pristine pybv + prototypes
python benchmarks/io_dl/workspaces/W7/bench_w7_write.py --only edf     # write-only vs end-to-end
python benchmarks/io_dl/workspaces/W7/bench_w7_consolidated.py         # single-session pass (run 2-3x)
# cold-save mne_bids A/B:
python -c "import time,sys; sys.path.insert(0,'.'); import mne; mne.set_log_level('ERROR'); \
  from mne.io import read_raw_edf; r=read_raw_edf('benchmarks/io_dl/data/bench.edf',preload=True); \
  t=time.perf_counter(); r.save('/tmp/x.fif',overwrite=True); print((time.perf_counter()-t)*1e3)"
# compare with 'import mne_bids' inserted before the timer.
```

JSON artifacts: `results.json`, `consolidated.json` next to this file.
Requires `pip download pybv==0.7.6 --no-deps -d /tmp/pybv_up && unzip
/tmp/pybv_up/*.whl -d /tmp/pybv_up/x` for the pristine-pybv arm.

COMPLETE: W7 - The FIF "fixed overhead" is a lazy `from mne_bids import BIDSPath` inside `_check_fname(check_bids_split=True)` costing ~105 ms of every process's first save (121→16 ms measured); worst writer is the locally-patched pybv at 199 MB/s (fixable to ~2600 MB/s, 12×), best is edfio serialization at ~3600 MB/s.
