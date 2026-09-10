# AGENTS.md

This file provides guidance to AI coding agents when working with code in this repository.

## What this is

MNE-Python is a large open-source library for exploring, analyzing, and visualizing human
neurophysiological data (MEG, EEG, sEEG, ECoG, fNIRS, etc.): I/O for dozens of vendor formats,
preprocessing, source estimation, time-frequency, connectivity, statistics, decoding, and 2D/3D
visualization.

## AI-assistance policy (read first)

This project has an explicit policy on AI-generated contributions:

@CONTRIBUTING.md

Two rules for agents on top of that policy:

- **Do not add AI co-authorship trailers to commits.** No `Co-Authored-By:` line for Claude /
  Copilot / Cursor / any other tool, and no bot as the commit author. The human opening the PR is
  the sole author, takes full responsibility for the contents, and must be able to explain and
  defend every line on request. Disclose the tools used and the scope of their assistance in the
  PR description instead, as CONTRIBUTING.md requires.
- **Optimize for the reviewer and for long-term maintainability, not for output volume.** Every
  line you add is read by a volunteer reviewer and then maintained by humans in perpetuity. See
  "Keep changes small when possible" below — oversized diffs are the single most common problem
  with agent-assisted PRs here.

## Keep changes small when possible

A big diff is a cost, not an accomplishment. Aim for the smallest change that delivers the
user-visible behavior; anything else can be added later, when something actually needs it.

- **Sketch before you generate.** For anything beyond a local fix, summarize the design in a few
  sentences for the human you are working with, and confirm with them that it is the smallest one
  that works before writing code. Getting agreement on a design for larger work is worthwhile, but
  the human decides whether and how to raise it publicly, and writes it in their own words — as
  CONTRIBUTING.md says, do not paste agent-written text into issues, PRs, or comments. Prefer
  landing a minimal version and iterating in follow-ups over one large PR.
- **Stop and re-plan at roughly 150 new lines of implementation.** Growth past that — or reaching
  for a new module, a new test file, or a new dependency — is a signal that the design is too
  elaborate, not that progress is being made. Stop, tell the human, and offer the smaller version.
- **YAGNI: do not build what the feature does not yet need.** No manifests, format versioning or
  generations, garbage collection, custom locking, retry logic, JSON side-car metadata,
  platform-specific fallbacks, or extension points unless a test in the same PR fails without
  them. Unrequested "robustness" is the main way an agent PR reaches +1500. This goes double for
  public API: a new parameter, keyword value, or config knob needs a demonstrated end-user need,
  not a hypothetical one, because every one of them is documented, tested, and supported forever.
  Hard-code the single behavior that is actually wanted and add the option later if someone asks.
- **DRY: reuse or refactor before writing something new.** Grep for machinery that already
  exists — `__hash__` and other dunders on the object, helpers in `mne/utils/`, `docdict` entries
  for parameter text, existing generic test harnesses, joblib / NumPy / SciPy idioms — and build
  on it rather than writing a bespoke implementation. When the code you need is almost right but
  not reusable as it stands, factor the shared part out into a private helper (or generalize an
  existing private function) and call it from both places, rather than copying it and editing the
  copy. Near-duplicate blocks that drift apart are a long-term maintenance cost here.
- **Implement at the highest layer that already exists.** Adding behavior once to a base class
  (e.g. `BaseRaw`) is usually both smaller and broader than adding it to three subclasses, and it
  covers formats added later for free.
- **Never promote an optional dependency to a required one** as a side effect of a feature, and do
  not add a dependency at all without asking first.
- **Add to an existing test before writing a new one.** In order of preference: extend a test
  function that already builds the objects you need — a parametrized one especially, since the new
  assertion then runs across every case for free — then add to the existing test module for that
  code, and only then write a new test function or file. A new `mne/**/tests/test_<thing>.py` is a
  signal to stop and look, and new I/O behavior usually belongs in the generic
  `mne/io/tests/test_raw.py::_test_raw_reader`, which runs for every format. A few compact
  assertions that run everywhere beat hundreds of lines that run once, and re-created setup is one
  of the most common things reviewers ask to have deleted. In mne-tools/mne-python#14248 a 30-line
  standalone test became fewer than 10 lines added to the existing parametrized
  `test_anonymize_with_io`, which already had the fixture, the save/load round trip, and the
  `daysback` parametrization that exposed the bug.
- **Check that a new API spelling does not already mean something else.** A new sentinel or
  keyword value (`preload="auto"`, `memmap="auto"`, ...) must not collide with an existing meaning
  of the same string on a related argument.

Worked example of what to avoid: mne-tools/mne-python#14216 opened at +1523/-27, with a 628-line
private module, a 679-line dedicated test file, and an optional dependency promoted to required.
After review it delivered the same feature in ~114 implementation lines and ~150 test lines, with
no manifests, generations, scavenger, or lock file, implemented on `BaseRaw` so it worked for
every reader instead of three, and smoke-tested from `_test_raw_reader`. That second version is
what should have been written first.


## Common commands

Install an editable dev environment (see `pyproject.toml` dependency groups):
```bash
pip install -e ".[test_extra,doc]"   # or use `uv sync` with the [dependency-groups] in pyproject.toml
pre-commit install --install-hooks
```

Lint / format (ruff, codespell, yamllint, rstcheck, toml-sort, zizmor — all via pre-commit):
```bash
make ruff          # alias for `pre-commit run -a`
```

Run tests:
```bash
# whole suite (slow; needs the testing dataset, fetched automatically via pooch)
pytest -m "not ultraslowtest" mne

# a single test file / test / by keyword
pytest mne/tests/test_evoked.py::test_io_evoked --verbose
pytest mne/tests/test_evoked.py -k test_io_evoked --verbose

# fetch datasets explicitly if needed
python -c "import mne; mne.datasets.testing.data_path(verbose=True)"
python -c "import mne; mne.datasets.sample.data_path(verbose=True)"

# useful flags: -x (stop on first failure), --pdb, --durations=5,
# --cov=mne.viz --cov-report=term-missing (see which lines are covered)
```

Docstring / doctest checks:
```bash
pytest mne/tests/test_docstring_parameters.py
make test-doc          # runs doctests across doc/ (requires sample + testing datasets, generally only needed when changing example code in doc/ itself)
```

Build the docs (Sphinx + sphinx-gallery, in `doc/`):
```bash
PATTERN=some_regex_pattern make -C doc html-pattern  # can choose some_regex_pattern to subselect relevant examples and tutorials to run
make -C doc html        # full build, takes about an hour, generally only needed if changing docs extensively
```

Other:
```bash
make nesting        # import-nesting checks (mne/tests/test_import_nesting.py)
make clean           # remove build artifacts, __pycache__, *.pyc/*.so
```

There is no separate "build" step for the library itself beyond the editable install (pure
Python + hatchling/hatch-vcs for versioning from git tags).

## Architecture

### Lazy public API via stub files
`mne/__init__.py` uses `lazy_loader.attach_stub` against `mne/__init__.pyi` — the `.pyi` file is
the actual source of truth for what's in `mne.__all__` and lazily importable, not the `.py` file.
Many subpackages (`mne/io`, `mne/utils`, etc.) follow the same `__init__.py` + `__init__.pyi`
pattern. When adding a new public function/class, it typically needs to be added to the relevant
`__init__.pyi` (and, for docs, to `doc/python_reference.rst`) as well as implemented.

### I/O readers: one subpackage per format
`mne/io/<format>/` (ant, array, artemis123, bci2k, besa, boxy, brainvision, bti, cnt, ctf, curry,
edf, eeglab, egi, eximia, eyelink, fieldtrip, fil, hitachi, kit, mef, nedf, neuralynx, nicolet,
nihon, nirx, nsx, persyst, snirf, ...) each implement a `read_raw_<format>` function and a
format-specific `Raw<Format>` subclass of `BaseRaw` (`mne/io/base.py`). `mne/io/_read_raw.py`
provides the generic `read_raw()` dispatcher. New format support follows this same shape: a
subpackage with a reader function + `BaseRaw` subclass + its own `tests/` dir with small
synthetic/testing-dataset-backed fixtures.

### FIF internals live in `mne/_fiff`, not `mne/io/fiff`
Neuromag FIF is MNE's native format and many core objects (`Info`, projections, compensators,
channel picking, tag/tree reading) depend on it, so that logic was pulled out of `mne/io/` into
`mne/_fiff/` (private) to avoid import cycles and because it's used well beyond raw I/O.
`mne/io/_fiff_wrap.py` re-exports select `mne._fiff` symbols for backward compatibility (some
were previously public under `mne.io`).

### Core data containers and mixins
`BaseRaw` (`mne/io/base.py`), `Epochs`/`BaseEpochs` (`mne/epochs.py`), and `Evoked`
(`mne/evoked.py`) are the central objects; shared behavior (channel picking/renaming, filtering,
cropping, projections, export) lives in mixins under `mne/channels/`, `mne/filter.py`,
`mne/utils/mixin.py`, etc. and is composed via multiple inheritance rather than duplicated per
class.

### Shared/templated docstrings
Common parameter descriptions live in a central dict in `mne/utils/docs.py` and are spliced into
function/method docstrings via the `@fill_doc` decorator + `%(param_name)s` placeholders — grep
for `docdict[` / `@fill_doc` before writing out a parameter docstring by hand, it's likely already
defined.

### Changelog is per-PR fragment files (towncrier), not a single hand-edited file
User-facing changes need a file `doc/changes/dev/<PR-number>.<type>.rst` (types: `notable`,
`dependency`, `bugfix`, `apichange`, `newfeature`, `other` — see `doc/development/contributing.rst`
"Describe your changes in the changelog" section for full guidance). These get aggregated into
`doc/changes/dev.rst` at release time; don't edit `doc/changes/dev.rst` or the versioned
`doc/changes/vX.Y.rst` files directly for new changes. New contributors must also add themselves
to `doc/changes/names.inc` (build fails otherwise) and are credited with `:newcontrib:` in their
changelog entry instead of a plain name link.

The `<PR-number>` for a not-yet-opened PR is one more than the highest number currently in use;
issues and PRs share a single number sequence, so query the most recently created of either
(the `issues` API endpoint includes PRs):
```bash
gh api "repos/mne-tools/mne-python/issues?state=all&per_page=1&sort=created&direction=desc" \
    --jq '.[0].number'
```

## Code conventions (beyond what ruff enforces)

- Classes: `CamelCase`. Functions/variables: `snake_case`, no abbreviated names like `nsamples`.
- Docstrings: numpydoc style with a few local deviations — no "optional" on kwargs with defaults,
  `str | None` instead of "str or None", no `Raises`/`Warns` sections, citations via
  `sphinxcontrib-bibtex` (`:footcite:`/`footbibliography::`, keys defined in `doc/references.bib`).
- Cross-reference liberally in docstrings/docs using Sphinx roles (`:func:`, `:class:`, `:meth:`,
  `:attr:`, `:mod:`, `:ref:`) — but note an API element must appear in `doc/python_reference.rst`
  for the cross-reference to resolve.
- Imports: Use absolute imports for new code (historical relative imports are tolerated in
  existing code). Optional/heavy deps (matplotlib, scipy, sklearn, pandas, ...) are imported lazily
  inside the function/method that needs them, not at module level.
- Methods mutate in place and return `self`; module-level functions return copies.
- No bare `*args`/`**kwargs` in signatures; no nested functions/methods (use private
  module-level functions instead).
- Visualization: add a function in `mne.viz` and have the corresponding object method
  (e.g. `Epochs.plot`) call it, not the reverse. All viz functions take a `show` bool. Default
  colormap is `RdBu_r` for signed/zero-centered data, `Reds` otherwise.
- Deprecations use the `@mne.utils.deprecated` decorator (functions/classes) or
  `mne.utils.warn(..., FutureWarning)` (parameters); add a test asserting the warning fires, and
  grep for internal call sites to update immediately rather than at end-of-cycle.
- Workarounds that exist only because of a minimum-version floor (an upstream bug fixed in a
  newer release, a fallback for an older Python/NumPy/SciPy/...) must be marked with a
  `# TODO VERSION` comment naming the version at which they can be removed and, when there is
  one, the upstream issue, e.g.
  `# TODO VERSION: segfaults on NumPy < 2.2.5 (numpy/numpy#28609)`. These are grepped for
  when bumping minimum versions, so a workaround without the marker tends to outlive its
  reason for existing.
- Prefer the `testing` dataset over `sample`/other large datasets in tests (smaller, faster).
- Prefer to keep unit tests compact and add to existing tests when possible. The full test suite takes about an hour on CIs, so minimizing test time (for CIs) and test verbosity (for reviewers) is important.
- When new functionality is added, it is good in general to add it somewhere in an example (`examples/`) or a tutorial (`tutorials/`) to help with discoverability and documentation.
- Code adapted from an outside source must be under a BSD-compatible license (BSD, MIT, ISC, Apache-2.0, public domain, ...); GPL/LGPL/AGPL and non-commercial or no-derivatives licenses are not acceptable. Attribute it in a comment directly above the adapted code, naming the source (URL and/or author) and its license, e.g.:
  ```python
  # Adapted from https://example.com/post by A. Nother, released under the MIT license
  # (BSD-compatible).
  ```
  If the license of a snippet cannot be determined, do not adapt it.

- Benchmarking performance changes: only interleaved A/B runs against a
  pristine snapshot built from the exact upstream base are trustworthy;
  whole-suite back-to-back runs drift ±20–100 %. Verify which installed mne a
  benchmark actually imports (`print(mne.__file__)`) before trusting numbers,
  and keep fixture data out of commits.
- Changelog fragments (`doc/changes/dev/<PR#>.<type>.rst`): pick `<type>` by
  intent — performance improvements are `newfeature`, not `bugfix`. Keep the
  entry to one short sentence ending with the contributor name link, e.g.
  "Speed up X by optimizing Y, by `Jane Doe`_", and make sure the name anchors
  in `doc/changes/names.inc` (add it if missing). Read an existing fragment
  or two before writing yours.