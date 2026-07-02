# CLAUDE.md

This file provides guidance to AI coding agents when working with code in this repository.

## What this is

MNE-Python is a large open-source library for exploring, analyzing, and visualizing human
neurophysiological data (MEG, EEG, sEEG, ECoG, fNIRS, etc.): I/O for dozens of vendor formats,
preprocessing, source estimation, time-frequency, connectivity, statistics, decoding, and 2D/3D
visualization.

## AI-assistance policy (read first)

This project has an explicit policy on AI-generated contributions (`CONTRIBUTING.md`):
- Fully-automated PRs/issues are not accepted; a human must review, understand, and be able to
  explain every change before it's submitted under their name.
- Any PR that used AI assistance must disclose that in the PR description (tool + scope of use).
- Do not paste AI-generated text into issue/PR descriptions or comments.
- Be careful that generated code for specialized algorithms doesn't reproduce copyrighted
  passages from papers/other codebases (license risk).

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
python -c "import mne; mne.datasets.sample.data_path(verbose=True)

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
cd doc && make html        # or `make html-noplot` for a faster build without executing examples
```

Other:
```bash
make nesting        # import-nesting checks (mne/tests/test_import_nesting.py)
make codespell       # manual spellcheck pass
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
- Prefer the `testing` dataset over `sample`/other large datasets in tests (smaller, faster).
