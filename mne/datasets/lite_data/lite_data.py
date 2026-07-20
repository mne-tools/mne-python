# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

"""Curated data subset used by the JupyterLite browser documentation.

The canonical MNE datasets (``sample``, ``kiloword``, ``erp_core``, ``mtrf``,
``eegbci``) ship as single archives, so touching a handful of files would mean
downloading whole multi-GB datasets -- both in the docs build and in the
browser. ``lite_data`` re-hosts just the files the JupyterLite notebooks use --
same content, same md5 checksums -- addressed individually and written in the
canonical on-disk layout the datasets expect. Fetching this slim subset in the
docs build (instead of the full datasets) is what speeds the build up; because
the layout matches, the notebooks need no changes.

Total: 36 files, ~672 MB.
"""

import time
from importlib.resources import files as _pkg_files
from pathlib import Path

from ...utils import verbose
from ..utils import _do_path_update, _downloader_params, _get_path, _log_time_size

# The curated files are hosted individually on OSF (project osf.io/u6pej).
# Each file's absolute download URL lives in ``lite_data_urls.txt`` and
# overrides ``base_url`` per file in ``pooch.create``; ``base_url`` is only a
# nominal fallback (OSF download links are opaque IDs, not path-based).
LITE_DATA_BASE_URL = "https://osf.io/download/"

LITE_DATA_VERSION = "0.1"

_CONFIG_KEY = "MNE_DATASETS_LITE_DATA_PATH"


def _load_registry():
    """Parse the packaged registry file into a {relpath: 'md5:...'} dict."""
    text = (_pkg_files("mne.datasets.lite_data") / "lite_data_registry.txt").read_text()
    registry = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        relpath, known_hash = line.split()
        registry[relpath] = known_hash
    return registry


def _load_urls():
    """Parse the packaged url file into a {relpath: absolute_url} dict."""
    text = (_pkg_files("mne.datasets.lite_data") / "lite_data_urls.txt").read_text()
    urls = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        relpath, url = line.split()
        urls[relpath] = url
    return urls


@verbose
def data_path(path=None, force_update=False, update_path=True, *, verbose=None):  # noqa: D103
    import pooch

    registry = _load_registry()
    urls = _load_urls()
    missing = set(registry) - set(urls)
    if missing:
        raise RuntimeError(
            f"lite_data is missing download URLs for {len(missing)} file(s), "
            f"e.g. {sorted(missing)[0]!r}. Regenerate lite_data_urls.txt."
        )
    key = _CONFIG_KEY
    name = "LITE_DATA"
    path = _get_path(path, key, name)
    root = Path(path)
    pup = pooch.create(
        path=root,
        base_url=LITE_DATA_BASE_URL,
        registry=registry,
        urls=urls,
    )
    downloader = pooch.HTTPDownloader(**_downloader_params())
    t0 = time.time()
    n_bytes = 0
    for relpath in pup.registry:
        if force_update:
            stale = root / relpath
            if stale.is_file():
                stale.unlink()
        dest = Path(pup.fetch(relpath, downloader=downloader))
        n_bytes += dest.stat().st_size
    _do_path_update(path, update_path, key, name)
    _log_time_size(t0, n_bytes)
    return root


data_path.__doc__ = """Get path to a local copy of the curated JupyterLite data subset.

Parameters
----------
path : None | path-like
    Location to download the curated data to. If ``None``, the config value
    ``MNE_DATASETS_LITE_DATA_PATH`` (or ``~/mne_data``) is used. Files land in
    canonical dataset layout, so e.g. ``sample`` data is written under
    ``<path>/MNE-sample-data`` -- exactly where
    :func:`mne.datasets.sample.data_path` looks for it.
force_update : bool
    Force update of the dataset even if a local copy exists.
update_path : bool | None
    If ``True``, set ``MNE_DATASETS_LITE_DATA_PATH`` in the config to ``path``.
verbose : bool | str | int | None
    Control verbosity of the logging output.

Returns
-------
path : instance of Path
    Path to the directory holding the curated data (the ``mne_data`` root).
"""


def get_version():  # noqa: D103
    return LITE_DATA_VERSION


get_version.__doc__ = """Get the version of the curated lite_data subset.

Returns
-------
version : str
    Version of the curated lite_data subset.
"""
