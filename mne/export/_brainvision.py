# -*- coding: utf-8 -*-
# Authors: MNE Developers
#
# License: BSD-3-Clause

import os

from ..utils import _check_pybv_installed
_check_pybv_installed()
from pybv._export import _export_mne_raw  # noqa: E402


def _export_raw(fname, raw, overwrite):
    """Export Raw object to BrainVision via pybv."""
    fname = str(fname)
    ext = os.path.splitext(fname)[-1]
    if ext != ".vhdr":
        fname = fname.replace(ext, ".vhdr")
    _export_mne_raw(raw=raw, fname=fname, overwrite=overwrite)
