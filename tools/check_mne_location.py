#!/usr/bin/env python

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from pathlib import Path

import mne

want_mne_dir = Path(__file__).parents[1] / "mne"
got_mne_dir = Path(mne.__file__).parent
want_mne_dir = want_mne_dir.resolve()
got_mne_dir = got_mne_dir.resolve()
print(f"Expected import mne from:\n{want_mne_dir}\nAnd got:\n{got_mne_dir}")
assert want_mne_dir == got_mne_dir
