#!/usr/bin/env python

from pathlib import Path
import mne

want_mne_dir = Path(__file__).parent.parent / "mne"
got_mne_dir = Path(mne.__file__).parent
want_mne_dir = want_mne_dir.resolve()
got_mne_dir = got_mne_dir.resolve()
print(f"Expected import mne from:\n{want_mne_dir}\nAnd got:\n{got_mne_dir}")
assert want_mne_dir == got_mne_dir
