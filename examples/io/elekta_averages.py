"""
================================
Getting averaging info from fiff
================================

Get averaging information from a fiff file (for Vectorview/TRIUX systems)
and extract epochs accordingly.
"""
# Author: Jussi Nurminen (jnu@iki.fi)
#
# License: BSD (3-clause)


import mne
import os
from mne.datasets import testing
from mne.event import ElektaAverager

elekta_base_dir = os.path.join(testing.data_path(), 'misc')
fname_raw_elekta = os.path.join(elekta_base_dir, 'test_elekta_3ch_raw.fif')


print(__doc__)

raw = mne.io.read_raw_fif(fname_raw_elekta)

eav = ElektaAverager(raw.info)

# check out which averaging categories were defined in DACQ
print eav.categories

# extract epochs corresponding to a category
cat = eav['Event 1 followed by 2 within 1100 ms']
eps = eav.get_epochs(raw, cat)

""" Read all categories, extract corresponding epochs, average and save to new
 fiff file. """
evokeds = []
for cat in eav.categories:
    eps = eav.get_epochs(raw, cat)
    evoked = eps.average()
    evoked.comment = cat['comment']
    evokeds.append(evoked)

fn_out = 'eav_evokeds.fif'
mne.write_evokeds(fn_out, evokeds)

