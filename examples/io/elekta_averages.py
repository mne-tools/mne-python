"""
================================
Getting averaging info from fiff
================================

Get averaging information defined in Elekta Vectorview/TRIUX DACQ (data
acquisition). Extract and extract epochs accordingly.
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

""" Extract epochs corresponding to a category. Copy rejection
limits from DACQ settings. """
eps = eav.get_epochs(raw, 'Event 1 followed by 2 within 1100 ms',
                     reject=True, flat=True)

""" Read all categories, extract corresponding epochs, average, add
comments from to the DACQ categories and save to new
 fiff file. """
evokeds = []
for cat in eav.categories:
    eps = eav.get_epochs(raw, cat)
    evoked = eps.average()
    evoked.comment = cat['comment']
    evokeds.append(evoked)

fn_out = 'elekta_evokeds-ave.fif'
mne.write_evokeds(fn_out, evokeds)
