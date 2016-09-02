"""
================================
Getting averaging info from fiff
================================

Get averaging information defined in Elekta Vectorview/TRIUX DACQ (data
acquisition). Extract and average epochs accordingly.
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

fname_out = 'elekta_evokeds-ave.fif'
mne.write_evokeds(fname_out, evokeds)

""" Make a new category using an existing one as a template and extract
corresponding epochs. """
newcat = eav.categories[0].copy()
newcat['comment'] = 'New category'
newcat['event'] = 1  # reference event
newcat['start'] = -.1  # epoch start rel. to ref. event (in seconds)
newcat['end'] = .5  # epoch end
newcat['reqevent'] = 2  # additional required event; 0 if none
newcat['reqwithin'] = 1.5  # req. event required within 1.5 sec of ref. event
newcat['reqwhen'] = 1  # required before (1) or after (2) ref. event

eps = eav.get_epochs(raw, newcat)
