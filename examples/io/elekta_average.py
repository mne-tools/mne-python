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

# read raw file and create averager instance
raw = mne.io.read_raw_fif(fname_raw_elekta)
eav = ElektaAverager(raw.info)

# check DACQ defined averaging categories
print(eav)

# extract epochs corresponding to a category
cond = eav.get_condition(raw, 'Test event 3')[0]  # always returns a list
eps = mne.Epochs(raw, **cond)

# get epochs corresponding to each category, average and save all averages
# into a new evoked fiff file
evokeds = []
for cat in eav.categories:
    cond = eav.get_condition(raw, cat)[0]
    # copy (supported) rejection parameters from DACQ settings
    eps = mne.Epochs(raw, reject=eav.reject, flat=eav.flat, **cond)
    evoked = eps.average()
    evoked.comment = cat['comment']
    evokeds.append(evoked)
fname_out = 'elekta_evokeds-ave.fif'
mne.write_evokeds(fname_out, evokeds)

# make a new category using existing one as a template and extract
# corresponding epochs
newcat = eav.categories[0].copy()
newcat['comment'] = 'New category'
newcat['event'] = 1  # reference event
newcat['start'] = -.1  # epoch start rel. to ref. event (in seconds)
newcat['end'] = .5  # epoch end
newcat['reqevent'] = 2  # additional required event; 0 if none
newcat['reqwithin'] = 1.5  # req. event required within 1.5 sec of ref. event
newcat['reqwhen'] = 1  # required before (1) or after (2) ref. event
cond = eav.get_condition(raw, newcat)[0]
eps = mne.Epochs(raw, **cond)
