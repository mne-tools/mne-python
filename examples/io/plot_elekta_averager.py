"""
======================================
Getting averaging info from .fif files
======================================

Get averaging information defined in Elekta Vectorview/TRIUX DACQ (data
acquisition). Extract and average epochs accordingly. Modify some
averaging parameters and get epochs.
"""
# Author: Jussi Nurminen (jnu@iki.fi)
#
# License: BSD (3-clause)


import mne
import os
from mne.datasets import testing, somato
from mne import AcqParserFIF

fname_raw = os.path.join(testing.data_path(), 'misc',
                         'test_elekta_3ch_raw.fif')

#fname_raw = os.path.join(somato.data_path(), 'MEG', 'somato',
#                         'sef_raw_sss.fif')

# fname_raw = '/Users/hus20664877/Dropbox/jn_multimodal01_raw.fif'


print(__doc__)

###############################################################################
# Read raw file and create averager instance
raw = mne.io.read_raw_fif(fname_raw)
eav = AcqParserFIF(raw.info)

###############################################################################
# Check DACQ defined averaging categories and other info
print(eav)

###############################################################################
# Extract epochs corresponding to a category
cond = eav.get_condition(raw, 'Test event 3')
epochs = mne.Epochs(raw, **cond)

###############################################################################
# Get epochs from all conditions, average, save to an evoked fiff file
evokeds = []
for cat in eav.categories:
    cond = eav.get_condition(raw, cat)
    # copy (supported) rejection parameters from DACQ settings
    epochs = mne.Epochs(raw, reject=eav.reject, flat=eav.flat, **cond)
    evoked = epochs.average()
    evoked.comment = cat['comment']
    evokeds.append(evoked)
fname_out = 'elekta_evokeds-ave.fif'
mne.write_evokeds(fname_out, evokeds)

###############################################################################
# Make a new category using existing one as a template, extract epochs
newcat = eav.categories[0].copy()
newcat['comment'] = 'My new category'
newcat['event'] = 1  # reference event
newcat['start'] = -.1  # epoch start rel. to ref. event (in seconds)
newcat['end'] = .5  # epoch end
newcat['reqevent'] = 2  # additional required event; 0 if none
newcat['reqwithin'] = 1.5  # req. event required within 1.5 sec of ref. event
newcat['reqwhen'] = 1  # required before (1) or after (2) ref. event
cond = eav.get_condition(raw, newcat)
epochs = mne.Epochs(raw, **cond)
