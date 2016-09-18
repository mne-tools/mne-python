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
from mne.datasets import multimodal
from mne import AcqParserFIF

fname_raw = os.path.join(multimodal.data_path(), 'multimodal_raw.fif')


print(__doc__)

###############################################################################
# Read raw file and create averager instance
raw = mne.io.read_raw_fif(fname_raw)
ap = AcqParserFIF(raw.info)

###############################################################################
# Check DACQ defined averaging categories and other info
print(ap)

###############################################################################
# Extract epochs corresponding to a category
cond = ap.get_condition(raw, 'Auditory right')
epochs = mne.Epochs(raw, **cond)
epochs.average().plot_topo()

###############################################################################
# Get epochs from all conditions, average, save to an evoked fiff file
evokeds = []
for cat in ap.categories:
    cond = ap.get_condition(raw, cat)
    # copy (supported) rejection parameters from DACQ settings
    epochs = mne.Epochs(raw, reject=ap.reject, flat=ap.flat, **cond)
    evoked = epochs.average()
    evoked.comment = cat['comment']
    evokeds.append(evoked)
# fname_out = 'elekta_evokeds-ave.fif'
# mne.write_evokeds(fname_out, evokeds)

###############################################################################
# Make a new category using existing one as a template, extract epochs
newcat = ap.categories[0].copy()
newcat['comment'] = 'SEF right, longer'
newcat['event'] = 5  # reference event
newcat['start'] = -.5  # epoch start rel. to ref. event (in seconds)
newcat['end'] = 1  # epoch end
newcat['reqevent'] = 0  # additional required event; 0 if none
newcat['reqwithin'] = 0.0  # req. event required within 1.5 sec of ref. event
newcat['reqwhen'] = 1  # required before (1) or after (2) ref. event
cond = ap.get_condition(raw, newcat)
epochs = mne.Epochs(raw, **cond)
epochs.average().plot()
