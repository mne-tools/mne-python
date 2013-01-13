"""
============================
Export Raw Objects to NiTime
============================

This script shows how to export raw files to the NiTime library
for further signal processing and data analysis.

"""

# Author: Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

print __doc__

import numpy as np
import mne

from mne.fiff import Raw
from mne.datasets import sample

data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'

###############################################################################
# get raw data
raw = Raw(raw_fname)

# set picks
picks = mne.fiff.pick_types(raw.info, meg=True, eeg=False, eog=False,
                            stim=False, exclude='bads')

# pick times relative to the onset of the MEG measurement.
start, stop = raw.time_as_index([100, 115], use_first_samp=False)

# export to nitime using a copy of the data
raw_ts = raw.to_nitime(start=start, stop=stop, picks=picks, copy=True)

###############################################################################
# explore some nitime timeseries features

# get start
print raw_ts.t0

# get duration
print raw_ts.duration

# get sample duration (sampling interval)
print raw_ts.sampling_interval

# get exported raw infor
print raw_ts.metadata.keys()

# index at certain time
print raw_ts.at(110.5)

# get channel names (attribute added during export)
print raw_ts.ch_names[:3]

###############################################################################
# investigate spectral density

import matplotlib.pylab as pl

import nitime.algorithms as tsa

ch_sel = raw_ts.ch_names.index('MEG 0122')

data_ch = raw_ts.data[ch_sel]

f, psd_mt, nu = tsa.multi_taper_psd(data_ch, Fs=raw_ts.sampling_rate,
                                    BW=1, adaptive=False, jackknife=False)

# Convert PSD to dB
psd_mt = 10 * np.log10(psd_mt)

pl.close('all')
pl.plot(f, psd_mt)
pl.xlabel('Frequency (Hz)')
pl.ylabel('Power Spectrald Density (db/Hz)')
pl.title('Multitaper Power Spectrum \n %s' % raw_ts.ch_names[ch_sel])
pl.show()
