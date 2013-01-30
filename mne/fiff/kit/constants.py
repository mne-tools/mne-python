"""
Created on Jan 27, 2012

@author: teon

"""
from mne.fiff.constants import Bunch

KIT = Bunch()
KIT.BASIC_INFO = 16
KIT.INT = 4
KIT.DOUBLE = 8
KIT.STRING = 128
KIT.CALIB_FACTOR = 1.0
KIT.AMPLIFIER_INFO = 112
KIT.CHAN_SENS = 80
KIT.SAMPLE_INFO = 128
KIT.DATA_OFFSET = 144
KIT.nchan = 192
KIT.nmegchan = 157
KIT.nrefchan = 3
KIT.ntrigchan = 8
KIT.nmiscchan = (KIT.nchan - KIT.nmegchan - KIT.nrefchan - KIT.ntrigchan)
KIT.RANGE = 1.
