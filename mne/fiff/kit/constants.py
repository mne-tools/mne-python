"""KIT constants"""

# Author: Teon Brooks <teon@nyu.edu>
#
# License: BSD (3-clause)

from mne.fiff.constants import Bunch

KIT = Bunch()

# byte values
KIT.INT = 4
KIT.DOUBLE = 8
KIT.STRING = 128

# pointer locations
KIT.AMPLIFIER_INFO = 112
KIT.BASIC_INFO = 16
KIT.CHAN_SENS = 80
KIT.DATA_OFFSET = 144
KIT.SAMPLE_INFO = 128

# system channel information
KIT.nchan = 192
KIT.nmegchan = 157
KIT.nrefchan = 3
KIT.ntrigchan = 8
KIT.nmiscchan = 24
KIT.n_sens = KIT.nmegchan + KIT.nrefchan

# parameters
KIT.DYNAMIC_RANGE = 2 ** 12 / 2  # signed integer. range +/- 2048
KIT.VOLTAGE_RANGE = 5.
KIT.CALIB_FACTOR = 1.0  # mne_manual p.272
KIT.RANGE = 1.  # mne_manual p.272
KIT.UNIT_MUL = 0  # default is 0 mne_manual p.273

# amplifier information
KIT.input_gain_bit = 11  # stored in Bit-11 to 12
KIT.input_gain_mask = 6144  # (0x1800)
KIT.output_gain_bit = 0  # stored in Bit-0 to 2
KIT.output_gain_mask = 7  # (0x0007)
# input_gain: 0:x1, 1:x2, 2:x5, 3:x10
KIT.input_gains = [1, 2, 5, 10]
# output_gain: 0:x1, 1:x2, 2:x5, 3:x10, 4:x20, 5:x50, 6:x100, 7:x200
KIT.output_gains = [1, 2, 5, 10, 20, 50, 100, 200]
