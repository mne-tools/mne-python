"""KIT constants"""

# Author: Teon Brooks <teon@nyu.edu>
#
# License: BSD (3-clause)

from ..constants import Bunch


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
KIT.MRK_INFO = 192

# parameters
KIT.VOLTAGE_RANGE = 5.
KIT.CALIB_FACTOR = 1.0  # mne_manual p.272
KIT.RANGE = 1.  # mne_manual p.272
KIT.UNIT_MUL = 0  # default is 0 mne_manual p.273

# gain: 0:x1, 1:x2, 2:x5, 3:x10, 4:x20, 5:x50, 6:x100, 7:x200
KIT.GAINS = [1, 2, 5, 10, 20, 50, 100, 200]

# coreg constants
KIT.DIG_POINTS = 10000

# create system specific dicts
KIT_NY = Bunch()
KIT_NY.update(KIT)
KIT_AD = Bunch()
KIT_AD.update(KIT)

# NYU-system channel information
KIT_NY.nchan = 192
KIT_NY.nmegchan = 157
KIT_NY.nrefchan = 3
KIT_NY.ntrigchan = 8
KIT_NY.nmiscchan = 24
KIT_NY.n_sens = KIT_NY.nmegchan + KIT_NY.nrefchan
# amplifier information
KIT_NY.GAIN1_BIT = 11  # stored in Bit 11-12
KIT_NY.GAIN1_MASK = 2 ** 11 + 2 ** 12
KIT_NY.GAIN2_BIT = 0  # stored in Bit 0-2
KIT_NY.GAIN2_MASK = 2 ** 0 + 2 ** 1 + 2 ** 2  # (0x0007)
KIT_NY.GAIN3_BIT = None
KIT_NY.GAIN3_MASK = None
# 12-bit A-to-D converter, one bit for signed integer. range +/- 2048
KIT_NY.DYNAMIC_RANGE = 2 ** 12 / 2

# AD-system channel information
KIT_AD.nchan = 256
KIT_AD.nmegchan = 208
KIT_AD.nrefchan = 16
KIT_AD.ntrigchan = 8
KIT_AD.nmiscchan = 24
KIT_AD.n_sens = KIT_AD.nmegchan + KIT_AD.nrefchan
# amplifier information
KIT_AD.GAIN1_BIT = 12  # stored in Bit 12-14
KIT_AD.GAIN1_MASK = 2 ** 12 + 2 ** 13 + 2 ** 14
KIT_AD.GAIN2_BIT = 28  # stored in Bit 28-30
KIT_AD.GAIN2_MASK = 2 ** 28 + 2 ** 29 + 2 ** 30
KIT_AD.GAIN3_BIT = 24  # stored in Bit 24-26
KIT_AD.GAIN3_MASK = 2 ** 24 + 2 ** 25 + 2 ** 26
# 16-bit A-to-D converter, one bit for signed integer. range +/- 32768
KIT_AD.DYNAMIC_RANGE = 2 ** 16 / 2
