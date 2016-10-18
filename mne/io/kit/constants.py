"""KIT constants."""

# Author: Teon Brooks <teon.brooks@gmail.com>
#
# License: BSD (3-clause)

from ..constants import Bunch


KIT = Bunch()

# byte values
KIT.SHORT = 2
KIT.INT = 4
KIT.DOUBLE = 8
KIT.STRING = 128

# pointer locations
KIT.AMPLIFIER_INFO = 112
KIT.BASIC_INFO = 16
KIT.CHAN_SENS = 80
KIT.RAW_OFFSET = 144
KIT.AVE_OFFSET = 160
KIT.SAMPLE_INFO = 128
KIT.MRK_INFO = 192
KIT.CHAN_LOC_OFFSET = 64

# parameters
KIT.VOLTAGE_RANGE = 5.
KIT.CALIB_FACTOR = 1.0  # mne_manual p.272
KIT.RANGE = 1.  # mne_manual p.272
KIT.UNIT_MUL = 0  # default is 0 mne_manual p.273

# gain: 0:x1, 1:x2, 2:x5, 3:x10, 4:x20, 5:x50, 6:x100, 7:x200
KIT.GAINS = [1, 2, 5, 10, 20, 50, 100, 200]
# BEF options: 0:THROUGH, 1:50Hz, 2:60Hz, 3:50Hz
KIT.BEFS = [0, 50, 60, 50]

# coreg constants
KIT.DIG_POINTS = 10000

# create system specific dicts
KIT_NY = Bunch(**KIT)
KIT_AD = Bunch(**KIT)

# NY-system channel information
KIT_NY.NCHAN = 192
KIT_NY.NMEGCHAN = 157
KIT_NY.NREFCHAN = 3
KIT_NY.NMISCCHAN = 32
KIT_NY.N_SENS = KIT_NY.NMEGCHAN + KIT_NY.NREFCHAN
# 12-bit A-to-D converter, one bit for signed integer. range +/- 2048
KIT_NY.DYNAMIC_RANGE = 2 ** 11
# amplifier information
KIT_NY.GAIN1_BIT = 11  # stored in Bit 11-12
KIT_NY.GAIN1_MASK = 2 ** 11 + 2 ** 12
KIT_NY.GAIN2_BIT = 0  # stored in Bit 0-2
KIT_NY.GAIN2_MASK = 2 ** 0 + 2 ** 1 + 2 ** 2  # (0x0007)
KIT_NY.GAIN3_BIT = None
KIT_NY.GAIN3_MASK = None
KIT_NY.HPF_BIT = 4  # stored in Bit 4-5
KIT_NY.HPF_MASK = 2 ** 4 + 2 ** 5
KIT_NY.LPF_BIT = 8  # stored in Bit 8-10
KIT_NY.LPF_MASK = 2 ** 8 + 2 ** 9 + 2 ** 10
KIT_NY.BEF_BIT = 14  # stored in Bit 14-15
KIT_NY.BEF_MASK = 2 ** 14 + 2 ** 15
# HPF options: 0:0, 1:1, 2:3
KIT_NY.HPFS = [0, 1, 3]
# LPF options: 0:10Hz, 1:20Hz, 2:50Hz, 3:100Hz, 4:200Hz, 5:500Hz,
#              6:1,000Hz, 7:2,000Hz
KIT_NY.LPFS = [10, 20, 50, 100, 200, 500, 1000, 2000]


# University of Maryland - system channel information
# Virtually the same as the NY-system except new ADC in July 2014
# 16-bit A-to-D converter, one bit for signed integer. range +/- 32768
KIT_UMD = KIT_NY
KIT_UMD_2014 = Bunch(**KIT_UMD)
KIT_UMD_2014.DYNAMIC_RANGE = 2 ** 15


# AD-system channel information
KIT_AD.NCHAN = 256
KIT_AD.NMEGCHAN = 208
KIT_AD.NREFCHAN = 16
KIT_AD.NMISCCHAN = 32
KIT_AD.N_SENS = KIT_AD.NMEGCHAN + KIT_AD.NREFCHAN
# 16-bit A-to-D converter, one bit for signed integer. range +/- 32768
KIT_AD.DYNAMIC_RANGE = 2 ** 15
# amplifier information
KIT_AD.GAIN1_BIT = 12  # stored in Bit 12-14
KIT_AD.GAIN1_MASK = 2 ** 12 + 2 ** 13 + 2 ** 14
KIT_AD.GAIN2_BIT = 28  # stored in Bit 28-30
KIT_AD.GAIN2_MASK = 2 ** 28 + 2 ** 29 + 2 ** 30
KIT_AD.GAIN3_BIT = 24  # stored in Bit 24-26
KIT_AD.GAIN3_MASK = 2 ** 24 + 2 ** 25 + 2 ** 26
KIT_AD.HPF_BIT = 8  # stored in Bit 8-10
KIT_AD.HPF_MASK = 2 ** 8 + 2 ** 9 + 2 ** 10
KIT_AD.LPF_BIT = 16  # stored in Bit 16-18
KIT_AD.LPF_MASK = 2 ** 16 + 2 ** 17 + 2 ** 18
KIT_AD.BEF_BIT = 0  # stored in Bit 0-1
KIT_AD.BEF_MASK = 2 ** 0 + 2 ** 1
# HPF options: 0:0Hz, 1:0.03Hz, 2:0.1Hz, 3:0.3Hz, 4:1Hz, 5:3Hz, 6:10Hz, 7:30Hz
KIT_AD.HPFS = [0, 0.03, 0.1, 0.3, 1, 3, 10, 30]
# LPF options: 0:10Hz, 1:20Hz, 2:50Hz, 3:100Hz, 4:200Hz, 5:500Hz,
#              6:1,000Hz, 7:10,000Hz
KIT_AD.LPFS = [10, 20, 50, 100, 200, 500, 1000, 10000]


# KIT recording system is encoded in the SQD file as integer:
KIT.SYSTEM_NYU_2008 = 32  # NYU-NY, July 7, 2008 -
KIT.SYSTEM_NYU_2009 = 33  # NYU-NY, January 24, 2009 -
KIT.SYSTEM_NYU_2010 = 34  # NYU-NY, January 22, 2010 -
KIT.SYSTEM_NYUAD_2011 = 440  # NYU-AD initial launch May 20, 2011 -
KIT.SYSTEM_NYUAD_2012 = 441  # NYU-AD more channels July 11, 2012 -
KIT.SYSTEM_NYUAD_2014 = 442  # NYU-AD move to NYUAD campus Nov 20, 2014 -
KIT.SYSTEM_UMD_2004 = 51  # UMD Marie Mount Hall, October 1, 2004 -
KIT.SYSTEM_UMD_2014_07 = 52  # UMD update to 16 bit ADC, July 4, 2014 -
KIT.SYSTEM_UMD_2014_12 = 53  # UMD December 4, 2014 -

KIT_CONSTANTS = {KIT.SYSTEM_NYU_2008: KIT_NY,
                 KIT.SYSTEM_NYU_2009: KIT_NY,
                 KIT.SYSTEM_NYU_2010: KIT_NY,
                 KIT.SYSTEM_NYUAD_2011: KIT_AD,
                 KIT.SYSTEM_NYUAD_2012: KIT_AD,
                 KIT.SYSTEM_NYUAD_2014: KIT_AD,
                 KIT.SYSTEM_UMD_2004: KIT_UMD,
                 KIT.SYSTEM_UMD_2014_07: KIT_UMD_2014,
                 KIT.SYSTEM_UMD_2014_12: KIT_UMD_2014}

KIT_LAYOUT = {KIT.SYSTEM_NYU_2008: 'KIT-157',
              KIT.SYSTEM_NYU_2009: 'KIT-157',
              KIT.SYSTEM_NYU_2010: 'KIT-157',
              KIT.SYSTEM_NYUAD_2011: 'KIT-AD',
              KIT.SYSTEM_NYUAD_2012: 'KIT-AD',
              KIT.SYSTEM_NYUAD_2014: 'KIT-AD',
              KIT.SYSTEM_UMD_2004: None,
              KIT.SYSTEM_UMD_2014_07: None,
              KIT.SYSTEM_UMD_2014_12: 'KIT-UMD-3'}

# Names stored along with ID in SQD files
SYSNAMES = {KIT.SYSTEM_NYU_2009: 'NYU 160ch System since Jan24 2009',
            KIT.SYSTEM_NYU_2010: 'NYU 160ch System since Jan24 2009',
            KIT.SYSTEM_NYUAD_2012: "New York University Abu Dhabi",
            KIT.SYSTEM_NYUAD_2014: "New York University Abu Dhabi",
            KIT.SYSTEM_UMD_2004: "University of Maryland",
            KIT.SYSTEM_UMD_2014_07: "University of Maryland",
            KIT.SYSTEM_UMD_2014_12: "University of Maryland"}
