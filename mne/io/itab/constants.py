"""ITAB constants"""

# Author: Vittorio Pizzella <vittorio.pizzella@unich.it>
#
# License: BSD (3-clause)

from ..constants import BunchConst


ITAB = BunchConst()


# Channel types
ITAB.ITABV_EEG_CH = 1
ITAB.ITABV_MAG_CH = 2
ITAB.ITABV_REF_EEG_CH = 4
ITAB.ITABV_REF_MAG_CH = 8
ITAB.ITABV_AUX_CH = 16
ITAB.ITABV_PARAM_CH =32
ITAB.ITABV_DIGIT_CH = 64
ITAB.ITABV_FLAG_CH = 128


# mhd file pointers
ITAB.MHD_TIME      =    656
ITAB.MHD_NREFCH    =  82780
ITAB.MHD_RAWHDTYPE =  85348
ITAB.MHD_CHINFO    =  85444
ITAB.MHD_SENSPOS   = 295364
ITAB.MHD_MARKER    = 295436
ITAB.MHD_BESTCHI   = 297232

ITAB.MHD_CH_SIZE  = 328
