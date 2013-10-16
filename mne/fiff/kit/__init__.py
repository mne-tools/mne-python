"""KIT module for conversion to FIF"""

# Author: Teon Brooks <teon@nyu.edu>
#
# License: BSD (3-clause)

from .kit import read_raw_kit
from .coreg import read_elp, read_hsp, read_mrk, write_hsp, write_mrk
from . import kit
from . import coreg
from . import constants
