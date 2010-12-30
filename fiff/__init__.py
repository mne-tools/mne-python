__version__ = '0.1.git'

from .constants import FIFF
from .open import fiff_open
from .evoked import read_evoked
from .cov import read_cov, write_cov, write_cov_file
from .raw import setup_read_raw, read_raw_segment, read_raw_segment_times
from .event import read_events
from .forward import read_forward_solution
from .stc import read_stc

