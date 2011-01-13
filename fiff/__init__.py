__version__ = '0.1.git'

from .constants import FIFF
from .open import fiff_open
from .evoked import read_evoked, write_evoked
from .cov import read_cov, write_cov, write_cov_file
from .raw import setup_read_raw, read_raw_segment, read_raw_segment_times, \
                 start_writing_raw, write_raw_buffer, finish_writing_raw
from .event import read_events, write_events
from .forward import read_forward_solution
from .stc import read_stc, write_stc
from .bem_surfaces import read_bem_surfaces
from .inverse import read_inverse_operator
from .pick import pick_types
from .meas_info import get_current_comp
from .epochs import read_epochs

