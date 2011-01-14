__version__ = '0.1.git'

from .cov import read_cov, write_cov, write_cov_file
from .event import read_events, write_events
from .forward import read_forward_solution
from .stc import read_stc, write_stc
from .bem_surfaces import read_bem_surfaces
from .inverse import read_inverse_operator, compute_inverse
from .epochs import read_epochs
import fiff
