"""Functions for fetching remote datasets.

See `datasets`_ for more information.
"""

from . import fieldtrip_cmc
from . import brainstorm
from . import visual_92_categories
from . import eegbci
from . import hf_sef
from . import megsim
from . import misc
from . import mtrf
from . import sample
from . import somato
from . import multimodal
from . import spm_face
from . import testing
from . import _fake
from .utils import _download_all_example_data, fetch_hcp_mmp_parcellation
