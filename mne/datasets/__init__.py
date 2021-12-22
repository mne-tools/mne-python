"""Functions for fetching remote datasets.

See :ref:`datasets` for more information.
"""

from . import fieldtrip_cmc
from . import brainstorm
from . import visual_92_categories
from . import kiloword
from . import eegbci
from . import hf_sef
from . import misc
from . import mtrf
from . import sample
from . import somato
from . import multimodal
from . import fnirs_motor
from . import opm
from . import spm_face
from . import testing
from . import _fake
from . import phantom_4dbti
from . import sleep_physionet
from . import limo
from . import refmeg_noise
from . import ssvep
from . import erp_core
from . import epilepsy_ecog
from ._fetch import fetch_dataset
from .utils import (_download_all_example_data, fetch_hcp_mmp_parcellation,
                    fetch_aparc_sub_parcellation, has_dataset)
from ._fsaverage.base import fetch_fsaverage
from ._infant.base import fetch_infant_template
from ._phantom.base import fetch_phantom

__all__ = [
    '_download_all_example_data', '_fake', 'brainstorm', 'eegbci',
    'fetch_aparc_sub_parcellation', 'fetch_fsaverage', 'fetch_infant_template',
    'fetch_hcp_mmp_parcellation', 'fieldtrip_cmc', 'hf_sef', 'kiloword',
    'misc', 'mtrf', 'multimodal', 'opm', 'phantom_4dbti', 'sample',
    'sleep_physionet', 'somato', 'spm_face', 'ssvep', 'testing',
    'visual_92_categories', 'limo', 'erp_core', 'epilepsy_ecog',
    'fetch_dataset', 'fetch_phantom', 'has_dataset', 'refmeg_noise',
    'fnirs_motor'
]
