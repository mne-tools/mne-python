"""Functions for fetching remote datasets.

See :ref:`datasets` for more information.
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from ._fetch import fetch_dataset
from .utils import (
    _download_all_example_data,
    fetch_hcp_mmp_parcellation,
    fetch_aparc_sub_parcellation,
    has_dataset,
)
from ._fsaverage.base import fetch_fsaverage
from ._infant.base import fetch_infant_template
from ._phantom.base import fetch_phantom

__all__ = [
    "_download_all_example_data",
    "_fake",
    "brainstorm",
    "eegbci",
    "fetch_aparc_sub_parcellation",
    "fetch_fsaverage",
    "fetch_infant_template",
    "fetch_hcp_mmp_parcellation",
    "fieldtrip_cmc",
    "hf_sef",
    "kiloword",
    "misc",
    "mtrf",
    "multimodal",
    "opm",
    "phantom_4dbti",
    "sample",
    "sleep_physionet",
    "somato",
    "spm_face",
    "ssvep",
    "testing",
    "visual_92_categories",
    "limo",
    "erp_core",
    "epilepsy_ecog",
    "fetch_dataset",
    "fetch_phantom",
    "has_dataset",
    "refmeg_noise",
    "fnirs_motor",
    "eyelink",
]
