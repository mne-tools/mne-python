# # # WARNING # # #
# This list must also be updated in doc/_templates/class.rst if it is
# changed here!
_doc_special_members = ('__contains__', '__getitem__', '__iter__', '__len__',
                        '__add__', '__sub__', '__mul__', '__div__',
                        '__neg__', '__hash__')

from .testing import object_diff
from .testing import run_tests_if_main
from .testing import requires_sklearn
from .testing import requires_version
from .testing import requires_nibabel
from .testing import requires_mayavi
from .testing import requires_good_network
from .testing import requires_mne
from .testing import requires_pandas
from .testing import requires_h5py
from .testing import traits_test
from .testing import requires_pysurfer
from .testing import _get_call_line
from .testing import SilenceStdout
from .testing import has_freesurfer
from .testing import has_mne_c
from .testing import _TempDir
from .testing import has_nibabel
from .testing import _import_mlab

from .deprecation import deprecated

from ._logging import warn

from .misc import verbose, logger, run_subprocess
from .misc import set_log_level
from .misc import set_log_file
from .misc import use_log_level
from .misc import catch_logging

from .misc import set_config
from .misc import get_config
from .misc import get_config_path
from .misc import set_cache_dir
from .misc import set_memmap_min_size
from .misc import run_subprocess

from .misc import _pl
from .misc import sys_info
from .misc import open_docs
from .misc import get_subjects_dir
from .misc import _clean_names
from .misc import _get_stim_channel
from .misc import _Counter
from .misc import pformat
from .misc import _explain_exception
from .misc import _get_argvalues
from .misc import _get_extra_data_path
from .misc import copy_function_doc_to_method_doc

from .progressbar import ProgressBar

from .check import check_fname
from .check import check_version
from .check import check_random_state
from .check import _check_fname
from .check import _check_subject
from .check import _check_pandas_installed
from .check import _check_pandas_index_arguments
from .check import _check_mayavi_version
from .check import _check_event_id
from .check import _check_ch_locs
from .check import _check_compensation_grade
from .check import _check_if_nan
from .check import _check_type_picks
from .check import _is_numeric
from .check import _ensure_int
from .check import _check_preload
from .check import _validate_type

from .fetching import _fetch_file
from .fetching import _url_to_local_path

from .numerics import hashfunc
from .numerics import md5sum
from .numerics import estimate_rank
from .numerics import _compute_row_norms
from .numerics import _reg_pinv
from .numerics import random_permutation
from .numerics import _reject_data_segments
from .numerics import compute_corr
from .numerics import _get_inst_data
from .numerics import array_split_idx
from .numerics import sum_squared
from .numerics import split_list
from .numerics import _gen_events
from .numerics import create_slices
from .numerics import _time_mask
from .numerics import grand_average

from .mixin import sizeof_fmt
from .mixin import SizeMixin
from .mixin import GetEpochsMixin
from .mixin import _prepare_read_metadata
from .mixin import _prepare_write_metadata
