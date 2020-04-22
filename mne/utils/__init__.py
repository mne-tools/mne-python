# # # WARNING # # #
# This list must also be updated in doc/_templates/autosummary/class.rst if it
# is changed here!
_doc_special_members = ('__contains__', '__getitem__', '__iter__', '__len__',
                        '__add__', '__sub__', '__mul__', '__div__',
                        '__neg__', '__hash__')

from ._bunch import Bunch, BunchConst, BunchConstNamed
from .check import (check_fname, check_version, check_random_state,
                    _check_fname, _check_subject, _check_pandas_installed,
                    _check_pandas_index_arguments, _check_mayavi_version,
                    _check_event_id, _check_ch_locs, _check_compensation_grade,
                    _check_if_nan, _is_numeric, _ensure_int,  _check_preload,
                    _validate_type, _check_info_inv, _check_pylsl_installed,
                    _check_channels_spatial_filter, _check_one_ch_type,
                    _check_rank, _check_option, _check_depth, _check_combine,
                    _check_path_like, _check_src_normal, _check_stc_units,
                    _check_pyqt5_version, _check_sphere, _check_time_format,
                    _check_freesurfer_home)
from .config import (set_config, get_config, get_config_path, set_cache_dir,
                     set_memmap_min_size, get_subjects_dir, _get_stim_channel,
                     sys_info, _get_extra_data_path, _get_root_dir,
                     _get_numpy_libs)
from .docs import (copy_function_doc_to_method_doc, copy_doc, linkcode_resolve,
                   open_docs, deprecated, fill_doc, copy_base_doc_to_subclass_doc)
from .fetching import _fetch_file, _url_to_local_path
from ._logging import (verbose, logger, set_log_level, set_log_file,
                       use_log_level, catch_logging, warn, filter_out_warnings,
                       ETSContext, wrapped_stdout, _get_call_line)
from .misc import (run_subprocess, _pl, _clean_names, pformat, _file_like,
                   _explain_exception, _get_argvalues, sizeof_fmt,
                   running_subprocess, _DefaultEventParser)
from .progressbar import ProgressBar
from ._testing import (run_tests_if_main, run_command_if_main,
                       requires_sklearn,
                       requires_version, requires_nibabel, requires_mayavi,
                       requires_good_network, requires_mne, requires_pandas,
                       requires_h5py, traits_test, requires_pysurfer,
                       ArgvSetter, SilenceStdout, has_freesurfer, has_mne_c,
                       _TempDir, has_nibabel, _import_mlab, buggy_mkl_svd,
                       requires_numpydoc, requires_vtk, requires_freesurfer,
                       requires_nitime, requires_dipy,
                       requires_neuromag2ft, requires_pylsl, assert_object_equal,
                       assert_and_remove_boundary_annot, _raw_annot,
                       assert_dig_allclose, assert_meg_snr, assert_snr,
                       assert_stcs_equal, modified_env)
from .numerics import (hashfunc, _compute_row_norms,
                       _reg_pinv, random_permutation, _reject_data_segments,
                       compute_corr, _get_inst_data, array_split_idx,
                       sum_squared, split_list, _gen_events, create_slices,
                       _time_mask, _freq_mask, grand_average, object_diff,
                       object_hash, object_size, _apply_scaling_cov,
                       _undo_scaling_cov, _apply_scaling_array,
                       _undo_scaling_array, _scaled_array, _replace_md5, _PCA,
                       _mask_to_onsets_offsets, _array_equal_nan,
                       _julian_to_cal, _cal_to_julian, _dt_to_julian,
                       _julian_to_dt, _dt_to_stamp, _stamp_to_dt,
                       _check_dt, _ReuseCycle)
from .mixin import (SizeMixin, GetEpochsMixin, _prepare_read_metadata,
                    _prepare_write_metadata, _FakeNoPandas, ShiftTimeMixin)
from .linalg import (_svd_lwork, _repeated_svd, sqrtm_sym,
                     dgesdd, dgemm, zgemm, dgemv, ddot, LinAlgError, eigh,
                     _pos_semidef_inv)
from .dataframe import (_set_pandas_dtype, _scale_dataframe_data,
                        _convert_times, _build_data_frame, _check_scaling_time)
