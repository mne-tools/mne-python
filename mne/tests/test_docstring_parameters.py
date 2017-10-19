from __future__ import print_function

import inspect
import os.path as op
import re
import sys
from unittest import SkipTest
import warnings

from pkgutil import walk_packages
from inspect import getsource

import mne
from mne.utils import (run_tests_if_main, _doc_special_members,
                       requires_numpydoc)
from mne.fixes import _get_args

public_modules = [
    # the list of modules users need to access for all functionality
    'mne',
    'mne.beamformer',
    'mne.chpi',
    'mne.connectivity',
    'mne.datasets',
    'mne.datasets.brainstorm',
    'mne.datasets.hf_sef',
    'mne.datasets.megsim',
    'mne.datasets.sample',
    'mne.decoding',
    'mne.dipole',
    'mne.filter',
    'mne.inverse_sparse',
    'mne.io',
    'mne.io.kit',
    'mne.minimum_norm',
    'mne.preprocessing',
    'mne.realtime',
    'mne.report',
    'mne.simulation',
    'mne.source_estimate',
    'mne.source_space',
    'mne.stats',
    'mne.time_frequency',
    'mne.time_frequency.tfr',
    'mne.viz',
]


def get_name(func):
    parts = []
    module = inspect.getmodule(func)
    if module:
        parts.append(module.__name__)
    if hasattr(func, 'im_class'):
        parts.append(func.im_class.__name__)
    parts.append(func.__name__)
    return '.'.join(parts)


# functions to ignore args / docstring of
_docstring_ignores = [
    'mne.io.Info',  # Parameters
    'mne.io.write',  # always ignore these
    # Deprecations
    'mne.connectivity.effective.phase_slope_index',
    'mne.connectivity.spectral.spectral_connectivity',
    'mne.decoding.time_frequency.*__init__',  # TimeFrequency
    'mne.minimum_norm.time_frequency.source_induced_power',
    'mne.time_frequency.csd.*__init__',  # CrossSpectralDensity
    'mne.time_frequency.multitaper.tfr_array_multitaper',
    'mne.time_frequency.tfr.tfr_array_morlet',
    'mne.time_frequency.tfr.tfr_morlet',
    'mne.decoding.csp.*plot_.*',  # CSP and SPoc but on Py3k unbound methods
    'mne.decoding.csp.*plot_.*',  # are just functions so our naming is odd
    r'mne.*\.plot_topomap',
    r'mne.*\.to_data_frame',
    'mne.viz.topomap.plot_evoked_topomap',
]

_tab_ignores = [
]


def check_parameters_match(func, doc=None):
    """Helper to check docstring, returns list of incorrect results"""
    from numpydoc import docscrape
    incorrect = []
    name_ = get_name(func)
    if not name_.startswith('mne.') or name_.startswith('mne.externals'):
        return incorrect
    if inspect.isdatadescriptor(func):
        return incorrect
    args = _get_args(func)
    # drop self
    if len(args) > 0 and args[0] == 'self':
        args = args[1:]

    if doc is None:
        with warnings.catch_warnings(record=True) as w:
            try:
                doc = docscrape.FunctionDoc(func)
            except Exception as exp:
                incorrect += [name_ + ' parsing error: ' + str(exp)]
                return incorrect
        if len(w):
            raise RuntimeError('Error for %s:\n%s' % (name_, w[0]))
    # check set
    param_names = [name for name, _, _ in doc['Parameters']]
    # clean up some docscrape output:
    param_names = [name.split(':')[0].strip('` ') for name in param_names]
    param_names = [name for name in param_names if '*' not in name]
    if len(param_names) != len(args):
        bad = str(sorted(list(set(param_names) - set(args)) +
                         list(set(args) - set(param_names))))
        if not any(re.match(d, name_) for d in _docstring_ignores) and \
                'deprecation_wrapped' not in func.__code__.co_name:
            incorrect += [name_ + ' arg mismatch: ' + bad]
    else:
        for n1, n2 in zip(param_names, args):
            if n1 != n2:
                incorrect += [name_ + ' ' + n1 + ' != ' + n2]
    return incorrect


@requires_numpydoc
def test_docstring_parameters():
    """Test module docstring formatting."""
    from numpydoc import docscrape

    # skip modules that require mayavi if mayavi is not installed
    public_modules_ = public_modules[:]
    try:
        import mayavi  # noqa: F401 analysis:ignore
        public_modules_.append('mne.gui')
    except ImportError:
        pass

    incorrect = []
    for name in public_modules_:
        with warnings.catch_warnings(record=True):  # traits warnings
            module = __import__(name, globals())
        for submod in name.split('.')[1:]:
            module = getattr(module, submod)
        classes = inspect.getmembers(module, inspect.isclass)
        for cname, cls in classes:
            if cname.startswith('_') and cname not in _doc_special_members:
                continue
            with warnings.catch_warnings(record=True) as w:
                cdoc = docscrape.ClassDoc(cls)
            if len(w):
                raise RuntimeError('Error for __init__ of %s in %s:\n%s'
                                   % (cls, name, w[0]))
            if hasattr(cls, '__init__'):
                incorrect += check_parameters_match(cls.__init__, cdoc)
            for method_name in cdoc.methods:
                method = getattr(cls, method_name)
                incorrect += check_parameters_match(method)
            if hasattr(cls, '__call__'):
                incorrect += check_parameters_match(cls.__call__)
        functions = inspect.getmembers(module, inspect.isfunction)
        for fname, func in functions:
            if fname.startswith('_'):
                continue
            incorrect += check_parameters_match(func)
    msg = '\n' + '\n'.join(sorted(list(set(incorrect))))
    if len(incorrect) > 0:
        raise AssertionError(msg)


def test_tabs():
    """Test that there are no tabs in our source files"""
    # avoid importing modules that require mayavi if mayavi is not installed
    ignore = _tab_ignores[:]
    try:
        import mayavi  # noqa: F401 analysis:ignore
    except ImportError:
        ignore.extend('mne.gui.' + name for name in
                      ('_coreg_gui', '_fiducials_gui', '_file_traits', '_help',
                       '_kit2fiff_gui', '_marker_gui', '_viewer'))

    for importer, modname, ispkg in walk_packages(mne.__path__, prefix='mne.'):
        # because we don't import e.g. mne.tests w/mne
        if not ispkg and modname not in ignore:
            # mod = importlib.import_module(modname)  # not py26 compatible!
            try:
                with warnings.catch_warnings(record=True):  # traits
                    __import__(modname)
            except Exception:  # can't import properly
                continue
            mod = sys.modules[modname]
            try:
                source = getsource(mod)
            except IOError:  # user probably should have run "make clean"
                continue
            assert '\t' not in source, ('"%s" has tabs, please remove them '
                                        'or add it to the ignore list'
                                        % modname)


documented_ignored_mods = (
    'mne.cuda',
    'mne.io.write',
    'mne.utils',
    'mne.viz.utils',
)
documented_ignored_names = """
BaseEstimator
ContainsMixin
CrossSpectralDensity
FilterMixin
GeneralizationAcrossTime
RawFIF
TimeDecoding
TimeMixin
ToDataFrameMixin
TransformerMixin
UpdateChannelsMixin
adjust_axes
apply_lcmv
apply_lcmv_epochs
apply_lcmv_raw
apply_maxfilter
apply_trans
channel_type
check_n_jobs
combine_kit_markers
combine_tfr
combine_transforms
design_mne_c_filter
detrend
dir_tree_find
fast_cross_3d
fiff_open
find_outliers
find_source_space_hemi
find_tag
get_score_funcs
get_sosfiltfilt
get_version
invert_transform
is_power2
iter_topography
kit2fiff
label_src_vertno_sel
make_eeg_average_ref_proj
make_lcmv
make_projector
mesh_dist
mesh_edges
minimum_phase
next_fast_len
parallel_func
pick_channels_evoked
plot_epochs_psd
plot_epochs_psd_topomap
plot_raw_psd_topo
plot_source_spectrogram
prepare_inverse_operator
read_fiducials
read_tag
rescale
simulate_noise_evoked
source_estimate_quantification
whiten_evoked
write_fiducials
write_info
""".split('\n')


def test_documented():
    """Test that public functions and classes are documented."""
    # skip modules that require mayavi if mayavi is not installed
    public_modules_ = public_modules[:]
    try:
        import mayavi  # noqa: F401, analysis:ignore
        public_modules_.append('mne.gui')
    except ImportError:
        pass

    doc_file = op.abspath(op.join(op.dirname(__file__), '..', '..', 'doc',
                                  'python_reference.rst'))
    if not op.isfile(doc_file):
        raise SkipTest('Documentation file not found: %s' % doc_file)
    known_names = list()
    with open(doc_file, 'rb') as fid:
        for line in fid:
            line = line.decode('utf-8')
            if not line.startswith('  '):  # at least two spaces
                continue
            line = line.split()
            if len(line) == 1 and line[0] != ':':
                known_names.append(line[0].split('.')[-1])
    known_names = set(known_names)

    missing = []
    for name in public_modules_:
        with warnings.catch_warnings(record=True):  # traits warnings
            module = __import__(name, globals())
        for submod in name.split('.')[1:]:
            module = getattr(module, submod)
        classes = inspect.getmembers(module, inspect.isclass)
        functions = inspect.getmembers(module, inspect.isfunction)
        checks = list(classes) + list(functions)
        for name, cf in checks:
            if not name.startswith('_') and name not in known_names:
                from_mod = inspect.getmodule(cf).__name__
                if (from_mod.startswith('mne') and
                        not from_mod.startswith('mne.externals') and
                        from_mod not in documented_ignored_mods and
                        name not in documented_ignored_names):
                    missing.append('%s (%s.%s)' % (name, from_mod, name))
    if len(missing) > 0:
        raise AssertionError('\n\nFound new public members missing from '
                             'doc/python_reference.rst:\n\n* ' +
                             '\n* '.join(sorted(set(missing))))


run_tests_if_main()
