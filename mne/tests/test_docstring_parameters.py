import inspect
from inspect import getsource
import os.path as op
from pkgutil import walk_packages
import re
import sys
from unittest import SkipTest

import pytest

import mne
from mne.utils import run_tests_if_main, requires_numpydoc, _pl

public_modules = [
    # the list of modules users need to access for all functionality
    'mne',
    'mne.baseline',
    'mne.beamformer',
    'mne.channels',
    'mne.chpi',
    'mne.connectivity',
    'mne.cov',
    'mne.cuda',
    'mne.datasets',
    'mne.datasets.brainstorm',
    'mne.datasets.hf_sef',
    'mne.datasets.sample',
    'mne.decoding',
    'mne.dipole',
    'mne.filter',
    'mne.forward',
    'mne.inverse_sparse',
    'mne.io',
    'mne.io.kit',
    'mne.minimum_norm',
    'mne.preprocessing',
    'mne.report',
    'mne.simulation',
    'mne.source_estimate',
    'mne.source_space',
    'mne.surface',
    'mne.stats',
    'mne.time_frequency',
    'mne.time_frequency.tfr',
    'mne.viz',
]


def _func_name(func, cls=None):
    """Get the name."""
    parts = []
    if cls is not None:
        module = inspect.getmodule(cls)
    else:
        module = inspect.getmodule(func)
    if module:
        parts.append(module.__name__)
    if cls is not None:
        parts.append(cls.__name__)
    parts.append(func.__name__)
    return '.'.join(parts)


# functions to ignore args / docstring of
docstring_ignores = {
    'mne.externals',
    'mne.fixes',
    'mne.io.write',
    'mne.io.meas_info.Info',
    'mne.utils.docs.deprecated',
}
char_limit = 800  # XX eventually we should probably get this lower
tab_ignores = [
    'mne.externals.tqdm._tqdm.__main__',
    'mne.externals.tqdm._tqdm.cli',
    'mne.channels.tests.test_montage',
    'mne.io.curry.tests.test_curry',
]
error_ignores = {
    # These we do not live by:
    'GL01',  # Docstring should start in the line immediately after the quotes
    'EX01', 'EX02',  # examples failed (we test them separately)
    'ES01',  # no extended summary
    'SA01',  # no see also
    'YD01',  # no yields section
    'SA04',  # no description in See Also
    'PR04',  # Parameter "shape (n_channels" has no type
    'RT02',  # The first line of the Returns section should contain only the type, unless multiple values are being returned  # noqa
    # XXX should also verify that | is used rather than , to separate params
    # XXX should maybe also restore the parameter-desc-length < 800 char check
}
error_ignores_specific = {  # specific instances to skip
    ('regress_artifact', 'SS05'),  # "Regress" is actually imperative
}
subclass_name_ignores = (
    (dict, {'values', 'setdefault', 'popitems', 'keys', 'pop', 'update',
            'copy', 'popitem', 'get', 'items', 'fromkeys', 'clear'}),
    (list, {'append', 'count', 'extend', 'index', 'insert', 'pop', 'remove',
            'sort'}),
    (mne.fixes.BaseEstimator, {'get_params', 'set_params', 'fit_transform'}),
)


def check_parameters_match(func, cls=None):
    """Check docstring, return list of incorrect results."""
    from numpydoc.validate import validate
    name = _func_name(func, cls)
    skip = (not name.startswith('mne.') or
            any(re.match(d, name) for d in docstring_ignores) or
            'deprecation_wrapped' in getattr(
                getattr(func, '__code__', None), 'co_name', ''))
    if skip:
        return list()
    if cls is not None:
        for subclass, ignores in subclass_name_ignores:
            if issubclass(cls, subclass) and name.split('.')[-1] in ignores:
                return list()
    incorrect = ['%s : %s : %s' % (name, err[0], err[1])
                 for err in validate(name)['errors']
                 if err[0] not in error_ignores and
                 (name.split('.')[-1], err[0]) not in error_ignores_specific]
    return incorrect


@pytest.mark.slowtest
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
        # Assert that by default we import all public names with `import mne`
        if name not in ('mne', 'mne.gui'):
            extra = name.split('.')[1]
            assert hasattr(mne, extra)
        with pytest.warns(None):  # traits warnings
            module = __import__(name, globals())
        for submod in name.split('.')[1:]:
            module = getattr(module, submod)
        classes = inspect.getmembers(module, inspect.isclass)
        for cname, cls in classes:
            if cname.startswith('_'):
                continue
            incorrect += check_parameters_match(cls)
            cdoc = docscrape.ClassDoc(cls)
            for method_name in cdoc.methods:
                method = getattr(cls, method_name)
                incorrect += check_parameters_match(method, cls=cls)
            if hasattr(cls, '__call__') and \
                    'of type object' not in str(cls.__call__):
                incorrect += check_parameters_match(cls.__call__, cls)
        functions = inspect.getmembers(module, inspect.isfunction)
        for fname, func in functions:
            if fname.startswith('_'):
                continue
            incorrect += check_parameters_match(func)
    incorrect = sorted(list(set(incorrect)))
    msg = '\n' + '\n'.join(incorrect)
    msg += '\n%d error%s' % (len(incorrect), _pl(incorrect))
    if len(incorrect) > 0:
        raise AssertionError(msg)


def test_tabs():
    """Test that there are no tabs in our source files."""
    # avoid importing modules that require mayavi if mayavi is not installed
    ignore = tab_ignores[:]
    try:
        import mayavi  # noqa: F401 analysis:ignore
    except ImportError:
        ignore.extend('mne.gui.' + name for name in
                      ('_coreg_gui', '_fiducials_gui', '_file_traits', '_help',
                       '_kit2fiff_gui', '_marker_gui', '_viewer'))

    for _, modname, ispkg in walk_packages(mne.__path__, prefix='mne.'):
        # because we don't import e.g. mne.tests w/mne
        if not ispkg and modname not in ignore:
            # mod = importlib.import_module(modname)  # not py26 compatible!
            try:
                with pytest.warns(None):
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
    'mne.fixes',
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
TimeMixin
ToDataFrameMixin
TransformerMixin
UpdateChannelsMixin
activate_proj
adjust_axes
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
find_source_space_hemi
find_tag
get_score_funcs
get_version
invert_transform
is_power2
is_fixed_orient
kit2fiff
label_src_vertno_sel
make_eeg_average_ref_proj
make_projector
mesh_dist
mesh_edges
next_fast_len
parallel_func
pick_channels_evoked
plot_epochs_psd
plot_epochs_psd_topomap
plot_raw_psd_topo
plot_source_spectrogram
prepare_inverse_operator
read_bad_channels
read_fiducials
read_tag
rescale
setup_proj
source_estimate_quantification
tddr
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
    except ImportError:
        pass
    else:
        public_modules_.append('mne.gui')

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
        with pytest.warns(None):  # traits warnings
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
                        not any(from_mod.startswith(x)
                                for x in documented_ignored_mods) and
                        name not in documented_ignored_names):
                    missing.append('%s (%s.%s)' % (name, from_mod, name))
    if len(missing) > 0:
        raise AssertionError('\n\nFound new public members missing from '
                             'doc/python_reference.rst:\n\n* ' +
                             '\n* '.join(sorted(set(missing))))


run_tests_if_main()
