# -*- coding: utf-8 -*-
# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause

import importlib
import inspect
from inspect import getsource
import os.path as op
from pathlib import Path
from pkgutil import walk_packages
import re

import pytest

import mne
from mne.utils import requires_numpydoc, _pl, _record_warnings

public_modules = [
    # the list of modules users need to access for all functionality
    'mne',
    'mne.baseline',
    'mne.beamformer',
    'mne.channels',
    'mne.chpi',
    'mne.cov',
    'mne.cuda',
    'mne.datasets',
    'mne.datasets.brainstorm',
    'mne.datasets.hf_sef',
    'mne.datasets.sample',
    'mne.decoding',
    'mne.dipole',
    'mne.export',
    'mne.filter',
    'mne.forward',
    'mne.gui',
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
    'mne.fixes',
    'mne.io.write',
    'mne.io.meas_info.Info',
}
char_limit = 800  # XX eventually we should probably get this lower
tab_ignores = [
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
            any(re.match(d, name) for d in docstring_ignores))
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
    # Add a check that all public functions and methods that have "verbose"
    # set the default verbose=None
    if cls is None:
        mod_or_class = importlib.import_module('.'.join(name.split('.')[:-1]))
    else:
        mod_or_class = importlib.import_module('.'.join(name.split('.')[:-2]))
        mod_or_class = getattr(mod_or_class, cls.__name__.split('.')[-1])
    callable_ = getattr(mod_or_class, name.split('.')[-1])
    try:
        sig = inspect.signature(callable_)
    except ValueError as exc:
        msg = str(exc)
        # E   ValueError: no signature found for builtin type
        #     <class 'mne.forward.forward.Forward'>
        if inspect.isclass(callable_) and 'no signature found for buil' in msg:
            pass
        else:
            raise
    else:
        if 'verbose' in sig.parameters:
            verbose_default = sig.parameters['verbose'].default
            if verbose_default is not None:
                incorrect += [
                    f'{name} : verbose default is not None, '
                    f'got: {verbose_default}']
    return incorrect


@pytest.mark.slowtest
@requires_numpydoc
def test_docstring_parameters():
    """Test module docstring formatting."""
    from numpydoc import docscrape

    incorrect = []
    for name in public_modules:
        # Assert that by default we import all public names with `import mne`
        if name not in ('mne', 'mne.gui'):
            extra = name.split('.')[1]
            assert hasattr(mne, extra)
        with _record_warnings():  # traits warnings
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
                    'of type object' not in str(cls.__call__) and \
                    'of ABCMeta object' not in str(cls.__call__):
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
    for _, modname, ispkg in walk_packages(mne.__path__, prefix='mne.'):
        # because we don't import e.g. mne.tests w/mne
        if not ispkg and modname not in tab_ignores:
            try:
                mod = importlib.import_module(modname)
            except Exception:  # e.g., mne.export not having pybv
                continue
            source = getsource(mod)
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
    doc_dir = op.abspath(op.join(op.dirname(__file__), '..', '..', 'doc'))
    doc_file = op.join(doc_dir, 'python_reference.rst')
    if not op.isfile(doc_file):
        pytest.skip('Documentation file not found: %s' % doc_file)
    api_files = (
        'covariance', 'creating_from_arrays', 'datasets',
        'decoding', 'events', 'file_io', 'forward', 'inverse', 'logging',
        'most_used_classes', 'mri', 'preprocessing', 'reading_raw_data',
        'realtime', 'report', 'sensor_space', 'simulation', 'source_space',
        'statistics', 'time_frequency', 'visualization', 'export')
    known_names = list()
    for api_file in api_files:
        with open(op.join(doc_dir, f'{api_file}.rst'), 'rb') as fid:
            for line in fid:
                line = line.decode('utf-8')
                if not line.startswith('  '):  # at least two spaces
                    continue
                line = line.split()
                if len(line) == 1 and line[0] != ':':
                    known_names.append(line[0].split('.')[-1])
    known_names = set(known_names)

    missing = []
    for name in public_modules:
        with _record_warnings():  # traits warnings
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
                        not any(from_mod.startswith(x)
                                for x in documented_ignored_mods) and
                        name not in documented_ignored_names and
                        not hasattr(cf, '_deprecated_original')):
                    missing.append('%s (%s.%s)' % (name, from_mod, name))
    if len(missing) > 0:
        raise AssertionError('\n\nFound new public members missing from '
                             'doc/python_reference.rst:\n\n* ' +
                             '\n* '.join(sorted(set(missing))))


def test_docdict_order():
    """Test that docdict is alphabetical."""
    from mne.utils.docs import docdict

    # read the file as text, and get entries via regex
    docs_path = Path(__file__).parent.parent / 'utils' / 'docs.py'
    assert docs_path.is_file(), docs_path
    with open(docs_path, 'r', encoding='UTF-8') as fid:
        docs = fid.read()
    entries = re.findall(r'docdict\[["\'](.+)["\']\] = ', docs)
    # test length & uniqueness
    assert len(docdict) == len(entries)
    # test order
    assert sorted(entries) == entries
