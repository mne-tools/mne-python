from __future__ import print_function

from nose.tools import assert_true
import sys
import inspect
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
    'mne.connectivity',
    'mne.datasets',
    'mne.datasets.megsim',
    'mne.datasets.sample',
    'mne.datasets.spm_face',
    'mne.decoding',
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
    'mne.io.write',  # always ignore these
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
            doc = docscrape.FunctionDoc(func)
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
        if not any(d in name_ for d in _docstring_ignores) and \
                'deprecation_wrapped' not in func.__code__.co_name:
            incorrect += [name_ + ' arg mismatch: ' + bad]
    else:
        for n1, n2 in zip(param_names, args):
            if n1 != n2:
                incorrect += [name_ + ' ' + n1 + ' != ' + n2]
    return incorrect


@requires_numpydoc
def test_docstring_parameters():
    """Test module docstring formatting"""
    from numpydoc import docscrape

    # skip modules that require mayavi if mayavi is not installed
    public_modules_ = public_modules[:]
    try:
        import mayavi  # noqa: F401
        public_modules_.append('mne.gui')
    except ImportError:
        pass

    incorrect = []
    for name in public_modules_:
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
        import mayavi  # noqa: F401
    except ImportError:
        ignore.extend('mne.gui.' + name for name in
                      ('_coreg_gui', '_fiducials_gui', '_file_traits', '_help',
                       '_kit2fiff_gui', '_marker_gui', '_viewer'))

    for importer, modname, ispkg in walk_packages(mne.__path__, prefix='mne.'):
        # because we don't import e.g. mne.tests w/mne
        if not ispkg and modname not in ignore:
            # mod = importlib.import_module(modname)  # not py26 compatible!
            try:
                __import__(modname)
            except Exception:  # can't import properly
                continue
            mod = sys.modules[modname]
            source = getsource(mod)
            assert_true('\t' not in source,
                        '"%s" has tabs, please remove them or add it to the'
                        'ignore list' % modname)


run_tests_if_main()
