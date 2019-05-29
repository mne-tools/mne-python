# -*- coding: utf-8 -*-
"""The documentation functions."""
# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import inspect
import os
import os.path as op
import sys
import warnings
import webbrowser

from .config import get_config
from ..externals.doccer import filldoc, unindent_dict
from .check import _check_option


##############################################################################
# Define our standard documentation entries

docdict = dict()

# Verbose
docdict['verbose'] = """
verbose : bool, str, int, or None
    If not None, override default verbose level (see :func:`mne.verbose`
    and :ref:`Logging documentation <tut_logging>` for more)."""
docdict['verbose_meth'] = (docdict['verbose'] + ' Defaults to self.verbose.')

# Picks
docdict['picks_header'] = 'picks : str | list | slice | None'
docdict['picks_base'] = docdict['picks_header'] + """
    Channels to include. Slices and lists of integers will be
    interpreted as channel indices. In lists, channel *type* strings
    (e.g., ``['meg', 'eeg']``) will pick channels of those
    types, channel *name* strings (e.g., ``['MEG0111', 'MEG2623']``
    will pick the given channels. Can also be the string values
    "all" to pick all channels, or "data" to pick data channels.
    None (default) will pick """
docdict['picks_all'] = docdict['picks_base'] + 'all channels.\n'
docdict['picks_all_data'] = docdict['picks_base'] + 'all data channels.\n'
docdict['picks_all_data_noref'] = (docdict['picks_all_data'][:-2] +
                                   '(excluding reference MEG channels).\n')
docdict['picks_good_data'] = docdict['picks_base'] + 'good data channels.\n'
docdict['picks_good_data_noref'] = (docdict['picks_good_data'][:-2] +
                                    '(excluding reference MEG channels).\n')
docdict['picks_nostr'] = """
picks : list | slice | None
    Channels to include. Slices and lists of integers will be
    interpreted as channel indices. None (default) will pick all channels.
"""

# Filtering
docdict['l_freq'] = """
l_freq : float | None
    For FIR filters, the lower pass-band edge; for IIR filters, the upper
    cutoff frequency. If None the data are only low-passed.
"""
docdict['h_freq'] = """
h_freq : float | None
    For FIR filters, the upper pass-band edge; for IIR filters, the upper
    cutoff frequency. If None the data are only low-passed.
"""
docdict['filter_length'] = """
filter_length : str | int
    Length of the FIR filter to use (if applicable):

    * **'auto' (default)**: The filter length is chosen based
      on the size of the transition regions (6.6 times the reciprocal
      of the shortest transition band for fir_window='hamming'
      and fir_design="firwin2", and half that for "firwin").
    * **str**: A human-readable time in
      units of "s" or "ms" (e.g., "10s" or "5500ms") will be
      converted to that number of samples if ``phase="zero"``, or
      the shortest power-of-two length at least that duration for
      ``phase="zero-double"``.
    * **int**: Specified length in samples. For fir_design="firwin",
      this should not be used.

"""
docdict['l_trans_bandwidth'] = """
l_trans_bandwidth : float | str
    Width of the transition band at the low cut-off frequency in Hz
    (high pass or cutoff 1 in bandpass). Can be "auto"
    (default) to use a multiple of ``l_freq``::

        min(max(l_freq * 0.25, 2), l_freq)

    Only used for ``method='fir'``.
"""
docdict['h_trans_bandwidth'] = """
h_trans_bandwidth : float | str
    Width of the transition band at the high cut-off frequency in Hz
    (low pass or cutoff 2 in bandpass). Can be "auto"
    (default in 0.14) to use a multiple of ``h_freq``::

        min(max(h_freq * 0.25, 2.), info['sfreq'] / 2. - h_freq)

    Only used for ``method='fir'``.
"""
docdict['phase'] = """
phase : str
    Phase of the filter, only used if ``method='fir'``.
    Symmetric linear-phase FIR filters are constructed, and if ``phase='zero'``
    (default), the delay of this filter is compensated for, making it
    non-causal. If ``phase=='zero-double'``,
    then this filter is applied twice, once forward, and once backward
    (also making it non-causal). If 'minimum', then a minimum-phase filter will
    be constricted and applied, which is causal but has weaker stop-band
    suppression.

    .. versionadded:: 0.13
"""
docdict['fir_design'] = """
fir_design : str
    Can be "firwin" (default) to use :func:`scipy.signal.firwin`,
    or "firwin2" to use :func:`scipy.signal.firwin2`. "firwin" uses
    a time-domain design technique that generally gives improved
    attenuation using fewer samples than "firwin2".

    .. versionadded:: 0.15
"""
docdict['fir_window'] = """
fir_window : str
    The window to use in FIR design, can be "hamming" (default),
    "hann" (default in 0.13), or "blackman".

    .. versionadded:: 0.15
"""
docdict['pad-fir'] = """
pad : str
    The type of padding to use. Supports all :func:`numpy.pad` ``mode``
    options. Can also be "reflect_limited", which pads with a
    reflected version of each vector mirrored on the first and last
    values of the vector, followed by zeros. Only used for ``method='fir'``.
"""
docdict['method-fir'] = """
method : str
    'fir' will use overlap-add FIR filtering, 'iir' will use IIR
    forward-backward filtering (via filtfilt).
"""
docdict['n_jobs-fir'] = """
n_jobs : int | str
    Number of jobs to run in parallel. Can be 'cuda' if ``cupy``
    is installed properly and method='fir'.
"""
docdict['n_jobs-cuda'] = """
n_jobs : int | str
    Number of jobs to run in parallel. Can be 'cuda' if ``cupy``
    is installed properly.
"""
docdict['iir_params'] = """
iir_params : dict | None
    Dictionary of parameters to use for IIR filtering.
    If iir_params is None and method="iir", 4th order Butterworth will be used.
    For more information, see :func:`mne.filter.construct_iir_filter`.
"""
docdict['npad'] = """
npad : int | str
    Amount to pad the start and end of the data.
    Can also be "auto" to use a padding that will result in
    a power-of-two size (can be much faster).
"""
docdict['window-resample'] = """
window : str | tuple
    Frequency-domain window to use in resampling.
    See :func:`scipy.signal.resample`.
"""

# Rank
docdict['rank'] = """
rank : None | dict | 'info' | 'full'
    This controls the rank computation that can be read from the
    measurement info or estimated from the data. See ``Notes``
    of :func:`mne.compute_rank` for details."""
docdict['rank_None'] = docdict['rank'] + 'The default is None.'
docdict['rank_info'] = docdict['rank'] + 'The default is "info".'

# Inverses
docdict['depth'] = """
depth : None | float | dict
    How to weight (or normalize) the forward using a depth prior.
    If float (default 0.8), it acts as the depth weighting exponent (``exp``)
    to use, which must be between 0 and 1. None is equivalent to 0, meaning
    no depth weighting is performed. It can also be a `dict` containing
    keyword arguments to pass to :func:`mne.forward.compute_depth_prior`
    (see docstring for details and defaults).
"""

# Forward
docdict['on_missing'] = """
on_missing : str
        Behavior when ``stc`` has vertices that are not in ``fwd``.
        Can be "ignore", "warn"", or "raise"."""

# Simulation
docdict['interp'] = """
interp : str
    Either 'hann', 'cos2' (default), 'linear', or 'zero', the type of
    forward-solution interpolation to use between forward solutions
    at different head positions.
"""
docdict['head_pos'] = """
head_pos : None | str | dict | tuple | array
    Name of the position estimates file. Should be in the format of
    the files produced by MaxFilter. If dict, keys should
    be the time points and entries should be 4x4 ``dev_head_t``
    matrices. If None, the original head position (from
    ``info['dev_head_t']``) will be used. If tuple, should have the
    same format as data returned by `head_pos_to_trans_rot_t`.
    If array, should be of the form returned by
    :func:`mne.chpi.read_head_pos`.
"""
docdict['n_jobs'] = """
n_jobs : int
    The number of jobs to run in parallel (default 1).
    Requires the joblib package.
"""

# Random state
docdict['random_state'] = """
random_state : None | int | instance of ~numpy.random.mtrand.RandomState
    If ``random_state`` is an :class:`int`, it will be used as a seed for
    :class:`~numpy.random.mtrand.RandomState`. If ``None``, the seed will be
    obtained from the operating system (see
    :class:`~numpy.random.mtrand.RandomState` for details). Default is
    ``None``.
"""

docdict['seed'] = """
seed : None | int | instance of ~numpy.random.mtrand.RandomState
    If ``seed`` is an :class:`int`, it will be used as a seed for
    :class:`~numpy.random.mtrand.RandomState`. If ``None``, the seed will be
    obtained from the operating system (see
    :class:`~numpy.random.mtrand.RandomState` for details). Default is
    ``None``.
"""

# Finalize
docdict = unindent_dict(docdict)
fill_doc = filldoc(docdict, unindent_params=False)


##############################################################################
# Utilities for docstring manipulation.

def copy_doc(source):
    """Copy the docstring from another function (decorator).

    The docstring of the source function is prepepended to the docstring of the
    function wrapped by this decorator.

    This is useful when inheriting from a class and overloading a method. This
    decorator can be used to copy the docstring of the original method.

    Parameters
    ----------
    source : function
        Function to copy the docstring from

    Returns
    -------
    wrapper : function
        The decorated function

    Examples
    --------
    >>> class A:
    ...     def m1():
    ...         '''Docstring for m1'''
    ...         pass
    >>> class B (A):
    ...     @copy_doc(A.m1)
    ...     def m1():
    ...         ''' this gets appended'''
    ...         pass
    >>> print(B.m1.__doc__)
    Docstring for m1 this gets appended
    """
    def wrapper(func):
        if source.__doc__ is None or len(source.__doc__) == 0:
            raise ValueError('Cannot copy docstring: docstring was empty.')
        doc = source.__doc__
        if func.__doc__ is not None:
            doc += func.__doc__
        func.__doc__ = doc
        return func
    return wrapper


def copy_function_doc_to_method_doc(source):
    """Use the docstring from a function as docstring for a method.

    The docstring of the source function is prepepended to the docstring of the
    function wrapped by this decorator. Additionally, the first parameter
    specified in the docstring of the source function is removed in the new
    docstring.

    This decorator is useful when implementing a method that just calls a
    function.  This pattern is prevalent in for example the plotting functions
    of MNE.

    Parameters
    ----------
    source : function
        Function to copy the docstring from

    Returns
    -------
    wrapper : function
        The decorated method

    Examples
    --------
    >>> def plot_function(object, a, b):
    ...     '''Docstring for plotting function.
    ...
    ...     Parameters
    ...     ----------
    ...     object : instance of object
    ...         The object to plot
    ...     a : int
    ...         Some parameter
    ...     b : int
    ...         Some parameter
    ...     '''
    ...     pass
    ...
    >>> class A:
    ...     @copy_function_doc_to_method_doc(plot_function)
    ...     def plot(self, a, b):
    ...         '''
    ...         Notes
    ...         -----
    ...         .. versionadded:: 0.13.0
    ...         '''
    ...         plot_function(self, a, b)
    >>> print(A.plot.__doc__)
    Docstring for plotting function.
    <BLANKLINE>
        Parameters
        ----------
        a : int
            Some parameter
        b : int
            Some parameter
    <BLANKLINE>
            Notes
            -----
            .. versionadded:: 0.13.0
    <BLANKLINE>

    Notes
    -----
    The parsing performed is very basic and will break easily on docstrings
    that are not formatted exactly according to the ``numpydoc`` standard.
    Always inspect the resulting docstring when using this decorator.
    """
    def wrapper(func):
        doc = source.__doc__.split('\n')
        if len(doc) == 1:
            doc = doc[0]
            if func.__doc__ is not None:
                doc += func.__doc__
            func.__doc__ = doc
            return func

        # Find parameter block
        for line, text in enumerate(doc[:-2]):
            if (text.strip() == 'Parameters' and
                    doc[line + 1].strip() == '----------'):
                parameter_block = line
                break
        else:
            # No parameter block found
            raise ValueError('Cannot copy function docstring: no parameter '
                             'block found. To simply copy the docstring, use '
                             'the @copy_doc decorator instead.')

        # Find first parameter
        for line, text in enumerate(doc[parameter_block:], parameter_block):
            if ':' in text:
                first_parameter = line
                parameter_indentation = len(text) - len(text.lstrip(' '))
                break
        else:
            raise ValueError('Cannot copy function docstring: no parameters '
                             'found. To simply copy the docstring, use the '
                             '@copy_doc decorator instead.')

        # Find end of first parameter
        for line, text in enumerate(doc[first_parameter + 1:],
                                    first_parameter + 1):
            # Ignore empty lines
            if len(text.strip()) == 0:
                continue

            line_indentation = len(text) - len(text.lstrip(' '))
            if line_indentation <= parameter_indentation:
                # Reach end of first parameter
                first_parameter_end = line

                # Of only one parameter is defined, remove the Parameters
                # heading as well
                if ':' not in text:
                    first_parameter = parameter_block

                break
        else:
            # End of docstring reached
            first_parameter_end = line
            first_parameter = parameter_block

        # Copy the docstring, but remove the first parameter
        doc = ('\n'.join(doc[:first_parameter]) + '\n' +
               '\n'.join(doc[first_parameter_end:]))
        if func.__doc__ is not None:
            doc += func.__doc__
        func.__doc__ = doc
        return func
    return wrapper


def copy_base_doc_to_subclass_doc(subclass):
    """Use the docstring from a parent class methods in derived class.

    The docstring of a parent class method is prepended to the
    docstring of the method of the class wrapped by this decorator.

    Parameters
    ----------
    subclass : wrapped class
        Class to copy the docstring to.

    Returns
    -------
    subclass : Derived class
        The decorated class with copied docstrings.
    """
    ancestors = subclass.mro()[1:-1]

    for source in ancestors:
        methodList = [method for method in dir(source)
                      if callable(getattr(source, method))]
        for method_name in methodList:
            # discard private methods
            if method_name[0] == '_':
                continue
            base_method = getattr(source, method_name)
            sub_method = getattr(subclass, method_name)
            if base_method is not None and sub_method is not None:
                doc = base_method.__doc__
                if sub_method.__doc__ is not None:
                    doc += '\n' + sub_method.__doc__
                sub_method.__doc__ = doc

    return subclass


def linkcode_resolve(domain, info):
    """Determine the URL corresponding to a Python object.

    Parameters
    ----------
    domain : str
        Only useful when 'py'.
    info : dict
        With keys "module" and "fullname".

    Returns
    -------
    url : str
        The code URL.

    Notes
    -----
    This has been adapted to deal with our "verbose" decorator.

    Adapted from SciPy (doc/source/conf.py).
    """
    import mne
    if domain != 'py':
        return None

    modname = info['module']
    fullname = info['fullname']

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split('.'):
        try:
            obj = getattr(obj, part)
        except Exception:
            return None
    # deal with our decorators properly
    while hasattr(obj, '__wrapped__'):
        obj = obj.__wrapped__

    try:
        fn = inspect.getsourcefile(obj)
    except Exception:
        fn = None
    if not fn:
        try:
            fn = inspect.getsourcefile(sys.modules[obj.__module__])
        except Exception:
            fn = None
    if not fn:
        return None
    fn = op.relpath(fn, start=op.dirname(mne.__file__))
    fn = '/'.join(op.normpath(fn).split(os.sep))  # in case on Windows

    try:
        source, lineno = inspect.getsourcelines(obj)
    except Exception:
        lineno = None

    if lineno:
        linespec = "#L%d-L%d" % (lineno, lineno + len(source) - 1)
    else:
        linespec = ""

    if 'dev' in mne.__version__:
        kind = 'master'
    else:
        kind = 'maint/%s' % ('.'.join(mne.__version__.split('.')[:2]))
    return "http://github.com/mne-tools/mne-python/blob/%s/mne/%s%s" % (  # noqa
       kind, fn, linespec)


def open_docs(kind=None, version=None):
    """Launch a new web browser tab with the MNE documentation.

    Parameters
    ----------
    kind : str | None
        Can be "api" (default), "tutorials", or "examples".
        The default can be changed by setting the configuration value
        MNE_DOCS_KIND.
    version : str | None
        Can be "stable" (default) or "dev".
        The default can be changed by setting the configuration value
        MNE_DOCS_VERSION.
    """
    if kind is None:
        kind = get_config('MNE_DOCS_KIND', 'api')
    help_dict = dict(api='python_reference.html', tutorials='tutorials.html',
                     examples='auto_examples/index.html')
    _check_option('kind', kind, sorted(help_dict.keys()))
    kind = help_dict[kind]
    if version is None:
        version = get_config('MNE_DOCS_VERSION', 'stable')
    _check_option('version', version, ['stable', 'dev'])
    webbrowser.open_new_tab('https://martinos.org/mne/%s/%s' % (version, kind))


# Following deprecated class copied from scikit-learn

# force show of DeprecationWarning even on python 2.7
warnings.filterwarnings('always', category=DeprecationWarning, module='mne')


class deprecated(object):
    """Mark a function or class as deprecated (decorator).

    Issue a warning when the function is called/the class is instantiated and
    adds a warning to the docstring.

    The optional extra argument will be appended to the deprecation message
    and the docstring. Note: to use this with the default value for extra, put
    in an empty of parentheses::

        >>> from mne.utils import deprecated
        >>> deprecated() # doctest: +ELLIPSIS
        <mne.utils.docs.deprecated object at ...>

        >>> @deprecated()
        ... def some_function(): pass


    Parameters
    ----------
    extra: string
        To be added to the deprecation messages.
    """

    # Adapted from http://wiki.python.org/moin/PythonDecoratorLibrary,
    # but with many changes.

    # scikit-learn will not import on all platforms b/c it can be
    # sklearn or scikits.learn, so a self-contained example is used above

    def __init__(self, extra=''):  # noqa: D102
        self.extra = extra

    def __call__(self, obj):  # noqa: D105
        """Call.

        Parameters
        ----------
        obj : object
            Object to call.
        """
        if isinstance(obj, type):
            return self._decorate_class(obj)
        else:
            return self._decorate_fun(obj)

    def _decorate_class(self, cls):
        msg = "Class %s is deprecated" % cls.__name__
        if self.extra:
            msg += "; %s" % self.extra

        # FIXME: we should probably reset __new__ for full generality
        init = cls.__init__

        def deprecation_wrapped(*args, **kwargs):
            warnings.warn(msg, category=DeprecationWarning)
            return init(*args, **kwargs)
        cls.__init__ = deprecation_wrapped

        deprecation_wrapped.__name__ = '__init__'
        deprecation_wrapped.__doc__ = self._update_doc(init.__doc__)
        deprecation_wrapped.deprecated_original = init

        return cls

    def _decorate_fun(self, fun):
        """Decorate function fun."""
        msg = "Function %s is deprecated" % fun.__name__
        if self.extra:
            msg += "; %s" % self.extra

        def deprecation_wrapped(*args, **kwargs):
            warnings.warn(msg, category=DeprecationWarning)
            return fun(*args, **kwargs)

        deprecation_wrapped.__name__ = fun.__name__
        deprecation_wrapped.__dict__ = fun.__dict__
        deprecation_wrapped.__doc__ = self._update_doc(fun.__doc__)

        return deprecation_wrapped

    def _update_doc(self, olddoc):
        newdoc = ".. warning:: DEPRECATED"
        if self.extra:
            newdoc = "%s: %s" % (newdoc, self.extra)
        if olddoc:
            # Get the spacing right to avoid sphinx warnings
            n_space = 4
            for li, line in enumerate(olddoc.split('\n')):
                if li > 0 and len(line.strip()):
                    n_space = len(line) - len(line.lstrip())
                    break
            newdoc = "%s\n\n%s%s" % (newdoc, ' ' * n_space, olddoc)
        return newdoc
