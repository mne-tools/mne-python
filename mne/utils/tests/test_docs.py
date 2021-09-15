import pytest

from mne import open_docs, grade_to_tris
from mne.epochs import add_channels_epochs
from mne.utils import (copy_function_doc_to_method_doc, copy_doc,
                       linkcode_resolve, deprecated, deprecated_alias)
import webbrowser


@pytest.mark.parametrize('obj', (grade_to_tris, add_channels_epochs))
def test_doc_filling(obj):
    """Test that docs are filled properly."""
    doc = obj.__doc__
    assert 'verbose : ' in doc
    if obj is add_channels_epochs:
        assert 'keyword-argument only. Defaults to True if' in doc


def test_deprecated_alias():
    """Test deprecated_alias."""
    def new_func():
        """Do something."""
        pass

    deprecated_alias('old_func', new_func)
    assert old_func  # noqa
    assert 'has been deprecated in favor of new_func' in old_func.__doc__  # noqa
    assert 'deprecated' not in new_func.__doc__


@deprecated('message')
def deprecated_func():
    """Do something."""
    pass


@deprecated('message')
class deprecated_class(object):

    def __init__(self):
        pass


def test_deprecated():
    """Test deprecated function."""
    pytest.deprecated_call(deprecated_func)
    pytest.deprecated_call(deprecated_class)


def test_copy_doc():
    """Test decorator for copying docstrings."""
    class A:
        def m1():
            """Docstring for m1."""
            pass

    class B:
        def m1():
            pass

    class C (A):
        @copy_doc(A.m1)
        def m1():
            pass

    assert C.m1.__doc__ == 'Docstring for m1.'
    pytest.raises(ValueError, copy_doc(B.m1), C.m1)


def test_copy_function_doc_to_method_doc():
    """Test decorator for re-using function docstring as method docstrings."""
    def f1(object, a, b, c):
        """Docstring for f1.

        Parameters
        ----------
        object : object
            Some object. This description also has

            blank lines in it.
        a : int
            Parameter a
        b : int
            Parameter b
        """
        pass

    def f2(object):
        """Docstring for f2.

        Parameters
        ----------
        object : object
            Only one parameter

        Returns
        -------
        nothing.
        """
        pass

    def f3(object):
        """Docstring for f3.

        Parameters
        ----------
        object : object
            Only one parameter
        """
        pass

    def f4(object):
        """Docstring for f4."""
        pass

    def f5(object):  # noqa: D410, D411, D414
        """Docstring for f5.

        Parameters
        ----------
        Returns
        -------
        nothing.
        """
        pass

    class A:
        @copy_function_doc_to_method_doc(f1)
        def method_f1(self, a, b, c):
            pass

        @copy_function_doc_to_method_doc(f2)
        def method_f2(self):
            "method_f3 own docstring"
            pass

        @copy_function_doc_to_method_doc(f3)
        def method_f3(self):
            pass

    assert A.method_f1.__doc__ == """Docstring for f1.

        Parameters
        ----------
        a : int
            Parameter a
        b : int
            Parameter b
        """

    assert A.method_f2.__doc__ == """Docstring for f2.

        Returns
        -------
        nothing.
        method_f3 own docstring"""

    assert A.method_f3.__doc__ == 'Docstring for f3.\n\n        '
    pytest.raises(ValueError, copy_function_doc_to_method_doc(f5), A.method_f1)


def myfun(x):
    """Check url."""
    assert 'mne.tools' in x


def test_open_docs():
    """Test doc launching."""
    old_tab = webbrowser.open_new_tab
    try:
        # monkey patch temporarily to prevent tabs from actually spawning
        webbrowser.open_new_tab = myfun
        open_docs()
        open_docs('tutorials', 'dev')
        open_docs('examples', 'stable')
        pytest.raises(ValueError, open_docs, 'foo')
        pytest.raises(ValueError, open_docs, 'api', 'foo')
    finally:
        webbrowser.open_new_tab = old_tab


def test_linkcode_resolve():
    """Test linkcode resolving."""
    ex = '#L'
    url = linkcode_resolve('py', dict(module='mne', fullname='Epochs'))
    assert '/mne/epochs.py' + ex in url
    url = linkcode_resolve('py', dict(module='mne',
                                      fullname='compute_covariance'))
    assert '/mne/cov.py' + ex in url
    url = linkcode_resolve('py', dict(module='mne',
                                      fullname='convert_forward_solution'))
    assert '/mne/forward/forward.py' + ex in url
    url = linkcode_resolve('py', dict(module='mne',
                                      fullname='datasets.sample.data_path'))
    assert '/mne/datasets/sample/sample.py' + ex in url
