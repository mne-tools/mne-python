# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import webbrowser
from pathlib import Path
from types import SimpleNamespace

import pytest

from mne import grade_to_tris, open_docs
from mne.utils import (
    catch_logging,
    copy_doc,
    copy_doc_static,
    copy_function_doc_to_method_doc,
    copy_function_doc_to_method_doc_static,
    deprecated,
    deprecated_alias,
    fill_doc_static,
    legacy,
    linkcode_resolve,
    verbose_static,
)


@pytest.mark.parametrize("obj", (grade_to_tris,))
def test_doc_filling(obj):
    """Test that docs are filled properly."""
    doc = obj.__doc__
    assert "verbose : " in doc


def test_deprecated_alias():
    """Test deprecated_alias."""

    def new_func():
        """Do something."""
        pass

    deprecated_alias("old_func", new_func)
    assert old_func  # noqa
    assert "has been deprecated in favor of new_func" in old_func.__doc__  # noqa
    assert "deprecated" not in new_func.__doc__


@deprecated("deprecated func")
def deprecated_func():
    """Do something."""
    pass


@legacy("replacement_func")
def legacy_func():
    """Do something."""
    pass


@deprecated("deprecated class")
class deprecated_class:
    def __init__(self):
        pass

    @deprecated("deprecated method")
    def bad(self):
        pass


@legacy("replacement_class")
class legacy_class:  # noqa D101
    def __init__(self):
        pass

    @legacy("replacement_method")
    def bad(self):  # noqa D102
        pass


@pytest.mark.parametrize(
    ("msg", "klass", "func"),
    (
        ("deprecated", deprecated_class, deprecated_func),
        ("legacy", legacy_class, legacy_func),
    ),
)
def test_deprecated_and_legacy(msg, func, klass):
    """Test deprecated and legacy decorators."""
    if msg == "deprecated":
        with pytest.warns(FutureWarning, match=f"{msg} class"):
            _klass = klass()
        with pytest.warns(FutureWarning, match=f"{msg} method"):
            _klass.bad()
        with pytest.warns(FutureWarning, match=f"{msg} func"):
            func()
    else:
        with catch_logging(verbose="info") as log:
            _klass = klass()
            _klass.bad()
            func()
        log = log.getvalue()
        for kind in ("class", "method", "func"):
            assert f"New code should use replacement_{kind}" in log
    assert msg.upper() in klass.__init__.__doc__
    assert msg.upper() in klass.bad.__doc__
    assert msg.upper() in _klass.bad.__doc__
    assert msg.upper() in func.__doc__


def test_static_doc_markers():
    """Test that the static docstring decorators leave docstrings untouched."""

    @verbose_static("picks_all")
    def func(verbose=None):
        """Do a thing.

        Parameters
        ----------
        verbose : bool | str | int | None
            Control verbosity.
        """
        from mne.utils import logger

        logger.info("static hello")

    # the docstring is exactly what was written (modulo compile-time dedenting)
    assert func.__doc__.splitlines()[0] == "Do a thing."
    assert "Control verbosity." in func.__doc__
    assert func._static_doc_keys == ("verbose", "picks_all")
    with catch_logging() as log:
        func(verbose=True)
    assert "static hello" in log.getvalue()
    with catch_logging() as log:
        func(verbose=False)
    assert log.getvalue() == ""

    @fill_doc_static("picks_all")
    def filled():
        """Unchanged %(picks_all)s."""

    assert filled.__doc__ == "Unchanged %(picks_all)s."
    assert filled._static_doc_keys == ("picks_all",)

    @copy_function_doc_to_method_doc_static("func:mne.viz.plot_raw")
    def copied():
        """Unchanged."""

    assert copied.__doc__ == "Unchanged."
    assert copied._static_doc_copy == "func:mne.viz.plot_raw"
    with pytest.raises(ValueError, match="must look like"):
        copy_doc_static("func:mne.viz.plot_raw")


def test_copy_doc():
    """Test decorator for copying docstrings."""

    class A:
        def m1():
            """Docstring for m1."""
            pass

    class B:
        def m1():
            pass

    class C(A):
        @copy_doc(A.m1)
        def m1():
            pass

    assert C.m1.__doc__ == "Docstring for m1."
    pytest.raises(ValueError, copy_doc(B.m1), C.m1)


def test_copy_function_doc_to_method_doc():
    """Test decorator for reusing function docstring as method docstrings."""

    def f1(obj, a, b, c):
        """Docstring for f1.

        Parameters
        ----------
        obj : object
            Some object. This description also has

            blank lines in it.
        a : int
            Parameter a
        b : int
            Parameter b
        """
        pass

    def f2(obj):
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

    def f3(obj):
        """Docstring for f3.

        Parameters
        ----------
        object : object
            Only one parameter
        """
        pass

    def f4(obj):
        """Docstring for f4."""
        pass

    def f5(obj):  # noqa: D410, D411, D414
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

    assert (
        A.method_f1.__doc__
        == """\
Docstring for f1.

Parameters
----------
a : int
    Parameter a
b : int
    Parameter b
"""
    )

    assert (
        A.method_f2.__doc__
        == """\
Docstring for f2.

Returns
-------
nothing.
method_f3 own docstring
"""
    )

    assert A.method_f3.__doc__ == "Docstring for f3.\n\n"
    pytest.raises(ValueError, copy_function_doc_to_method_doc(f5), A.method_f1)


def myfun(x):
    """Check url."""
    assert "mne.tools" in x


def test_open_docs():
    """Test doc launching."""
    old_tab = webbrowser.open_new_tab
    try:
        # monkey patch temporarily to prevent tabs from actually spawning
        webbrowser.open_new_tab = myfun
        open_docs()
        open_docs("tutorials", "dev")
        open_docs("examples", "stable")
        pytest.raises(ValueError, open_docs, "foo")
        pytest.raises(ValueError, open_docs, "api", "foo")
    finally:
        webbrowser.open_new_tab = old_tab


def test_linkcode_resolve():
    """Test linkcode resolving."""
    ex = "#L"
    url = linkcode_resolve("py", dict(module="mne", fullname="Epochs"))
    assert "/mne/epochs.py" + ex in url
    url = linkcode_resolve("py", dict(module="mne", fullname="compute_covariance"))
    assert "/mne/cov.py" + ex in url
    url = linkcode_resolve(
        "py", dict(module="mne", fullname="convert_forward_solution")
    )
    assert "/mne/forward/forward.py" + ex in url
    url = linkcode_resolve(
        "py", dict(module="mne", fullname="datasets.sample.data_path")
    )
    assert "/mne/datasets/sample/sample.py" + ex in url


def _load_hook():
    import importlib.util

    import mne

    path = Path(mne.__file__).parents[1] / "tools" / "hooks" / "check_static_docs.py"
    if not path.is_file():
        pytest.skip("not running from a source checkout")
    spec = importlib.util.spec_from_file_location("check_static_docs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_HOOK_DOCDICT = {
    "alpha": "\nalpha : int\n    The alpha. It is shared.\n",
    "notes_shared": "\nFirst shared paragraph.\nLine two.\n\nSecond paragraph.\n",
    "verbose": "\nverbose : bool | str | int | None\n    Control verbosity.\n",
}
_HOOK_MODULE = '''\
from mne.utils import fill_doc_static, verbose_static
from mne.utils import copy_function_doc_to_method_doc_static


@verbose_static()
def func(alpha, verbose=None):
    """Do a thing.

    Parameters
    ----------
    %(alpha)s
    %(verbose)s

    Notes
    -----
    %(notes_shared)s
    This line is specific to func.
    """


class Klass:
    @copy_function_doc_to_method_doc_static("func:mne.baseline.rescale")
    def rescale(self, times, baseline, mode="mean", copy=True, picks=None):
        pass
'''


def test_check_static_docs(tmp_path, monkeypatch):
    """Test the static docstring pre-commit hook."""
    hook = _load_hook()
    docs_py = tmp_path / "docs.py"
    docs_py.write_text(
        "docdict = {}\n"
        + "".join(f'docdict["{k}"] = """{v}"""\n' for k, v in _HOOK_DOCDICT.items())
    )
    monkeypatch.setattr(hook, "docdict", dict(_HOOK_DOCDICT))
    monkeypatch.setattr(hook, "DOCS_PY", docs_py)
    monkeypatch.setattr(hook, "_old_docdict", lambda: dict(_HOOK_DOCDICT))
    path = tmp_path / "mod.py"
    path.write_text(_HOOK_MODULE)

    # 1. migration: placeholders are expanded and keys added to the decorator
    assert hook.process_file(path, True, {}) == []
    assert hook.process_file(path, False, {}) == []  # now in sync
    source = path.read_text()
    assert '@verbose_static("alpha", "notes_shared")' in source
    assert "    alpha : int\n        The alpha. It is shared.\n" in source
    assert "    Second paragraph.\n    This line is specific to func.\n" in source
    # the copied docstring was inserted (first parameter dropped)
    assert '"""Rescale (baseline correct) data.' in source
    assert "    data : array" not in source and "    times : 1D array" in source

    # 2. forward sync: docdict changed, the docstring (and only the shared part)
    #    is updated
    hook.docdict["alpha"] = "\nalpha : int\n    The alpha. It changed.\n"
    hook.docdict["notes_shared"] = "\nFirst shared paragraph, edited.\n\nSecond.\n"
    errors = hook.process_file(path, False, {})
    assert len(errors) == 1 and "out of sync" in errors[0]
    assert hook.process_file(path, True, {}) == []
    source = path.read_text()
    assert "It changed." in source and "It is shared." not in source
    assert "    Second.\n    This line is specific to func.\n" in source

    # 3. reverse sync: docdict unchanged since HEAD but a docstring copy edited
    monkeypatch.setattr(hook, "_old_docdict", lambda: dict(hook.docdict))
    path.write_text(path.read_text().replace("Control verbosity.", "Be loud."))
    reverse = {}
    assert hook.process_file(path, False, reverse) == []
    assert reverse == {"verbose": ["verbose : bool | str | int | None", "    Be loud."]}
    written, errors = hook._write_docdict_entries(reverse)
    assert written == ["verbose"] and errors == []
    want = 'docdict["verbose"] = """\nverbose : bool | str | int | None\n    Be loud.'
    assert want + '\n"""' in docs_py.read_text()
    # a templated entry cannot be written back
    docs_py.write_text(
        docs_py.read_text().replace(
            'docdict["alpha"] = """', 'docdict["alpha"] = "" + """'
        )
    )
    _, errors = hook._write_docdict_entries({"alpha": ["alpha : int", "    x"]})
    assert len(errors) == 1 and "by hand" in errors[0]

    # 4. site-specific text after a shared block: the previous version of the
    #    docstring (git HEAD) tells the two apart even when the shared part
    #    changes length
    own = "    This line is specific to func."
    source = path.read_text()
    assert "    Second.\n" + own in source
    # pretend the current file is what git HEAD has
    monkeypatch.setattr(hook, "REPO", tmp_path)
    monkeypatch.setattr(
        hook.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=source)
    )
    hook._old_bodies.cache_clear()
    # (a) a line added right after the shared text is shared (pushed to docdict)
    path.write_text(source.replace("    Second.\n", "    Second.\n    Third.\n"))
    reverse = {}
    assert hook.process_file(path, False, reverse) == []
    want = ["First shared paragraph, edited.", "", "Second.", "Third."]
    assert reverse["notes_shared"] == want
    # (b) ... unless it starts with a blank line, the site-specific convention
    alpha = "    alpha : int\n        The alpha. It changed.\n"
    assert alpha in source
    path.write_text(source.replace(alpha, alpha + "\n        .. versionadded:: 1.0\n"))
    reverse = {}
    assert hook.process_file(path, False, reverse) == [] and reverse == {}
    path.write_text(source.replace(alpha, alpha + "        More alpha.\n"))
    reverse = {}
    assert hook.process_file(path, False, reverse) == []
    assert reverse == {
        "alpha": ["alpha : int", "    The alpha. It changed.", "    More alpha."]
    }
    # (c) a line removed from the shared text does not swallow the site's own line
    path.write_text(source.replace("    Second.\n", ""))
    reverse = {}
    assert hook.process_file(path, False, reverse) == []
    assert reverse["notes_shared"] == ["First shared paragraph, edited."]
    assert own in path.read_text()
    # (d) editing the site-specific text alone is not a shared edit
    path.write_text(source.replace(own, "    Specific, edited."))
    reverse = {}
    assert hook.process_file(path, False, reverse) == [] and reverse == {}
    # (e) editing both is refused rather than guessed
    path.write_text(source.replace("    Second.\n", "").replace(own, "    Both."))
    errors = hook.process_file(path, False, {})
    assert len(errors) == 1 and "both the shared text" in errors[0]
    # (f) forward sync keeps the site's own text when docdict grows
    path.write_text(source)
    snapshot = dict(hook.docdict)
    monkeypatch.setattr(hook, "_old_docdict", lambda: snapshot)
    hook.docdict["notes_shared"] = "\nFirst shared paragraph, edited.\n\nSecond.\nMore."
    assert hook.process_file(path, True, {}) == []
    assert "    Second.\n    More.\n" + own in path.read_text()

    # 5. the E501 suppression comment follows the need for it
    source = path.read_text()
    assert '"""  # noqa: E501' in source  # the copied rescale docstring is wide
    stale = source.replace(
        '    """\n\n\nclass Klass', '    """  # noqa: E501\n\n\nclass Klass'
    )
    assert stale != source
    path.write_text(stale)
    errors = hook.process_file(path, False, {})
    assert len(errors) == 1 and "no longer needs" in errors[0]
    assert hook.process_file(path, True, {}) == []
    assert path.read_text() == source

    # 6. a block whose anchor (first line) is gone is an error, not a silent pass
    path.write_text(path.read_text().replace("First shared paragraph, edited.", "?"))
    errors = hook.process_file(path, False, {})
    assert len(errors) == 1 and "could not find" in errors[0]
