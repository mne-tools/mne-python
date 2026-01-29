# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import importlib
import inspect
import re
from pathlib import Path
from pkgutil import walk_packages

import pytest

import mne
from mne.utils import _pl, _record_warnings

public_modules = [
    # the list of modules users need to access for all functionality
    "mne",
    "mne.baseline",
    "mne.beamformer",
    "mne.channels",
    "mne.chpi",
    "mne.cov",
    "mne.cuda",
    "mne.datasets",
    "mne.datasets.brainstorm",
    "mne.datasets.hf_sef",
    "mne.datasets.sample",
    "mne.decoding",
    "mne.dipole",
    "mne.export",
    "mne.filter",
    "mne.forward",
    "mne.gui",
    "mne.inverse_sparse",
    "mne.io",
    "mne.io.kit",
    "mne.minimum_norm",
    "mne.preprocessing",
    "mne.report",
    "mne.simulation",
    "mne.source_estimate",
    "mne.source_space",
    "mne.surface",
    "mne.stats",
    "mne.time_frequency",
    "mne.time_frequency.tfr",
    "mne.viz",
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
    return ".".join(parts)


# functions to ignore args / docstring of
docstring_ignores = {
    "mne.fixes",
    "mne.io.meas_info.Info",
}
tab_ignores = [
    "mne.channels.tests.test_montage",
    "mne.io.curry.tests.test_curry",
]
error_ignores = {
    # These we do not live by:
    "GL01",  # Docstring should start in the line immediately after the quotes
    "GL02",  # Closing quotes on own line (doesn't work on Python 3.13 anyway)
    "EX01",
    "EX02",  # examples failed (we test them separately)
    "ES01",  # no extended summary
    "SA01",  # no see also
    "YD01",  # no yields section
    "SA04",  # no description in See Also
    "PR04",  # Parameter "shape (n_channels" has no type
    "RT02",  # The first line of the Returns section should contain only the type, unless multiple values are being returned  # noqa
    # XXX should also verify that | is used rather than , to separate params
    # XXX should maybe also restore the parameter-desc-length < 800 char check
}
error_ignores_specific = {  # specific instances to skip
    ("regress_artifact", "SS05"),  # "Regress" is actually imperative
}
subclass_name_ignores = (
    (
        dict,
        {
            "values",
            "setdefault",
            "popitems",
            "keys",
            "pop",
            "update",
            "copy",
            "popitem",
            "get",
            "items",
            "fromkeys",
            "clear",
        },
    ),
    (list, {"append", "count", "extend", "index", "insert", "pop", "remove", "sort"}),
)


def check_parameters_match(func, *, cls=None, where):
    """Check docstring, return list of incorrect results."""
    from numpydoc.validate import validate

    name = _func_name(func, cls)
    skip = not name.startswith("mne.") or any(
        re.match(d, name) for d in docstring_ignores
    )
    if skip:
        return list()
    if cls is not None:
        for subclass, ignores in subclass_name_ignores:
            if issubclass(cls, subclass) and name.split(".")[-1] in ignores:
                return list()
    incorrect = [
        f"{where} : {name} : {err[0]} : {err[1]}"
        for err in validate(name)["errors"]
        if err[0] not in error_ignores
        and (name.split(".")[-1], err[0]) not in error_ignores_specific
    ]
    # Add a check that all public functions and methods that have "verbose"
    # set the default verbose=None
    if cls is None:
        mod_or_class = importlib.import_module(".".join(name.split(".")[:-1]))
    else:
        mod_or_class = importlib.import_module(".".join(name.split(".")[:-2]))
        mod_or_class = getattr(mod_or_class, cls.__name__.split(".")[-1])
    callable_ = getattr(mod_or_class, name.split(".")[-1])
    try:
        sig = inspect.signature(callable_)
    except ValueError as exc:
        msg = str(exc)
        # E   ValueError: no signature found for builtin type
        #     <class 'mne.forward.forward.Forward'>
        if inspect.isclass(callable_) and "no signature found for builtin type" in msg:
            pass
        else:
            raise
    else:
        if "verbose" in sig.parameters:
            verbose_default = sig.parameters["verbose"].default
            if verbose_default is not None:
                incorrect += [
                    f"{name} : verbose default is not None, got: {verbose_default}"
                ]
    return incorrect


@pytest.mark.slowtest
def test_docstring_parameters():
    """Test module docstring formatting."""
    npd = pytest.importorskip("numpydoc")
    incorrect = []
    for name in public_modules:
        # Assert that by default we import all public names with `import mne`
        if name not in ("mne", "mne.gui"):
            extra = name.split(".")[1]
            assert hasattr(mne, extra)
        with _record_warnings():  # traits warnings
            module = __import__(name, globals())
        for submod in name.split(".")[1:]:
            module = getattr(module, submod)
        try:
            classes = inspect.getmembers(module, inspect.isclass)
        except ModuleNotFoundError as exc:  # e.g., mne.decoding but no sklearn
            if "'sklearn'" in str(exc):
                continue
            raise
        for cname, cls in classes:
            if cname.startswith("_"):
                continue
            incorrect += check_parameters_match(cls, where=name)
            cdoc = npd.docscrape.ClassDoc(cls)
            for method_name in cdoc.methods:
                method = getattr(cls, method_name)
                incorrect += check_parameters_match(method, cls=cls, where=name)
            if (
                hasattr(cls, "__call__")
                and "of type object" not in str(cls.__call__)
                and "of ABCMeta object" not in str(cls.__call__)
            ):
                incorrect += check_parameters_match(
                    cls.__call__,
                    cls=cls,
                    where=name,
                )
        functions = inspect.getmembers(module, inspect.isfunction)
        for fname, func in functions:
            if fname.startswith("_"):
                continue
            incorrect += check_parameters_match(func, where=name)
    incorrect = sorted(list(set(incorrect)))
    if len(incorrect) > 0:
        raise AssertionError(
            f"{len(incorrect)} error{_pl(incorrect)} found:\n" + "\n".join(incorrect)
        )


def test_tabs():
    """Test that there are no tabs in our source files."""
    for _, modname, ispkg in walk_packages(mne.__path__, prefix="mne."):
        # because we don't import e.g. mne.tests w/mne
        if not ispkg and modname not in tab_ignores:
            try:
                mod = importlib.import_module(modname)
            except Exception:  # e.g., mne.export not having pybv
                continue
            source = inspect.getsource(mod)
            assert "\t" not in source, (
                f'"{modname}" has tabs, please remove them or add it to the ignore list'
            )


documented_ignored_mods = (
    "mne.fixes",
    "mne.io.write",
    "mne.utils",
    "mne.viz.utils",
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
find_tag
get_score_funcs
get_version
invert_transform
is_fixed_orient
make_eeg_average_ref_proj
make_projector
mesh_dist
mesh_edges
next_fast_len
parallel_func
plot_epochs_psd
plot_epochs_psd_topomap
plot_raw_psd_topo
plot_source_spectrogram
prepare_inverse_operator
read_fiducials
rescale
setup_proj
source_estimate_quantification
tddr
whiten_evoked
write_fiducials
write_info
""".split("\n")


def test_documented():
    """Test that public functions and classes are documented."""
    doc_dir = (Path(__file__).parents[2] / "doc" / "api").absolute()
    doc_file = doc_dir / "python_reference.rst"
    if not doc_file.is_file():
        pytest.skip(f"Documentation file not found: {doc_file}")
    api_files = (
        "covariance",
        "creating_from_arrays",
        "datasets",
        "decoding",
        "events",
        "file_io",
        "forward",
        "inverse",
        "logging",
        "most_used_classes",
        "mri",
        "preprocessing",
        "reading_raw_data",
        "realtime",
        "report",
        "sensor_space",
        "simulation",
        "source_space",
        "statistics",
        "time_frequency",
        "visualization",
        "export",
    )
    known_names = list()
    for api_file in api_files:
        with open(doc_dir / f"{api_file}.rst", "rb") as fid:
            for line in fid:
                line = line.decode("utf-8")
                if not line.startswith("  "):  # at least two spaces
                    continue
                line = line.split()
                if len(line) == 1 and line[0] != ":":
                    known_names.append(line[0].split(".")[-1])
    known_names = set(known_names)

    missing = []
    for name in public_modules:
        with _record_warnings():  # traits warnings
            module = __import__(name, globals())
        for submod in name.split(".")[1:]:
            module = getattr(module, submod)
        try:
            classes = inspect.getmembers(module, inspect.isclass)
        except ModuleNotFoundError as exc:  # e.g., mne.decoding but no sklearn
            if "'sklearn'" in str(exc):
                continue
            raise
        functions = inspect.getmembers(module, inspect.isfunction)
        checks = list(classes) + list(functions)
        for this_name, cf in checks:
            if not this_name.startswith("_") and this_name not in known_names:
                from_mod = inspect.getmodule(cf).__name__
                if (
                    from_mod.startswith("mne")
                    and not any(from_mod.startswith(x) for x in documented_ignored_mods)
                    and this_name not in documented_ignored_names
                    and not hasattr(cf, "_deprecated_original")
                ):
                    missing.append(f"{name} : {from_mod}.{this_name}")
    missing = sorted(set(missing))
    if len(missing) > 0:
        raise AssertionError(
            f"{len(missing)} new public member{_pl(missing)} missing from "
            "doc/python_reference.rst:\n" + "\n".join(missing)
        )


def test_docdict_order():
    """Test that docdict is alphabetical."""
    from mne.utils.docs import docdict

    # read the file as text, and get entries via regex
    docs_path = Path(__file__).parents[1] / "utils" / "docs.py"
    assert docs_path.is_file(), docs_path
    with open(docs_path, encoding="UTF-8") as fid:
        docs = fid.read()
    entries = re.findall(r'docdict\[(?:\n    )?["\'](.+)["\']\n?\] = ', docs)
    # test length & uniqueness
    assert len(docdict) == len(entries)
    # test order
    assert sorted(entries) == entries
