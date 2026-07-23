# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import ast
import importlib
import inspect
import re
import types
import typing
from pathlib import Path
from pkgutil import walk_packages

import pytest

import mne
from mne.utils import _pl, _record_warnings
from mne.utils._typing import Color, FileLike

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

pyproject_path = Path(__file__).parents[2] / "pyproject.toml"
if not pyproject_path.is_file():
    pytest.skip(f"pyproject.toml not found: {pyproject_path}", allow_module_level=True)
try:
    import tomllib
except ModuleNotFoundError:
    # TODO VERSION: Remove this when Python 3.11+ is required
    pytest.skip("tomllib not available", allow_module_level=True)

pyproject = tomllib.loads(pyproject_path.read_text("utf-8"))
numpydoc_checks = pyproject["tool"]["numpydoc_validation"]["checks"]
assert numpydoc_checks[0] == "all"
error_ignores = set(numpydoc_checks[1:])

# The modules that ty type-checks strictly (see ``[tool.ty.src]`` in pyproject.toml);
# here we also enforce that their type hints agree with the rendered docstrings.
typed_modules = tuple(
    path.removesuffix(".py").replace("/", ".")
    for path in pyproject["tool"]["ty"]["src"]["include"]
)
assert typed_modules, "Could not find typed modules in [tool.ty.src] include"


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


# Use ``np.random.default_rng(seed)`` and its modern methods. The global RNG
# (``np.random.seed``/``np.random.randn``/...) makes tests order-dependent and
# flaky, and the legacy ``RandomState`` methods below don't exist on a
# ``Generator``, so calling them silently locks code to the old bit stream.
global_rng_ok = ("default_rng", "RandomState", "Generator", "mtrand")
legacy_rng_methods = {
    "randn": "standard_normal",
    "rand": "random",
    "random_sample": "random",
    "ranf": "random",
    "randint": "integers",
    "random_integers": "integers",
    "seed": "a local default_rng",
    "tomaxint": "integers",
}


def _is_np_random(node):
    """Return whether ``node`` is the ``np.random`` module attribute."""
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "random"
        and isinstance(node.value, ast.Name)
        and node.value.id in ("np", "numpy")
    )


def test_no_global_rng():
    """Test that we use local generators and the modern numpy RNG API."""
    root = pyproject_path.parent  # only available in a dev/editable checkout
    bad = []
    for sub in ("mne", "examples", "tutorials"):
        base = root / sub
        if not base.is_dir():  # e.g. examples/tutorials absent from a wheel
            continue
        for path in sorted(base.rglob("*.py")):
            rel = path.relative_to(root).as_posix()
            for node in ast.walk(ast.parse(path.read_text("utf-8"))):
                # 1. the global RNG: ``np.random.<attr>`` / ``numpy.random.<attr>``
                if (
                    isinstance(node, ast.Attribute)
                    and node.attr not in global_rng_ok
                    and _is_np_random(node.value)
                ):
                    bad.append(
                        f"{rel}:{node.lineno}: np.random.{node.attr} "
                        "(use a local np.random.default_rng)"
                    )
                # 2. legacy RandomState-only methods, e.g. ``rng.randn(...)``
                elif (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in legacy_rng_methods
                    and not _is_np_random(node.func.value)
                ):
                    want = legacy_rng_methods[node.func.attr]
                    bad.append(f"{rel}:{node.lineno}: .{node.func.attr}() (use {want})")
    if bad:
        raise AssertionError(
            f"{len(bad)} outdated numpy RNG use{_pl(bad)} found:\n" + "\n".join(bad)
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


def _documented_public_names():
    """Return the names documented in ``doc/api/*.rst`` autosummary blocks."""
    from sphinx.ext.autosummary.generate import find_autosummary_in_files

    api_dir = Path(__file__).parents[2] / "doc" / "api"
    files = [str(path) for path in sorted(api_dir.glob("*.rst"))]
    return {entry.name for entry in find_autosummary_in_files(files)}


def _resolve_dotted(name):
    """Resolve a documented dotted name (e.g. ``mne.io.read_raw_edf``) by dot-access."""
    assert name.startswith("mne."), name  # we should only document our own code
    obj = mne
    for part in name.split(".")[1:]:
        obj = getattr(obj, part)
    return obj


def _in_typed_module(obj):
    """Whether ``obj`` is defined in one of the strictly-typed modules."""
    name = inspect.getmodule(obj).__name__
    return any(name == m or name.startswith(f"{m}.") for m in typed_modules)


def _documented_callables():
    """Yield ``(callable, cls)`` for documented public API in typed modules."""
    seen = set()
    for dotted in sorted(_documented_public_names()):
        obj = _resolve_dotted(dotted)
        if not _in_typed_module(obj):
            continue
        if inspect.isclass(obj):
            yield obj, None  # constructor signature vs class docstring
            for mname, method in inspect.getmembers(obj, inspect.isfunction):
                if mname.startswith("_") or not _in_typed_module(method):
                    continue
                if method not in seen:
                    seen.add(method)
                    yield method, obj
        elif inspect.isfunction(obj) and obj not in seen:
            seen.add(obj)
            yield obj, None


def _annotation_to_str(ann):
    """Render a type annotation as a module-stripped string."""
    origin = typing.get_origin(ann)
    # unions and ``Literal["a", "b"]`` both flatten to their ``a | b`` members
    if origin in (typing.Union, types.UnionType, typing.Literal):
        return " | ".join(_annotation_to_str(a) for a in typing.get_args(ann))
    if origin is not None:  # e.g. list[Evoked], dict[str, int], tuple[int, ...]
        args = typing.get_args(ann)
        name = getattr(origin, "__name__", str(origin))
        if args:
            return f"{name}[{', '.join(_annotation_to_str(a) for a in args)}]"
        return name
    if isinstance(ann, str):  # unevaluated forward reference, e.g. "EpochsFIF"
        return ann
    if ann is type(None):
        return "None"
    return getattr(ann, "__name__", str(ann))


# numpydoc pseudo-types (informal type words) mapped to real MNE type aliases,
# so a documented ``color`` validates against a ``Color``-annotated parameter.
# Populated (below, once the helpers exist) with each alias's atom expansion.
_PSEUDO_ALIASES = {"color": Color, "color object": Color, "file-like": FileLike}
_PSEUDO_ALIAS_ATOMS: dict[str, set[str]] = {}


def _split_union(type_str):
    """Split on top-level ``|`` only, so ``tuple[int | None, str]`` stays whole."""
    parts, depth, current = [], 0, ""
    for char in type_str:
        if char in "[(":
            depth += 1
        elif char in "])":
            depth = max(depth - 1, 0)
        if char == "|" and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += char
    parts.append(current)
    return parts


def _type_atoms(type_str):
    """Reduce a numpydoc or annotation type string to a canonical set of atoms."""
    s = type_str.replace("``", "").replace("`", "")  # drop reST inline literals
    s = re.sub(r":\w+:", "", s)  # drop sphinx roles, e.g. :class:
    s = s.replace("~", "")  # drop the sphinx "abbreviate" marker
    # TODO: neither ``default X`` nor ``optional`` belongs in a numpydoc *type* --
    # MNE style puts the default in the parameter description prose instead. Drop
    # these three normalizations to find (and then fix) the docstrings doing it.
    s = re.sub(r"\s*\(\s*default[^)]*\)", "", s, flags=re.I)  # (default X)
    s = re.sub(r"\s*,\s*default[:=]?\s*[^,|]+", "", s, flags=re.I)  # , default X
    s = re.sub(r"\s*,?\s*optional\b", "", s, flags=re.I)  # , optional
    s = re.sub(r"\binstance of\b", "", s)
    s = re.sub(r",?\s*(?:of )?shape\s*\(?[^)|]*\)?", "", s)  # shape (n, m) suffixes
    s = re.sub(r"\btuple of length \d+\b", "tuple", s, flags=re.I)
    s = re.sub(  # list of X -> list ("X" may be hyphenated, e.g. "list of path-like")
        r"\b(list|tuple|dict|set) of [\w.-]+", r"\1", s, flags=re.I
    )
    s = re.sub(r"\barray(?:-?like)?\s+of\s+\w+", "array", s, flags=re.I)
    s = re.sub(r"\barray-?like\b", "array", s, flags=re.I)
    s = re.sub(  # textual Literal["a", "b"] (from string annotations) -> a | b
        r"\bLiteral\[([^\]]*)\]", lambda m: m.group(1).replace(",", " | "), s
    )
    s = re.sub(
        r"\{([^}]*)\}", lambda m: m.group(1).replace(",", " | "), s
    )  # {a, b} -> a | b
    s = s.replace(" or ", "|")
    atoms = set()
    for part in _split_union(s):
        part = re.sub(r"[\w\.]*\.(\w+)", r"\1", part.strip())  # drop module paths
        part = part.split("[", 1)[0]  # reduce generics to container: list[str] -> list
        part = part.strip(" .,;:'\"")  # drop stray surrounding punctuation/quotes
        if not part:
            continue
        low = part.lower()
        if len(low) > 4 and low.startswith("base"):
            low = low[
                4:
            ]  # MNE ``BaseEpochs``/``BaseRaw`` are documented ``Epochs``/``Raw``
        if low in _PSEUDO_ALIAS_ATOMS:
            atoms |= _PSEUDO_ALIAS_ATOMS[low]
        elif low in ("path-like", "pathlike", "path_like"):
            atoms |= {"path", "str"}
        elif low == "list-like":
            atoms |= {"list", "array"}
        elif "array" in low or low == "ndarray":
            atoms.add("array")
        elif low == "class":  # the ``class`` pseudo-type is a Python ``type`` object
            atoms.add("type")
        else:
            atoms.add(low)
    return atoms


def _is_type_like(atom):
    """Whether a normalized atom is a comparable type/value rather than free prose.

    A single token (``int``, ``'+'``, ``'mm/dd/yy'``) can be compared; anything
    with an internal space ("matplotlib colormap", "Raw object") is prose.
    """
    return bool(atom) and not re.search(r"\s", atom)


# Malformed numpydoc types (comma-unions, prose, parenthesized sub-unions) that
# can't be compared to annotations. The test fails on both unlisted prose and stale
# entries, so this can only shrink.
# TODO: whittle down by fixing the docstrings.
unparseable_docstring_types = {
    "Evoked instance, or list of Evoked instances",
    "None | colormap | (colormap, bool) | 'interactive'",
    "Raw object",
    "bool, str, or None (default None)",
    "instance of matplotlib Axes | None",
    "list of (int | str) | tuple of (int | str)",
    "list of (int | str) | tuple of (int | str) | ``'auto'``",
    "list of (n_epochs) list (of n_channels) | None",
    "list of Axes | dict of list of Axes | None",
    "list, or Raw instance",
    "matplotlib colormap | (colormap, bool) | 'interactive'",
    "str, {'power', 'amplitude'}",
}


_PSEUDO_ALIAS_ATOMS.update(
    {
        name: {a.lower() for a in _type_atoms(_annotation_to_str(alias))}
        for name, alias in _PSEUDO_ALIASES.items()
    }
)


def _check_type_hints(func, *, cls):
    """Compare type hints against docstring types; return (errors, allowlist hits)."""
    from numpydoc.docscrape import FunctionDoc

    incorrect, allowed = [], set()
    name = _func_name(func, cls)
    sig = inspect.signature(func)
    doc = FunctionDoc(func)
    # documented types, keyed by parameter name (and "return"); numpydoc keeps
    # combined entries like "fmin, fmax : float" under one key, so split them
    doc_types = {}
    for p in doc["Parameters"]:
        if p.name and p.type:
            for pname in p.name.split(", "):
                doc_types[pname.strip()] = p.type
    returns = [r.type for r in doc["Returns"] if r.type]
    if len(returns) == 1:  # skip multi-value (tuple) returns for simplicity
        doc_types["return"] = returns[0]

    checks = [
        (pname, param.annotation)
        for pname, param in sig.parameters.items()
        if pname not in ("self", "cls")
        and param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)
        and param.annotation is not inspect.Parameter.empty
    ]
    if sig.return_annotation is not inspect.Signature.empty:
        checks.append(("return", sig.return_annotation))

    for target, annotation in checks:
        if target not in doc_types:  # not documented (or not comparably); skip
            continue
        # ``Self`` describes the (sub)class, which docstrings spell out concretely
        if _annotation_to_str(annotation) == "Self":
            continue
        doc_atoms = {d.lower() for d in _type_atoms(doc_types[target])}
        # Prose types cannot be matched mechanically. They are docstring bugs, so
        # only the known (allowlisted) ones are tolerated; anything new is an error.
        if not doc_atoms or not all(_is_type_like(d) for d in doc_atoms):
            if doc_types[target] in unparseable_docstring_types:
                allowed.add(doc_types[target])
            else:
                incorrect.append(
                    f"{name} : {target} : docstring type {doc_types[target]!r} is not "
                    "machine-checkable; fix it to MNE style (``a | b | None``, "
                    "``instance of X``)"
                )
            continue
        ann_atoms = {a.lower() for a in _type_atoms(_annotation_to_str(annotation))}
        # The annotation must cover every documented type, but may be broader:
        # ty rejects ``= None``/``= ()`` defaults unless the annotation admits
        # them, so an accurate hint sometimes adds ``None``/``tuple`` that the
        # docstring omits. A missing documented atom, though, is a real mismatch.
        if not ann_atoms >= doc_atoms:
            incorrect.append(
                f"{name} : {target} : type hint "
                f"{sorted(ann_atoms)} does not cover docstring {sorted(doc_atoms)}"
            )
    return incorrect, allowed


@pytest.mark.slowtest
def test_type_hints_match_docstrings():
    """Test that type hints agree with numpydoc-rendered docstring types."""
    pytest.importorskip("numpydoc")
    pytest.importorskip("sphinx")
    pytest.importorskip("sklearn")

    incorrect, allowed = [], set()
    for func, cls in _documented_callables():
        errors, hits = _check_type_hints(func, cls=cls)
        incorrect.extend(errors)
        allowed |= hits

    incorrect = sorted(set(incorrect))
    if incorrect:
        raise AssertionError(
            f"{len(incorrect)} type hint / docstring mismatch{_pl(incorrect)} "
            f"found:\n" + "\n".join(incorrect)
        )
    # keep the allowlist honest: a fixed docstring must be removed from it
    stale = sorted(unparseable_docstring_types - allowed)
    if stale:
        raise AssertionError(
            f"{len(stale)} entr{'y is' if len(stale) == 1 else 'ies are'} no longer "
            "needed in unparseable_docstring_types; remove:\n" + "\n".join(stale)
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
