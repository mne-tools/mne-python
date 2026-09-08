# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import ast
import glob
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

import mne
from mne.utils import _pl, logger, run_subprocess

# To avoid circular import issues, we have a defined order of submodule
# priority. A submodule should nest an import from another submodule if and
# only if the other submodule is below it in this list.
# For example mne.fixes must nest all imports. mne.utils should nest all
# imports *except* mne.fixes, which should be an un-nested import.

IMPORT_NESTING_ORDER = (
    "fixes",
    "defaults",
    "utils",
    "cuda",
    "_fiff",
    "filter",
    "transforms",
    "surface",
    "_freesurfer",
    "viz",
    "annotations",
    "bem",
    "source_space",
    "channels",
    "event",
    "time_frequency",
    "evoked",
    "epochs",
    "io",
    "forward",
    "minimum_norm",
    "dipole",
    "inverse_sparse",
    "beamformer",
    "decoding",
    "preprocessing",
    # The rest of these are less critical after the above are sorted out,
    # so we'll just go alphabetical
    "chpi",
    "coreg",
    "datasets",
    "export",
    "gui",
    "report",
    "simulation",
    "stats",
)
# These are not listed in mne.__all__ but we want to consider them above
NON_ALL_SUBMODULES = (
    "_fiff",
    "_freesurfer",
    "annotations",
    "bem",
    "cuda",
    "evoked",
    "filter",
    "fixes",
    "surface",
    "transforms",
    "utils",
)
IGNORE_SUBMODULES = ("commands",)  # historically these are always root level
# mne.viz pulls in matplotlib, so importing it must always be nested rather than
# done at module level, whatever the hierarchy above would otherwise say. Checking
# this directly means a stray import is reported with its file and line, rather
# than having to be traced back from "matplotlib got imported".
MUST_ALWAYS_NEST = ("viz",)
# ...except in the submodules that are themselves about plotting, where importing
# mne.viz at module level is the whole point
MUST_ALWAYS_NEST_EXEMPT = ("gui", "report")


def test_import_nesting_hierarchy():
    """Test that our module nesting hierarchy is correct."""
    # First check that our IMPORT_NESTING_ORDER has all submodules
    submodule_names = [
        submodule_name
        for submodule_name in list(mne.__all__) + list(NON_ALL_SUBMODULES)
        if isinstance(getattr(mne, submodule_name), ModuleType)
        and submodule_name not in IGNORE_SUBMODULES
    ]
    missing = set(IMPORT_NESTING_ORDER) - set(submodule_names)
    assert missing == set(), "Submodules missing from mne.__init__"
    missing = set(submodule_names) - set(IMPORT_NESTING_ORDER)
    assert missing == set(), "Submodules missing from IMPORT_NESTING_ORDER"

    # AST-parse all .py files in a submod dir to check nesting
    class _ValidatingVisitor(ast.NodeVisitor):
        def __init__(self, *, rel_path, must_nest, must_not_nest):
            self.level = rel_path.count("/")  # e.g., mne/surface.py will be 1
            self.must_nest = set(must_nest)
            self.must_not_nest = set(must_not_nest)
            self.errors = list()
            super().__init__()

        def generic_visit(self, node):
            if not isinstance(node, ast.Import | ast.ImportFrom):
                super().generic_visit(node)
                return
            stmt = " " * node.col_offset
            if isinstance(node, ast.Import):
                stmt += "import "
            else:
                stmt += f"from {'.' * node.level}{node.module or ''} import "
            stmt += ", ".join(n.name for n in node.names)

            # No "import mne.*"
            err = (node.lineno, stmt)
            logger.debug(f"  {node.lineno:}".ljust(6) + ":" + stmt)
            if any(n.name == "mne" or n.name.startswith("mne.") for n in node.names):
                self.errors.append(err + ("non-relative mne import",))
            if isinstance(node, ast.ImportFrom):  # from
                if node.level != 0:  # from .
                    # now we need to triage based on whether this is nested
                    if node.module is None:
                        self.errors.append(err + ("non-explicit relative import",))
                    elif node.level == self.level:
                        module_name = node.module.split(".")[0]
                        if node.col_offset:  # nested
                            if (
                                module_name in self.must_not_nest
                                and module_name not in MUST_ALWAYS_NEST
                            ):
                                self.errors.append(
                                    err + (f"hierarchy: must not nest {module_name}",)
                                )
                        else:  # non-nested
                            if module_name in self.must_nest:
                                self.errors.append(
                                    err + (f"hierarchy: must nest {module_name}",)
                                )
            super().generic_visit(node)

        def visit_If(self, node):
            # The capital "I" is intentional: ``ast.NodeVisitor`` dispatches on the
            # node class name (``ast.If``), so the method must be ``visit_If``.
            # Imports guarded by ``if TYPE_CHECKING:`` never execute at runtime,
            # so they are exempt from the import-nesting hierarchy.
            test = node.test
            if (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
                isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
            ):
                for child in node.orelse:  # only the (runtime) else branch
                    self.visit(child)
                return
            self.generic_visit(node)

    ignores = (
        # File, statement, kind (omit line number because this can change)
        ("mne/utils/docs.py", "    import mne", "non-relative mne import"),
        (
            "mne/io/_read_raw.py",
            "    from . import read_raw_ant, read_raw_artemis123, read_raw_bci2k, read_raw_bdf, read_raw_boxy, read_raw_brainvision, read_raw_cnt, read_raw_ctf, read_raw_curry, read_raw_edf, read_raw_eeglab, read_raw_egi, read_raw_eximia, read_raw_eyelink, read_raw_fieldtrip, read_raw_fif, read_raw_fil, read_raw_gdf, read_raw_kit, read_raw_mef, read_raw_nedf, read_raw_nicolet, read_raw_nihon, read_raw_nirx, read_raw_nsx, read_raw_persyst, read_raw_snirf",  # noqa: E501
            "non-explicit relative import",
        ),
        (
            "mne/datasets/utils.py",
            "    from . import eegbci, fetch_fsaverage, fetch_hcp_mmp_parcellation, fetch_infant_template, fetch_phantom, limo, sleep_physionet",  # noqa: E501
            "non-explicit relative import",
        ),
        (
            "mne/datasets/sleep_physionet/__init__.py",
            "from . import age, temazepam, _utils",
            "non-explicit relative import",
        ),
        (
            "mne/datasets/brainstorm/__init__.py",
            "from . import bst_raw, bst_resting, bst_auditory, bst_phantom_ctf, bst_phantom_elekta",  # noqa: E501
            "non-explicit relative import",
        ),
        # nested on purpose: mne.time_frequency.tfr imports mne.viz.topomap at module
        # level (for @copy_function_doc_to_method_doc), so the core containers only
        # touch it when a TFR is actually requested
        (
            "mne/evoked.py",
            "        from .time_frequency.tfr import AverageTFR",
            "hierarchy: must not nest time_frequency",
        ),
        (
            "mne/epochs.py",
            "        from .time_frequency.tfr import AverageTFR, EpochsTFR",
            "hierarchy: must not nest time_frequency",
        ),
        (
            "mne/epochs.py",
            "    from .time_frequency.tfr import EpochsTFR",
            "hierarchy: must not nest time_frequency",
        ),
        (
            "mne/io/base.py",
            "        from ..time_frequency.tfr import RawTFR",
            "hierarchy: must not nest time_frequency",
        ),
        (
            "mne/channels/_standard_montage_utils.py",
            "from . import __file__",
            "non-explicit relative import",
        ),
        (
            "mne/source_space/__init__.py",
            "from . import _source_space",
            "non-explicit relative import",
        ),
    )
    root_dir = Path(mne.__file__).parent.resolve()
    all_errors = list()
    for si, submodule_name in enumerate(IMPORT_NESTING_ORDER):
        must_not_nest = IMPORT_NESTING_ORDER[:si]
        must_nest = IMPORT_NESTING_ORDER[si + 1 :]
        if submodule_name not in MUST_ALWAYS_NEST_EXEMPT:
            must_nest = must_nest + MUST_ALWAYS_NEST
        submodule_path = root_dir / submodule_name
        if submodule_path.is_dir():
            # Get all .py files to parse
            files = glob.glob(str(submodule_path / "**" / "*.py"), recursive=True)
            assert len(files) > 1
        else:
            submodule_path = submodule_path.with_suffix(".py")
            assert submodule_path.is_file()
            files = [submodule_path]
        del submodule_path
        for file in files:
            file = Path(file)
            rel_path = "mne" / file.relative_to(root_dir)
            if rel_path.parent.stem == "tests":
                continue  # never look at tests/*.py
            validator = _ValidatingVisitor(
                rel_path=rel_path.as_posix(),
                must_nest=must_nest,
                must_not_nest=must_not_nest,
            )
            tree = ast.parse(file.read_text(encoding="utf-8"), filename=file)
            assert isinstance(tree, ast.Module)
            rel_path = rel_path.as_posix()  # str
            logger.debug(rel_path)
            validator.visit(tree)
            errors = [
                err for err in validator.errors if (rel_path,) + err[1:] not in ignores
            ]
            # Format these for easy copy-paste
            all_errors.extend(
                f"Line {line}:".ljust(11) + f'("{rel_path}", "{stmt}", "{kind}"),'
                for line, stmt, kind in errors
                if not stmt.endswith((". import __version__", " import renderer"))
            )
    # Print a reasonable number of lines
    n = len(all_errors)
    all_errors = all_errors[:30] + (
        [] if n <= 30 else [f"... {len(all_errors) - 30} more"]
    )
    if all_errors:
        raise AssertionError(f"{n} nesting error{_pl(n)}:\n" + "\n".join(all_errors))

    # scheme obeys the above order


# SciPy submodules that must only ever be imported inside a function. They are
# expensive relative to what MNE uses them for (a handful of call sites each), and
# leaving one at module level silently drags it onto the import path of whatever
# imports that module -- see the level-2 test below.
NESTED_ONLY = ("scipy.ndimage", "scipy.sparse", "scipy.spatial", "scipy.stats")


def test_nested_only_imports():
    """Test that expensive SciPy submodules are only imported inside functions."""
    roots = {module.split(".")[0] for module in NESTED_ONLY}
    leaves = {module.split(".")[1] for module in NESTED_ONLY}
    root_dir = Path(mne.__file__).parent.resolve()
    bad = list()
    for file in sorted(root_dir.rglob("*.py")):
        if file.parent.name == "tests":
            continue
        tree = ast.parse(file.read_text(encoding="utf-8"), filename=file)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Import | ast.ImportFrom):
                continue
            if node.col_offset:  # indented, i.e. nested inside a function
                continue
            if isinstance(node, ast.Import):
                names = [
                    n.name
                    for n in node.names
                    if any(
                        n.name == module or n.name.startswith(f"{module}.")
                        for module in NESTED_ONLY
                    )
                ]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                # catches "scipy.spatial", "scipy.spatial.distance", ...
                if any(
                    node.module == module or node.module.startswith(f"{module}.")
                    for module in NESTED_ONLY
                ):
                    names = [node.module]
                elif node.module in roots:
                    names = [
                        f"{node.module}.{n.name}"
                        for n in node.names
                        if n.name in leaves
                    ]
                else:
                    names = []
            else:
                names = []
            rel_path = ("mne" / file.relative_to(root_dir)).as_posix()
            bad.extend(f"{rel_path}:{node.lineno}: {name}" for name in names)
    n = len(bad)
    assert not bad, (
        f"{n} module-level import{_pl(n)} of a nested-only SciPy submodule:\n"
        + "\n".join(bad)
    )


# These tests ensure that modules are lazily loaded by lazy_loader, using an
# allowlist: rather than naming the packages that must *not* be imported, we name the
# ones that may be, and work out what those pull in themselves. Anything else is an
# error, so a newly added eager dependency fails loudly instead of silently slowing
# every ``import mne`` down.

eager_import = os.getenv("EAGER_IMPORT", "")

# Level 1: what plain ``import mne`` may pull in. Every addition here costs every
# MNE user interpreter-startup time, so weigh it carefully.
LEVEL_1_ALLOWED = ("decorator", "lazy_loader", "numpy", "packaging")

# Level 2: submodules that should import without dragging in the heavy optional
# machinery -- notably numba and matplotlib, which stay off this list on purpose.
# SciPy is allowed, but submodule by submodule. Grow both tuples as more of MNE gets
# cleaned up (mne.io and mne.channels are the obvious next candidates).
LEVEL_2_TARGETS = (
    "from mne.transforms import *",
    "from mne import Epochs",
    "from mne.io import BaseRaw",
    "from mne.channels import DigMontage",
)
LEVEL_2_ALLOWED = LEVEL_1_ALLOWED + (
    "scipy.constants",
    "scipy.linalg",
    "scipy.sparse",
    "scipy.spatial",
    "scipy.special",
)

# Report scipy per-submodule and everything else per top-level package
_SPLIT_PACKAGES = ("scipy",)

_imports_script = """\
import sys
{statements}
split = {split!r}
out = set()
for name, mod in list(sys.modules.items()):
    if mod is None:
        continue
    parts = name.split(".")
    top = parts[0]
    if not top or top in sys.stdlib_module_names or top.startswith("_"):
        continue
    out.add(".".join(parts[:2]) if top in split and len(parts) > 1 else top)
print(" ".join(sorted(out)))
"""


def _imported(*statements):
    """Get the non-stdlib modules loaded by running ``statements``."""
    script = _imports_script.format(
        statements="\n".join(statements), split=_SPLIT_PACKAGES
    )
    stdout, stderr, code = run_subprocess(
        [sys.executable, "-c", script], return_code=True
    )
    assert code == 0, stdout + stderr
    return set(stdout.split())


def _check_imports(targets, allowed, level):
    # An empty run captures whatever the environment itself loads (.pth files,
    # editable-install finders, namespace packages, ...) so it can be subtracted.
    baseline = _imported()
    ok = _imported(*(f"import {module}" for module in allowed)) | baseline | {"mne"}
    extra = _imported(*targets) - ok
    assert extra == set(), (
        f"Level-{level} import check: {'; '.join(targets)} eagerly imported "
        f"{len(extra)} disallowed module{_pl(extra)}: {sorted(extra)}. Nest the "
        f"import{_pl(extra)} inside the function(s) that need them, or -- if truly "
        f"required at import time -- add to LEVEL_{level}_ALLOWED."
    )


@pytest.mark.skipif(bool(eager_import), reason=f"EAGER_IMPORT={eager_import}")
def test_lazy_loading():
    """Test that ``import mne`` pulls in nothing beyond its allowed dependencies."""
    _check_imports(("import mne",), LEVEL_1_ALLOWED, 1)


@pytest.mark.skipif(bool(eager_import), reason=f"EAGER_IMPORT={eager_import}")
def test_lazy_loading_level_2():
    """Test that cleaned-up submodules stay free of numba, matplotlib, etc."""
    _check_imports(LEVEL_2_TARGETS, LEVEL_2_ALLOWED, 2)


@pytest.mark.skipif(bool(eager_import), reason=f"EAGER_IMPORT={eager_import}")
def test_lazy_loading_works():
    """Test that lazily loaded attributes still resolve."""
    script = (
        "import sys\n"
        "import mne\n"
        "assert 'mne.io' not in sys.modules, 'mne.io imported eagerly'\n"
        "assert callable(mne.io.read_raw_fif), 'read_raw_fif did not resolve'\n"
        "assert 'mne.io.fiff.raw' in sys.modules, 'reader module not imported'\n"
    )
    stdout, stderr, code = run_subprocess(
        [sys.executable, "-c", script], return_code=True
    )
    assert code == 0, stdout + stderr
