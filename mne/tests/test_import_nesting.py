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
                                and node.module != "viz.backends.renderer"
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

    ignores = (
        # File, statement, kind (omit line number because this can change)
        ("mne/utils/docs.py", "    import mne", "non-relative mne import"),
        (
            "mne/io/_read_raw.py",
            "    from . import read_raw_ant, read_raw_artemis123, read_raw_bdf, read_raw_boxy, read_raw_brainvision, read_raw_cnt, read_raw_ctf, read_raw_curry, read_raw_edf, read_raw_eeglab, read_raw_egi, read_raw_eximia, read_raw_eyelink, read_raw_fieldtrip, read_raw_fif, read_raw_fil, read_raw_gdf, read_raw_kit, read_raw_nedf, read_raw_nicolet, read_raw_nihon, read_raw_nirx, read_raw_nsx, read_raw_persyst, read_raw_snirf",  # noqa: E501
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
        (
            "mne/time_frequency/spectrum.py",
            "        from ..viz._mpl_figure import _line_figure, _split_picks_by_type",
            "hierarchy: must not nest viz",
        ),
    )
    root_dir = Path(mne.__file__).parent.resolve()
    all_errors = list()
    for si, submodule_name in enumerate(IMPORT_NESTING_ORDER):
        must_not_nest = IMPORT_NESTING_ORDER[:si]
        must_nest = IMPORT_NESTING_ORDER[si + 1 :]
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


# This test ensures that modules are lazily loaded by lazy_loader.

eager_import = os.getenv("EAGER_IMPORT", "")
run_script = """
import sys
import mne

out = set()

# check scipy (Numba imports it to check the version)
ok_scipy_submodules = {'version'}
scipy_submodules = set(x.split('.')[1] for x in sys.modules.keys()
                       if x.startswith('scipy.') and '__' not in x and
                       not x.split('.')[1].startswith('_')
                       and sys.modules[x] is not None)
bad = scipy_submodules - ok_scipy_submodules
if len(bad) > 0:
    out |= {'scipy submodules: %s' % list(bad)}

# check sklearn and others
for x in sys.modules.keys():
    for key in ('sklearn', 'pandas', 'pyvista', 'matplotlib',
                'dipy', 'nibabel', 'cupy', 'picard', 'pyvistaqt', 'pooch',
                'tqdm', 'jinja2'):
        if x.startswith(key):
            x = '.'.join(x.split('.')[:2])
            out |= {x}
if len(out) > 0:
    print('\\nFound un-nested import(s) for %s' % (sorted(out),), end='')
exit(len(out))

# but this should still work
mne.io.read_raw_fif
assert "scipy.signal" in sys.modules, "scipy.signal not in sys.modules"
"""


@pytest.mark.skipif(bool(eager_import), reason=f"EAGER_IMPORT={eager_import}")
def test_lazy_loading():
    """Test that module imports are properly nested."""
    stdout, stderr, code = run_subprocess(
        [sys.executable, "-c", run_script], return_code=True
    )
    assert code == 0, stdout + stderr
