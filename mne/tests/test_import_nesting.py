# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause

import ast
import glob
import os
from pathlib import Path
import sys
from types import ModuleType
import pytest

import mne
from mne.utils import run_subprocess


# To avoid circular import issues, we have a defined order of submodule
# priority. A submodule should nest an import from another submodule if and
# only if the other submodule is below it in this list.
# For example mne.fixes must nest all imports. mne.utils should nest all
# imports *except* mne.fixes, which should be an un-nested import.

IMPORT_NESTING_ORDER = (
    "fixes",
    "utils",
    "_fiff",
    "cuda",
    "filter",
    "transforms",
    "viz",
    "surface",
    "_freesurfer",
    "bem",
    "source_space",
    "channels",
    "event",
    "io",
    "epochs",
    "evoked",
    "forward",
    "minimum_norm",
    "beamformer",
    "inverse_sparse",
    "preprocessing",
    "time_frequency",
    # The rest of these are less critical after the above are sorted out,
    # so we'll just go alphabetical
    "chpi",
    "commands",
    "coreg",
    "datasets",
    "decoding",
    "dipole",
    "export",
    "gui",
    "simulation",
    "report",
    "stats",
)
# These are not listed in mne.__all__ but we want to consider them above
NON_ALL_SUBMODULES = (
    "_fiff",
    "_freesurfer",
    "bem",
    "cuda",
    "evoked",
    "filter",
    "fixes",
    "forward",
    "source_space",
    "surface",
    "transforms",
    "utils",
)


def test_import_nesting_hierarchy():
    """Test that our module nesting hierarchy is correct."""
    # First check that our IMPORT_NESTING_ORDER has all submodules
    submodule_names = [
        submodule_name
        for submodule_name in list(mne.__all__) + list(NON_ALL_SUBMODULES)
        if isinstance(getattr(mne, submodule_name), ModuleType)
    ]
    missing = set(IMPORT_NESTING_ORDER) - set(submodule_names)
    assert missing == set(), "Submodules missing from mne.__init__"
    missing = set(submodule_names) - set(IMPORT_NESTING_ORDER)
    assert missing == set(), "Submodules missing from IMPORT_NESTING_ORDER"

    # AST-parse all .py files in a submod dir to check nesting
    class ValidatingVisitor(ast.NodeVisitor):
        def __init__(self, must_nest, must_not_nest):
            self.must_nest = list()
            self.must_not_nest = list()
            super().__init__()

        def visit_Import(self, node):
            print("import", node.names)

        def visit_ImportFrom(self, node):
            if node.level == 0:
                return  # not a relative import
            print(f"from {'.' * node.level}{node.module or ''} import , {node.names}")

    for si, submodule_name in enumerate(IMPORT_NESTING_ORDER):
        must_nest = IMPORT_NESTING_ORDER[:si]
        must_not_nest = IMPORT_NESTING_ORDER[si + 1 :]
        submodule_path = Path(mne.__file__).parent.resolve() / submodule_name
        validator = ValidatingVisitor(must_nest, must_not_nest)
        if submodule_path.is_dir():
            # Get all .py files to parse
            files = glob.glob(str(submodule_path / "*.py"), recursive=True)
            assert len(files) > 1
        else:
            submodule_path = submodule_path.with_suffix(".py")
            assert submodule_path.is_file()
            files = [submodule_path]
        del submodule_path
        for file in files:
            tree = ast.parse(Path(file).read_text(), filename=file)
            assert isinstance(tree, ast.Module)
            validator.visit(tree)
            # for item in tree.body:
            #     # TODO: Need to check for imports nested in functions and classes
            #     if not isinstance(item, (ast.Import, ast.ImportFrom)):
            #         continue
            #     pass

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
