# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause

import sys
from mne.utils import run_subprocess

# To keep `import mne` time down, we nest most imports, including those of
# required dependencies (e.g., pooch, tqdm, matplotlib). SciPy submodules
# (esp. linalg) can take a while to load, so these imports must be nested, too.
# NumPy is the exception -- it can be imported directly at the top of files.
# N.B. that jinja2 is imported directly in mne/html_templates, so all imports
# of mne.html_templates must be nested to achieve the jinja2 import being
# nested during import of mne.
#
# This test ensures that we don't accidentally un-nest any of these imports.

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
"""


def test_module_nesting():
    """Test that module imports are properly nested."""
    stdout, stderr, code = run_subprocess(
        [sys.executable, "-c", run_script], return_code=True
    )
    assert code == 0, stdout + stderr
