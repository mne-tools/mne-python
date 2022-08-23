# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause

import sys
from mne.utils import run_subprocess


run_script = """
import sys
import mne

out = set()

# check scipy (Numba imports it to check the version)
ok_scipy_submodules = set(['scipy', 'numpy',  # these appear in old scipy
                           'version'])
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
    stdout, stderr, code = run_subprocess([sys.executable, '-c', run_script],
                                          return_code=True)
    assert code == 0, stdout + stderr
