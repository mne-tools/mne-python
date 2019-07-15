import sys
from subprocess import Popen, PIPE


run_script = """
import sys
import mne

out = []

# check scipy
ok_scipy_submodules = set(['scipy', 'numpy',  # these appear in old scipy
                           'fftpack', 'lib', 'linalg',
                           'misc', 'sparse', 'version'])
scipy_submodules = set(x.split('.')[1] for x in sys.modules.keys()
                       if x.startswith('scipy.') and '__' not in x and
                       not x.split('.')[1].startswith('_')
                       and sys.modules[x] is not None)
bad = scipy_submodules - ok_scipy_submodules
if len(bad) > 0:
    out.append('scipy submodules: %s' % list(bad))

# check sklearn and others
_sklearn = _pandas = _mayavi = _matplotlib = False
for x in sys.modules.keys():
    if x.startswith('sklearn') and not _sklearn:
        out.append('sklearn')
        _sklearn = True
    if x.startswith('pandas') and not _pandas:
        out.append('pandas')
        _pandas = True
    if x.startswith('mayavi') and not _mayavi:
        out.append('mayavi')
        _mayavi = True
    if x.startswith('matplotlib') and not _matplotlib:
        out.append('matplotlib')
        _matplotlib = True
if len(out) > 0:
    print('\\nFound un-nested imports for:\\n' + '\\n'.join(out), end='')
    exit(1)
"""


def test_module_nesting():
    """Test that module imports are properly nested."""
    proc = Popen([sys.executable, '-c', run_script], stdout=PIPE, stderr=PIPE)
    stdout, stderr = proc.communicate()
    stdout = stdout.decode('utf-8')
    stderr = stderr.decode('utf-8')
    assert not proc.returncode, stdout + stderr
