import sys
from .. import _BACKEND

sys.stderr.write('Using {} backend.\n'.format(_BACKEND))

if _BACKEND == 'mlab':
    from .mlab_backend import *
else:
    import warnings
    warnings.warn('_BACKEND should be "mlab". '
'{} was given.'.format(_BACKEND))
