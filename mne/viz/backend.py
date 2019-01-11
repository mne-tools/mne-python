import sys
from .utils import MNE_3D_BACKEND

sys.stderr.write('Using {} backend.\n'.format(MNE_3D_BACKEND))

if MNE_3D_BACKEND == 'mlab':
    from .mlab_backend import *
else:
    import warnings
    warnings.warn('MNE_3D_BACKEND should be "mlab" : '
                  '{} was given.'.format(MNE_3D_BACKEND))
