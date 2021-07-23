# -*- coding: utf-8 -*-
# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause

from collections import OrderedDict

from ..fixes import _get_args
from ..utils import _check_fname, logger


def what(fname):
    """Try to determine the type of the FIF file.

    Parameters
    ----------
    fname : str
        The filename. Should end in .fif or .fif.gz.

    Returns
    -------
    what : str | None
        The type of the file. Will be 'unknown' if it could not be determined.

    Notes
    -----
    .. versionadded:: 0.19
    """
    from .fiff import read_raw_fif
    from ..epochs import read_epochs
    from ..evoked import read_evokeds
    from ..preprocessing import read_ica
    from ..forward import read_forward_solution
    from ..minimum_norm import read_inverse_operator
    from ..source_space import read_source_spaces
    from ..bem import read_bem_solution, read_bem_surfaces
    from ..cov import read_cov
    from ..transforms import read_trans
    from ..event import read_events
    from ..proj import read_proj
    from .meas_info import read_fiducials
    _check_fname(fname, overwrite='read', must_exist=True)
    checks = OrderedDict()
    checks['raw'] = read_raw_fif
    checks['ica'] = read_ica
    checks['epochs'] = read_epochs
    checks['evoked'] = read_evokeds
    checks['forward'] = read_forward_solution
    checks['inverse'] = read_inverse_operator
    checks['src'] = read_source_spaces
    checks['bem solution'] = read_bem_solution
    checks['bem surfaces'] = read_bem_surfaces
    checks['cov'] = read_cov
    checks['transform'] = read_trans
    checks['events'] = read_events
    checks['fiducials'] = read_fiducials
    checks['proj'] = read_proj
    for what, func in checks.items():
        args = _get_args(func)
        assert 'verbose' in args, func
        kwargs = dict(verbose='error')
        if 'preload' in args:
            kwargs['preload'] = False
        try:
            func(fname, **kwargs)
        except Exception as exp:
            logger.debug('Not %s: %s' % (what, exp))
        else:
            return what
    return 'unknown'
