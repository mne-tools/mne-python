# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
# License: BSD Style.

from ...utils import verbose
from ...fixes import partial
from ..utils import has_dataset, _data_path, _doc


has_somato_data = partial(has_dataset, name='somato')


@verbose
def data_path(path=None, force_update=False, update_path=True, download=True,
              verbose=None):
    return _data_path(path=path, force_update=force_update,
                      update_path=update_path, name='somato',
                      download=download)

data_path.__doc__ = _doc.format(name='somato',
                                conf='MNE_DATASETS_SOMATO_PATH')
