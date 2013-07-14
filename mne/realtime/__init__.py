""" Module for realtime MEG data using mne_rt_server """

# Authors: Christoph Dinh <chdinh@nmr.mgh.harvard.edu>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Mainak Jas <mainak@neuro.hut.fi>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

from .client import RtClient
from .epochs import RtEpochs
from .classifier import Scaler, FilterEstimator, PSDEstimator, ConcatenateChannels
from .mockclient import MockRtClient
from .mixin import TransformerMixin
