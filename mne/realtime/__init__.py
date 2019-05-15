"""Realtime MEG data processing with servers and clients."""

# Authors: Christoph Dinh <chdinh@nmr.mgh.harvard.edu>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Mainak Jas <mainakjas@gmail.com>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Teon Brooks <teon.brooks@gmail.com>
#
# License: BSD (3-clause)

from .client import RtClient
from .epochs import RtEpochs
from .lsl_client import LSLClient
from .mock_lsl_stream import MockLSLStream
from .mockclient import MockRtClient
from .fieldtrip_client import FieldTripClient
from .stim_server_client import StimServer, StimClient
