"""Module dedicated to manipulation of channels.

Can be used for setting of sensor locations used for processing and plotting.
"""

from .layout import (Layout, make_eeg_layout, make_grid_layout, read_layout,
                     find_layout, generate_2d_layout)
from .montage import (read_montage, read_dig_montage, Montage, DigMontage,
                      get_builtin_montages, make_dig_montage,
                      read_dig_egi, read_dig_captrack, read_dig_fif)
from .channels import (equalize_channels, rename_channels, fix_mag_coil_types,
                       read_ch_connectivity, _get_ch_type,
                       find_ch_connectivity, make_1020_channel_selections)

from .._digitization._utils import _read_dig_points as read_polhemus_fastscan
