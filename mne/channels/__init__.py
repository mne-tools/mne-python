"""Module dedicated to manipulation of channels.

Can be used for setting of sensor locations used for processing and plotting.
"""

from .layout import (Layout, make_eeg_layout, make_grid_layout, read_layout,
                     find_layout, generate_2d_layout)
from .montage import read_montage, read_dig_montage, Montage, DigMontage

from .channels import (equalize_channels, rename_channels, fix_mag_coil_types,
                       read_ch_connectivity, _get_ch_type)
