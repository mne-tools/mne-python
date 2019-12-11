"""Module dedicated to manipulation of channels.

Can be used for setting of sensor locations used for processing and plotting.
"""

from ..defaults import HEAD_SIZE_DEFAULT
from .layout import (Layout, make_eeg_layout, make_grid_layout, read_layout,
                     find_layout, generate_2d_layout)
from .montage import (DigMontage,
                      get_builtin_montages, make_dig_montage, read_dig_dat,
                      read_dig_egi, read_dig_captrack, read_dig_fif,
                      read_dig_polhemus_isotrak, read_polhemus_fastscan,
                      compute_dev_head_t, make_standard_montage,
                      read_custom_montage, read_dig_hpts,
                      compute_native_head_t)
from .channels import (equalize_channels, rename_channels, fix_mag_coil_types,
                       read_ch_connectivity, _get_ch_type,
                       find_ch_connectivity, make_1020_channel_selections)

__all__ = [
    # Data Structures
    'DigMontage', 'Layout',

    # Factory Methods
    'make_dig_montage', 'make_eeg_layout', 'make_grid_layout',
    'make_standard_montage',

    # Readers
    'read_ch_connectivity', 'read_dig_captrack', 'read_dig_dat',
    'read_dig_egi', 'read_dig_fif', 'read_dig_montage',
    'read_dig_polhemus_isotrak', 'read_layout', 'read_montage',
    'read_polhemus_fastscan', 'read_custom_montage', 'read_dig_hpts',

    # Helpers
    'rename_channels', 'make_1020_channel_selections',
    '_get_ch_type', 'equalize_channels', 'find_ch_connectivity', 'find_layout',
    'fix_mag_coil_types', 'generate_2d_layout', 'get_builtin_montages',

    # Other
    'compute_dev_head_t', 'compute_native_head_t',
]
