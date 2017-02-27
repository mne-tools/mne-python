"""Visualization routines."""

from .topomap import (plot_evoked_topomap, plot_projs_topomap,
                      plot_ica_components, plot_tfr_topomap, plot_topomap,
                      plot_epochs_psd_topomap, plot_layout)
from .topo import plot_topo_image_epochs, iter_topography
from .utils import (tight_layout, mne_analyze_colormap, compare_fiff,
                    ClickableImage, add_background_image, plot_sensors)
from ._3d import (plot_sparse_source_estimates, plot_source_estimates,
                  plot_trans, plot_evoked_field, plot_dipole_locations,
                  snapshot_brain_montage)
from .misc import (plot_cov, plot_bem, plot_events, plot_source_spectrogram,
                   _get_presser, plot_dipole_amplitudes, plot_ideal_filter,
                   plot_filter, adjust_axes)
from .evoked import (plot_evoked, plot_evoked_image, plot_evoked_white,
                     plot_snr_estimate, plot_evoked_topo,
                     plot_evoked_joint, plot_compare_evokeds)
from .circle import plot_connectivity_circle, circular_layout
from .epochs import (plot_drop_log, plot_epochs, plot_epochs_psd,
                     plot_epochs_image)
from .raw import plot_raw, plot_raw_psd, plot_raw_psd_topo
from .ica import (plot_ica_scores, plot_ica_sources, plot_ica_overlay,
                  _plot_sources_raw, _plot_sources_epochs, plot_ica_properties)
from .montage import plot_montage
from .decoding import plot_gat_matrix, plot_gat_times
