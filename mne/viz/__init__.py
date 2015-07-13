"""Visualization routines
"""

from .topomap import (plot_evoked_topomap, plot_projs_topomap,
                      plot_ica_components, plot_tfr_topomap, plot_topomap,
                      plot_epochs_psd_topomap)
from .topo import (plot_topo, plot_topo_image_epochs,
                   iter_topography)
from .utils import (tight_layout, mne_analyze_colormap, compare_fiff,
                    ClickableImage, add_background_image)
from ._3d import (plot_sparse_source_estimates, plot_source_estimates,
                  plot_trans, plot_evoked_field, plot_dipole_locations)
from .misc import (plot_cov, plot_bem, plot_events, plot_source_spectrogram,
                   _get_presser, plot_dipole_amplitudes)
from .evoked import (plot_evoked, plot_evoked_image, plot_evoked_white,
                     plot_snr_estimate)
from .circle import plot_connectivity_circle, circular_layout
from .epochs import (plot_image_epochs, plot_drop_log, plot_epochs,
                     plot_epochs_trellis, _drop_log_stats, plot_epochs_psd)
from .raw import plot_raw, plot_raw_psd
from .ica import plot_ica_scores, plot_ica_sources, plot_ica_overlay
from .ica import _plot_sources_raw, _plot_sources_epochs
from .montage import plot_montage
from .decoding import plot_gat_matrix, plot_gat_times
