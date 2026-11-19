"""
.. _ex-psf-empirical:

===================================================
Plot point-spread functions (PSFs) with added noise
===================================================

Visualise the point-spread of a vertex in a sLORETA source estimate when we assume the
data to be noisy.
"""
# Authors: Olaf Hauk <olaf.hauk@mrc-cbu.cam.ac.uk>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import mne
from mne.datasets import sample
from mne.minimum_norm import get_point_spread, make_inverse_resolution_matrix

print(__doc__)

data_path = sample.data_path()
subjects_dir = data_path / "subjects"
meg_path = data_path / "MEG" / "sample"
fname_fwd = meg_path / "sample_audvis-meg-eeg-oct-6-fwd.fif"
fname_cov = meg_path / "sample_audvis-cov.fif"
fname_evo = meg_path / "sample_audvis-ave.fif"

# read forward solution
forward = mne.read_forward_solution(fname_fwd)
# forward operator with fixed source orientations
mne.convert_forward_solution(forward, surf_ori=True, force_fixed=True, copy=False)

# noise covariance matrix
noise_cov = mne.read_cov(fname_cov)

# evoked data for info
info = mne.io.read_info(fname_evo)

# make inverse operator from forward solution
# free source orientation
inverse_operator = mne.minimum_norm.make_inverse_operator(
    info=info, forward=forward, noise_cov=noise_cov, loose=0.0, depth=None
)

# regularisation parameter
snr = 3.0
lambda2 = 1.0 / snr**2
method = "MNE"

# %%
# We compute noisy PSFs for two locations, one deep and one superficial.
# For each location we compute two PSFs::
# 1) For a high SNR, approximating the noise-less case.
# 2) With the SNR of regularization and the added noise matched (realistic).

# vertices of a deep and a superficial source
sources = {"deep": 2146, "super": 2209}

# point-spread functions of the deep and superficial source
stc_psf = {"deep": {}, "super": {}}

# for a deep source
# compute resolution matrix with added noise for columns (PSFs)
# added noise matches the SNR used for regularization
rm_matched = make_inverse_resolution_matrix(
    forward, inverse_operator, method=method, lambda2=lambda2, noise_cov=noise_cov
)
stc_psf["deep"]["matched"] = get_point_spread(
    rm_matched, forward["src"], [sources["deep"]], norm=True
)
del rm_matched

# compute resolution matrix with added noise for very high SNR
rm_highsnr = make_inverse_resolution_matrix(
    forward,
    inverse_operator,
    method=method,
    lambda2=lambda2,
    noise_cov=noise_cov,
    snr=1000.0,
)
stc_psf["deep"]["highsnr"] = get_point_spread(
    rm_highsnr, forward["src"], [sources["deep"]], norm=True
)
del rm_highsnr

# the same for a superficial source
# compute matching resolution matrix with added noise for columns (PSFs)
# added noise matches the SNR used for regularization
rm_matched = make_inverse_resolution_matrix(
    forward, inverse_operator, method=method, lambda2=lambda2, noise_cov=noise_cov
)
stc_psf["super"]["matched"] = get_point_spread(
    rm_matched, forward["src"], [sources["super"]], norm=True
)
del rm_matched

# compute resolution matrix with added noise for very high SNR
rm_highsnr = make_inverse_resolution_matrix(
    forward,
    inverse_operator,
    method=method,
    lambda2=lambda2,
    noise_cov=noise_cov,
    snr=1000.0,
)
stc_psf["super"]["highsnr"] = get_point_spread(
    rm_highsnr, forward["src"], [sources["super"]], norm=True
)
del rm_highsnr

##############################################################################
# Visualize
# ---------
# PSF:

vertno_lh = forward["src"][0]["vertno"]
for source in sources:
    # Which vertex corresponds to selected source
    verttrue = [vertno_lh[sources[source]]]  # just one vertex
    for psf in ["matched", "highsnr"]:
        brain_psf = stc_psf[source][psf].plot(
            "sample", "inflated", "lh", subjects_dir=subjects_dir
        )
        brain_psf.show_view("ventral")
        title_str = f"{method} {source} {psf}"
        brain_psf.add_text(
            0.1, 0.9, title_str, "title", font_size=16, color=(255, 255, 255)
        )
        # mark true source location
        brain_psf.add_foci(
            verttrue, coords_as_verts=True, scale_factor=1.0, hemi="lh", color="green"
        )

# %%
# The green spheres indicate the true source location.
# For high SNR, PSFs for neither location are ideal. For the superficial
# it is at least centred around the true source location, but still widespread
# and with sidelobes. For the deep location the peaks occur at large distances
# to the true location. This shows that spatial resolution is not an issue of
# insufficient SNR but is inherently limited.
# For matched (more realistic) SNR the PSF resembles the noise-less PSF, albeit
# with more widely spread-out sidelobes. This indicates that moderate
# regularization still achieves good spatial resolution while effectively
# suppressing noise.
