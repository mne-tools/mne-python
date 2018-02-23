
import numpy as np
import mne

from mne.inverse_sparse import tf_mixed_norm
from mne.viz import plot_sparse_source_estimates

data_path = 'SEF_data/'

ave_fname = data_path + 'mind006_051209_median01_raw_daniel_long-ave.fif'
cov_fname = data_path + 'mind006_051209_median01_raw_daniel_long-cov.fif'
fwd_fname = data_path + 'mind006_051209_median01_raw-oct-6-fwd.fif'

evoked = mne.read_evokeds(ave_fname, condition=0, baseline=(None, 0))
noise_cov = mne.read_cov(cov_fname)
forward = mne.read_forward_solution(fwd_fname, surf_ori=True,
                                    force_fixed=False)

# Parameters
wsize = np.array([64, 16])
tstep = np.array([4, 2])

alpha, rho = 30., 0.05
alpha_space = (1. - rho) * alpha
alpha_time = alpha * rho
# alpha_space = 25.
# alpha_time = 3.5

window = 0.01

loose = 1.0
depth = 0.9
maxit = 10000
tol = 1e-6

tmin, tmax = 0.008, 0.21
evoked.crop(tmin, tmax)
evoked.resample(1000.)

out = tf_mixed_norm(
    evoked, forward, noise_cov, alpha_space, alpha_time, loose=loose,
    depth=depth, maxit=maxit, tol=tol, wsize=wsize, tstep=tstep,
    window=window, n_tfmxne_iter=100, return_residual=True,
    verbose=True)

stc = out[0]
residual = out[-1]
# Crop to remove edges
stc.crop(tmin=0.01, tmax=0.2)
evoked.crop(tmin=0.01, tmax=0.2)
residual.crop(tmin=0.01, tmax=0.2)

# Show the evoked response and the residual for gradiometers
ylim = dict(grad=[-250, 250])
evoked.pick_types(meg='grad', exclude='bads')
evoked.plot(titles=dict(grad='Evoked Response: Gradiometers'), ylim=ylim,
            proj=True)

residual.pick_types(meg='grad', exclude='bads')
residual.plot(titles=dict(grad='Residuals: Gradiometers'), ylim=ylim,
              proj=True)

###############################################################################
# View in 2D and 3D ("glass" brain like 3D plot)
plot_sparse_source_estimates(forward['src'], stc, bgcolor=(1, 1, 1),
                             opacity=0.1, fig_name="irTF-MxNE",
                             modes=['sphere'], scale_factors=[1.])



