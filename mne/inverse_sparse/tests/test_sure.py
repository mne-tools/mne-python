import mne
from mne.inverse_sparse.mxne_inverse import mixed_norm
from mne.viz import plot_sparse_source_estimates
from mne.datasets import sample

if __name__ == "__main__":
    data_path = sample.data_path()
    fwd_fname = data_path + "/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif"
    ave_fname = data_path + "/MEG/sample/sample_audvis-ave.fif"
    cov_fname = data_path + "/MEG/sample/sample_audvis-shrunk-cov.fif"

    noise_cov = mne.read_cov(cov_fname)

    evoked = mne.read_evokeds(
        ave_fname, condition="Left visual", baseline=(None, 0)
    )
    evoked.crop(tmin=0.05, tmax=0.15)
    evoked = evoked.pick_types(eeg=False, meg=True)

    forward = mne.read_forward_solution(fwd_fname)

    stc = mixed_norm(
        evoked, forward, noise_cov, "sure", n_mxne_iter=5, loose=0, depth=0.9,
        grid_lower_bound=0.1, alpha_grid_length=30)

    plot_sparse_source_estimates(
        forward["src"], stc, bgcolor=(1, 1, 1), opacity=0.1)
