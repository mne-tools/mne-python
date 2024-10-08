# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import re

import numpy as np
import pytest

from mne import (
    SourceEstimate,
    VolSourceEstimate,
    VolVectorSourceEstimate,
    compute_covariance,
    compute_source_morph,
    make_fixed_length_epochs,
    make_forward_solution,
    read_bem_solution,
    read_forward_solution,
    read_trans,
    setup_volume_source_space,
)
from mne.datasets import testing
from mne.io import read_raw_fif
from mne.minimum_norm import apply_inverse, make_inverse_operator
from mne.utils import _record_warnings, catch_logging
from mne.viz import plot_volume_source_estimates
from mne.viz.utils import _fake_click, _fake_keypress

data_dir = testing.data_path(download=False)
subjects_dir = data_dir / "subjects"
fwd_fname = data_dir / "MEG" / "sample" / "sample_audvis_trunc-meg-vol-7-fwd.fif"


@pytest.mark.slowtest  # can be slow on OSX
@testing.requires_testing_data
@pytest.mark.parametrize(
    "mode, stype, init_t, want_t, init_p, want_p, bg_img",
    [
        ("glass_brain", "s", None, 2, None, (-30.9, 18.4, 56.7), None),
        ("stat_map", "vec", 1, 1, None, (15.7, 16.0, -6.3), None),
        ("glass_brain", "vec", None, 1, (10, -10, 20), (6.6, -9.0, 19.9), None),
        ("stat_map", "s", 1, 1, (-10, 5, 10), (-12.3, 2.0, 7.7), "brain.mgz"),
    ],
)
def test_plot_volume_source_estimates_basic(
    mode, stype, init_t, want_t, init_p, want_p, bg_img
):
    """Test interactive plotting of volume source estimates."""
    pytest.importorskip("nibabel")
    pytest.importorskip("dipy")
    pytest.importorskip("nilearn")
    forward = read_forward_solution(fwd_fname)
    sample_src = forward["src"]
    if init_p is not None:
        init_p = np.array(init_p) / 1000.0

    vertices = [s["vertno"] for s in sample_src]
    n_verts = sum(len(v) for v in vertices)
    n_time = 2
    data = np.random.RandomState(0).rand(n_verts, n_time)

    if stype == "vec":
        stc = VolVectorSourceEstimate(
            np.tile(data[:, np.newaxis], (1, 3, 1)), vertices, 1, 1
        )
    else:
        assert stype == "s"
        stc = VolSourceEstimate(data, vertices, 1, 1)
    # sometimes get scalars/index warning
    with _record_warnings():
        with catch_logging(verbose="debug") as log:
            fig = stc.plot(
                sample_src,
                subject="sample",
                subjects_dir=subjects_dir,
                mode=mode,
                initial_time=init_t,
                initial_pos=init_p,
                bg_img=bg_img,
                verbose=True,
            )
    log = log.getvalue()
    want_str = f"t = {want_t:0.3f} s"
    assert want_str in log, (want_str, init_t)
    want_str = f"({want_p[0]:0.1f}, {want_p[1]:0.1f}, {want_p[2]:0.1f}) mm"
    assert want_str in log, (want_str, init_p)
    for ax_idx in [0, 2, 3, 4]:
        _fake_click(fig, fig.axes[ax_idx], (0.3, 0.5))
    _fake_keypress(fig, "left")
    _fake_keypress(fig, "shift+right")
    if bg_img is not None:
        with pytest.raises(FileNotFoundError, match="MRI file .* not found"):
            stc.plot(
                sample_src,
                subject="sample",
                subjects_dir=subjects_dir,
                mode="stat_map",
                bg_img="junk.mgz",
            )
    use_ax = None
    for ax in fig.axes:
        if ax.get_xlabel().startswith("Time"):
            use_ax = ax
            break
    assert use_ax is not None
    label = use_ax.get_legend().get_texts()[0].get_text()
    assert re.match("[0-9]*", label) is not None, label


@pytest.mark.slowtest  # can be slow on OSX
@testing.requires_testing_data
def test_plot_volume_source_estimates_morph():
    """Test interactive plotting of volume source estimates with morph."""
    pytest.importorskip("nibabel")
    pytest.importorskip("dipy")
    pytest.importorskip("nilearn")
    forward = read_forward_solution(fwd_fname)
    sample_src = forward["src"]
    vertices = [s["vertno"] for s in sample_src]
    n_verts = sum(len(v) for v in vertices)
    n_time = 2
    data = np.random.RandomState(0).rand(n_verts, n_time)
    stc = VolSourceEstimate(data, vertices, 1, 1)
    sample_src[0]["subject_his_id"] = "sample"  # old src
    morph = compute_source_morph(
        sample_src, "sample", "fsaverage", zooms=5, subjects_dir=subjects_dir
    )
    initial_pos = (-0.05, -0.01, -0.006)
    # sometimes get scalars/index warning
    with _record_warnings():
        with catch_logging() as log:
            stc.plot(
                morph,
                subjects_dir=subjects_dir,
                mode="glass_brain",
                initial_pos=initial_pos,
                verbose=True,
            )
    log = log.getvalue()
    assert "t = 1.000 s" in log
    assert "(-52.0, -8.0, -7.0) mm" in log

    with pytest.raises(ValueError, match="Allowed values are"):
        stc.plot(sample_src, "sample", subjects_dir, mode="abcd")
    vertices.append([])
    surface_stc = SourceEstimate(data, vertices, 1, 1)
    with pytest.raises(TypeError, match="an instance of VolSourceEstimate"):
        plot_volume_source_estimates(surface_stc, sample_src, "sample", subjects_dir)
    with pytest.raises(ValueError, match="Negative colormap limits"):
        stc.plot(
            sample_src, "sample", subjects_dir, clim=dict(lims=[-1, 2, 3], kind="value")
        )


@testing.requires_testing_data
def test_plot_volume_source_estimates_on_vol_labels():
    """Test plot of source estimate on srcs setup on 2 labels."""
    pytest.importorskip("nibabel")
    pytest.importorskip("dipy")
    pytest.importorskip("nilearn")
    raw = read_raw_fif(
        data_dir / "MEG" / "sample" / "sample_audvis_trunc_raw.fif", preload=False
    )
    raw.pick("meg").crop(0, 10)
    raw.pick(raw.ch_names[::2]).del_proj().load_data()
    epochs = make_fixed_length_epochs(raw, preload=True).apply_baseline((None, None))
    evoked = epochs.average()
    subject = "sample"
    bem = read_bem_solution(
        subjects_dir / f"{subject}" / "bem" / "sample-320-bem-sol.fif"
    )
    pos = 25.0  # spacing in mm
    volume_label = [
        "Right-Cerebral-Cortex",
        "Left-Cerebral-Cortex",
    ]
    src = setup_volume_source_space(
        subject,
        subjects_dir=subjects_dir,
        pos=pos,
        mri=subjects_dir / subject / "mri" / "aseg.mgz",
        bem=bem,
        volume_label=volume_label,
        add_interpolator=False,
    )
    trans = read_trans(data_dir / "MEG" / "sample" / "sample_audvis_trunc-trans.fif")
    fwd = make_forward_solution(
        evoked.info,
        trans,
        src,
        bem,
        meg=True,
        eeg=False,
        mindist=0,
        n_jobs=1,
    )
    cov = compute_covariance(
        epochs,
        tmin=None,
        tmax=None,
        method="empirical",
    )
    inverse_operator = make_inverse_operator(evoked.info, fwd, cov, loose=1, depth=0.8)
    stc = apply_inverse(
        evoked, inverse_operator, 1.0 / 3**2, method="sLORETA", pick_ori=None
    )
    stc.plot(src, subject, subjects_dir, initial_time=0.03)
