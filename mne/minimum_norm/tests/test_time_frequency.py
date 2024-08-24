# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mne import Epochs, find_events, pick_types
from mne._fiff.constants import FIFF
from mne.datasets import testing
from mne.io import read_raw_fif
from mne.label import BiHemiLabel, read_label
from mne.minimum_norm import (
    INVERSE_METHODS,
    apply_inverse_epochs,
    prepare_inverse_operator,
    read_inverse_operator,
)
from mne.minimum_norm.time_frequency import (
    compute_source_psd,
    compute_source_psd_epochs,
    source_band_induced_power,
    source_induced_power,
)
from mne.time_frequency.multitaper import psd_array_multitaper

data_path = testing.data_path(download=False)
fname_inv = (
    data_path / "MEG" / "sample" / "sample_audvis_trunc-meg-eeg-oct-4-meg-inv.fif"
)
fname_data = data_path / "MEG" / "sample" / "sample_audvis_trunc_raw.fif"
fname_label = data_path / "MEG" / "sample" / "labels" / "Aud-lh.label"
fname_label2 = data_path / "MEG" / "sample" / "labels" / "Aud-rh.label"


@testing.requires_testing_data
@pytest.mark.parametrize("method", INVERSE_METHODS)
def test_tfr_with_inverse_operator(method):
    """Test time freq with MNE inverse computation."""
    tmin, tmax, event_id = -0.2, 0.5, 1

    # Setup for reading the raw data
    raw = read_raw_fif(fname_data)
    events = find_events(raw, stim_channel="STI 014")
    inv = read_inverse_operator(fname_inv)
    inv = prepare_inverse_operator(inv, nave=1, lambda2=1.0 / 9.0, method=method)

    raw.info["bads"] += ["MEG 2443", "EEG 053"]  # bads + 2 more

    # picks MEG gradiometers
    picks = pick_types(
        raw.info, meg=True, eeg=False, eog=True, stim=False, exclude="bads"
    )

    # Load condition 1
    event_id = 1
    events3 = events[:3]  # take 3 events to keep the computation time low
    epochs = Epochs(
        raw,
        events3,
        event_id,
        tmin,
        tmax,
        picks=picks,
        baseline=(None, 0),
        reject=dict(grad=4000e-13, eog=150e-6),
        preload=True,
    )

    # Compute a source estimate per frequency band
    bands = dict(alpha=[10, 10])
    label = read_label(fname_label)

    # XXX someday we should refactor this so that you don't have to pass
    # method -- maybe `prepare_inverse_operator` should add a `method`
    # to it and when `prepared=True` the value passed in can be ignored
    # (or better, default method=None means "dSPM if unprepared" and if they
    # actually pass a value, we check against `inv['method']`)
    stcs = source_band_induced_power(
        epochs,
        inv,
        bands,
        method=method,
        n_cycles=2,
        use_fft=False,
        pca=True,
        label=label,
        prepared=True,
    )

    stc = stcs["alpha"]
    assert len(stcs) == len(list(bands.keys()))
    assert np.all(stc.data > 0)
    assert_allclose(stc.times, epochs.times, atol=1e-6)

    stcs_no_pca = source_band_induced_power(
        epochs,
        inv,
        bands,
        method=method,
        n_cycles=2,
        use_fft=False,
        pca=False,
        label=label,
        prepared=True,
    )

    assert_allclose(stcs["alpha"].data, stcs_no_pca["alpha"].data)

    # Compute a source estimate per frequency band
    epochs = Epochs(
        raw,
        events[:10],
        event_id,
        tmin,
        tmax,
        picks=picks,
        baseline=(None, 0),
        reject=dict(grad=4000e-13, eog=150e-6),
        preload=True,
    )

    freqs = np.arange(7, 30, 2)  # define frequencies of interest
    power, phase_lock = source_induced_power(
        epochs,
        inv,
        freqs,
        label,
        baseline=(-0.1, 0),
        baseline_mode="percent",
        n_cycles=2,
        n_jobs=None,
        method=method,
        prepared=True,
    )
    assert power.shape == phase_lock.shape
    assert np.all(phase_lock > 0)
    assert np.all(phase_lock <= 1)
    assert 5 < np.max(power) < 7
    # fairly precise spot check that our values match what we had on 2023/09/28
    if method != "eLORETA":
        # check phase-lock using arbitrary index value since pl max is 1
        assert_allclose(phase_lock[1, 0, 0], 0.576, rtol=1e-3)
        # check power
        max_inds = np.unravel_index(np.argmax(power), power.shape)
        assert_allclose(max_inds, [0, 11, 135])
        assert_allclose(power[max_inds], 6.05, rtol=1e-3)


@testing.requires_testing_data
def test_tfr_multi_label():
    """Test multi-label functionality."""
    tmin, tmax, event_id = -0.2, 0.5, 1

    # Setup for reading the raw data
    raw = read_raw_fif(fname_data)
    events = find_events(raw, stim_channel="STI 014")
    inv = read_inverse_operator(fname_inv)
    inv = prepare_inverse_operator(inv, nave=1, lambda2=1.0 / 9.0, method="dSPM")

    raw.info["bads"] += ["MEG 2443", "EEG 053"]  # bads + 2 more

    # picks MEG gradiometers
    picks = pick_types(
        raw.info, meg=True, eeg=False, eog=True, stim=False, exclude="bads"
    )

    # Load condition 1
    event_id = 1
    epochs = Epochs(
        raw,
        events[:3],  # take 3 events to keep the computation time low
        event_id,
        tmin,
        tmax,
        picks=picks,
        baseline=(None, 0),
        reject=dict(grad=4000e-13, eog=150e-6),
        preload=True,
    )

    freqs = np.arange(7, 30, 2)

    n_times = len(epochs.times)
    n_freqs = len(freqs)

    # prepare labels
    label = read_label(fname_label)  # lh Aud
    label2 = read_label(fname_label2)  # rh Aud
    labels = [label, label2]
    bad_lab = label.copy()
    bad_lab.vertices = np.hstack((label.vertices, [2121]))  # add 1 unique vert
    bad_lbls = [label, bad_lab]
    nverts_lh = len(np.intersect1d(inv["src"][0]["vertno"], label.vertices))
    nverts_rh = len(np.intersect1d(inv["src"][1]["vertno"], label2.vertices))
    assert nverts_lh + 1 == nverts_rh == 3

    # prepare instances of BiHemiLabel
    fname_lvis = data_path / "MEG" / "sample" / "labels" / "Vis-lh.label"
    fname_rvis = data_path / "MEG" / "sample" / "labels" / "Vis-rh.label"
    lvis = read_label(fname_lvis)
    rvis = read_label(fname_rvis)
    bihl = BiHemiLabel(lh=label, rh=label2)  # auditory labels
    bihl.name = "Aud"
    bihl2 = BiHemiLabel(lh=lvis, rh=rvis)  # visual labels
    bihl2.name = "Vis"
    bihls = [bihl, bihl2]
    bad_bihl = BiHemiLabel(lh=bad_lab, rh=rvis)  # 1 unique vert on lh, rh ok
    bad_bihls = [bihl, bad_bihl]
    print("BiHemi label verts:", bihl.lh.vertices.shape, bihl.rh.vertices.shape)

    # check error handling
    sip_kwargs = dict(
        baseline=(-0.1, 0),
        baseline_mode="mean",
        n_cycles=2,
        n_jobs=None,
        return_plv=False,
        method="dSPM",
        prepared=True,
    )
    # label input errors
    with pytest.raises(TypeError, match="Label or BiHemi"):
        source_induced_power(epochs, inv, freqs, label="bad_input", **sip_kwargs)
    with pytest.raises(TypeError, match="Label or BiHemi"):
        source_induced_power(
            epochs, inv, freqs, label=[label, "bad_input"], **sip_kwargs
        )

    # error handling for multi-label and plv
    sip_kwargs_bad = sip_kwargs.copy()
    sip_kwargs_bad["return_plv"] = True
    with pytest.raises(RuntimeError, match="value cannot be calculated"):
        source_induced_power(epochs, inv, freqs, labels, **sip_kwargs_bad)

    # check multi-label handling
    label_sets = dict(Label=(labels, bad_lbls), BiHemi=(bihls, bad_bihls))
    for ltype, lab_set in label_sets.items():
        n_verts = nverts_lh if ltype == "Label" else nverts_lh + nverts_rh
        # check overlapping verts error handling
        with pytest.raises(RuntimeError, match="overlapping vertices"):
            source_induced_power(epochs, inv, freqs, lab_set[1], **sip_kwargs)

        # TODO someday, eliminate both levels of this nested for-loop and use
        # pytest.mark.parametrize, but not unless/until the data IO and the loading /
        # preparing of the inverse operator have been made into fixtures (the overhead
        # of those operations makes it a bad idea to parametrize now)
        for ori in (None, "normal"):  # check loose and normal orientations
            sip_kwargs.update(pick_ori=ori)
            lbl = lab_set[0][0]

            # check label=Label vs label=[Label]
            no_list_pow = source_induced_power(
                epochs, inv, freqs, label=lbl, **sip_kwargs
            )
            assert no_list_pow.shape == (n_verts, n_freqs, n_times)

            list_pow = source_induced_power(
                epochs, inv, freqs, label=[lbl], **sip_kwargs
            )
            assert list_pow.shape == (1, n_freqs, n_times)

            nlp_ave = np.mean(no_list_pow, axis=0)
            assert_allclose(nlp_ave, list_pow[0], rtol=1e-3)

            # check label=[Label1, Label2]
            multi_lab_pow = source_induced_power(
                epochs, inv, freqs, label=lab_set[0], **sip_kwargs
            )
            assert multi_lab_pow.shape == (2, n_freqs, n_times)


@testing.requires_testing_data
@pytest.mark.parametrize("method", INVERSE_METHODS)
@pytest.mark.parametrize("pick_ori", (None, "normal"))  # XXX vector someday?
@pytest.mark.parametrize("pca", (True, False))
def test_source_psd(method, pick_ori, pca):
    """Test source PSD computation from raw."""
    raw = read_raw_fif(fname_data)
    raw.crop(0, 5).load_data()
    inverse_operator = read_inverse_operator(fname_inv)
    fmin, fmax = 40, 65  # Hz
    n_fft = 512

    assert inverse_operator["source_ori"] == FIFF.FIFFV_MNE_FREE_ORI

    stc, ev = compute_source_psd(
        raw,
        inverse_operator,
        lambda2=1.0 / 9.0,
        method=method,
        fmin=fmin,
        fmax=fmax,
        pick_ori=pick_ori,
        n_fft=n_fft,
        overlap=0.0,
        return_sensor=True,
        pca=pca,
        dB=True,
    )

    assert ev.data.shape == (len(ev.info["ch_names"]), len(stc.times))
    assert ev.times[0] >= fmin
    assert ev.times[-1] <= fmax
    # Time max at line frequency (60 Hz in US)
    assert 58 <= ev.times[np.argmax(np.sum(ev.data, axis=0))] <= 61
    assert ev.nave == 2

    assert stc.shape[0] == inverse_operator["nsource"]
    assert stc.times[0] >= fmin
    assert stc.times[-1] <= fmax
    assert 58 <= stc.times[np.argmax(np.sum(stc.data, axis=0))] <= 61

    if method in ("sLORETA", "dSPM"):
        stc_dspm = stc
        stc_mne, _ = compute_source_psd(
            raw,
            inverse_operator,
            lambda2=1.0 / 9.0,
            method="MNE",
            fmin=fmin,
            fmax=fmax,
            pick_ori=pick_ori,
            n_fft=n_fft,
            overlap=0.0,
            return_sensor=True,
            dB=True,
        )
        # normalize each source point by its power after undoing the dB
        stc_dspm.data = 10 ** (stc_dspm.data / 10.0)
        stc_dspm /= stc_dspm.mean()
        stc_mne.data = 10 ** (stc_mne.data / 10.0)
        stc_mne /= stc_mne.mean()
        assert_allclose(stc_dspm.data, stc_mne.data, atol=1e-4)


@testing.requires_testing_data
@pytest.mark.parametrize("method", INVERSE_METHODS)
def test_source_psd_epochs(method):
    """Test multi-taper source PSD computation in label from epochs."""
    raw = read_raw_fif(fname_data)
    inverse_operator = read_inverse_operator(fname_inv)
    label = read_label(fname_label)
    label2 = read_label(fname_label2)

    event_id, tmin, tmax = 1, -0.2, 0.5
    lambda2 = 1.0 / 9.0
    bandwidth = 8.0
    fmin, fmax = 0, 100

    picks = pick_types(
        raw.info,
        meg=True,
        eeg=False,
        stim=True,
        ecg=True,
        eog=True,
        include=["STI 014"],
        exclude="bads",
    )
    reject = dict(grad=4000e-13, mag=4e-12, eog=150e-6)

    events = find_events(raw, stim_channel="STI 014")
    epochs = Epochs(
        raw,
        events,
        event_id,
        tmin,
        tmax,
        picks=picks,
        baseline=(None, 0),
        reject=reject,
    )

    # only look at one epoch
    epochs.drop_bad()
    one_epochs = epochs[:1]

    inv = prepare_inverse_operator(
        inverse_operator, nave=1, lambda2=1.0 / 9.0, method="dSPM"
    )

    # return list
    stc_psd = compute_source_psd_epochs(
        one_epochs,
        inv,
        lambda2=lambda2,
        method=method,
        pick_ori="normal",
        label=label,
        bandwidth=bandwidth,
        fmin=fmin,
        fmax=fmax,
        prepared=True,
    )[0]

    # return generator
    stcs = compute_source_psd_epochs(
        one_epochs,
        inv,
        lambda2=lambda2,
        method=method,
        pick_ori="normal",
        label=label,
        bandwidth=bandwidth,
        fmin=fmin,
        fmax=fmax,
        return_generator=True,
        prepared=True,
    )

    for stc in stcs:
        stc_psd_gen = stc

    assert_allclose(stc_psd.data, stc_psd_gen.data, atol=1e-7)

    # compare with direct computation
    stc = apply_inverse_epochs(
        one_epochs,
        inv,
        lambda2=lambda2,
        method=method,
        pick_ori="normal",
        label=label,
        prepared=True,
    )[0]

    sfreq = epochs.info["sfreq"]
    psd, freqs = psd_array_multitaper(
        stc.data, sfreq=sfreq, bandwidth=bandwidth, fmin=fmin, fmax=fmax
    )

    assert_allclose(psd, stc_psd.data, atol=1e-7)
    assert_allclose(freqs, stc_psd.times)

    # Check corner cases caused by tiny bandwidth
    with pytest.raises(ValueError, match="use a value of at least"):
        compute_source_psd_epochs(
            one_epochs,
            inv,
            lambda2=lambda2,
            method=method,
            pick_ori="normal",
            label=label,
            bandwidth=0.01,
            low_bias=True,
            fmin=fmin,
            fmax=fmax,
            return_generator=False,
            prepared=True,
        )

    # check error handling for label
    with pytest.raises(TypeError, match="Label or BiHemi"):
        compute_source_psd_epochs(one_epochs, inv, label=[label, label2])
