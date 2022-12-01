# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#          Jaakko Leppakangas <jaeilepp@student.jyu.fi>
#          Daniel McCloy <dan@mccloy.info>
#
# License: Simplified BSD

import matplotlib.pyplot as plt
import numpy as np
import pytest

from mne import Epochs, create_info, EpochsArray
from mne.datasets import testing
from mne.event import make_fixed_length_events
from mne.viz import plot_drop_log


def test_plot_epochs_not_preloaded(epochs_unloaded, browser_backend):
    """Test plotting non-preloaded epochs."""
    assert epochs_unloaded._data is None
    epochs_unloaded.plot()
    assert epochs_unloaded._data is None


def test_plot_epochs_basic(epochs, epochs_full, noise_cov_io, capsys,
                           browser_backend):
    """Test epoch plotting."""
    assert len(epochs.events) == 1
    with epochs.info._unlock():
        epochs.info['lowpass'] = 10.  # allow heavy decim during plotting
    fig = epochs.plot(scalings=None, title='Epochs')
    ticks = fig._get_ticklabels('x')
    assert ticks == ['2']
    browser_backend._close_all()
    # covariance / whitening
    assert len(noise_cov_io['names']) == 366  # all channels
    assert noise_cov_io['bads'] == []
    assert epochs.info['bads'] == []  # all good
    with pytest.warns(RuntimeWarning, match='projection'):
        epochs.plot(noise_cov=noise_cov_io)
    browser_backend._close_all()
    # add a channel to the epochs.info['bads']
    epochs.info['bads'] = [epochs.ch_names[0]]
    with pytest.warns(RuntimeWarning, match='projection'):
        epochs.plot(noise_cov=noise_cov_io)
    browser_backend._close_all()
    # add a channel to cov['bads']
    noise_cov_io['bads'] = [epochs.ch_names[1]]
    with pytest.warns(RuntimeWarning, match='projection'):
        epochs.plot(noise_cov=noise_cov_io)
    browser_backend._close_all()
    # have a data channel missing from the covariance
    noise_cov_io['names'] = noise_cov_io['names'][:306]
    noise_cov_io['data'] = noise_cov_io['data'][:306][:306]
    with pytest.warns(RuntimeWarning, match='projection'):
        epochs.plot(noise_cov=noise_cov_io)
    browser_backend._close_all()
    # other options
    fig = epochs[0].plot(picks=[0, 2, 3], scalings=None)
    fig._fake_keypress('escape')
    with pytest.raises(ValueError, match='No appropriate channels found'):
        epochs.plot(picks=[])
    # gh-5906
    assert len(epochs_full) == 7
    epochs_full.info['bads'] = [epochs_full.ch_names[0]]
    capsys.readouterr()
    # test title error handling
    with pytest.raises(TypeError, match='title must be None or a string, got'):
        epochs_full.plot(title=7)
    # test auto-generated title, and selection mode
    epochs_full.plot(group_by='selection', title='')


@pytest.mark.parametrize('scalings', (dict(mag=1e-12, grad=1e-11, stim='auto'),
                                      None, 'auto'))
def test_plot_epochs_scalings(epochs, scalings, browser_backend):
    """Test the valid options for scalings."""
    epochs.plot(scalings=scalings)


def test_plot_epochs_colors(epochs, browser_backend):
    """Test epoch_colors, for compatibility with autoreject."""
    epoch_colors = [['r'] * len(epochs.ch_names) for _ in
                    range(len(epochs.events))]
    epochs.plot(epoch_colors=epoch_colors)
    with pytest.raises(ValueError, match='length equal to the number of epo'):
        epochs.plot(epoch_colors=[['r'], ['b']])  # epochs obj has only 1 epoch
    with pytest.raises(ValueError, match=r'epoch colors for epoch \d+ has'):
        epochs.plot(epoch_colors=[['r']])  # need 1 color for each channel
    # also test event_color
    epochs.plot(event_color='b')


def test_plot_epochs_scale_bar(epochs, browser_backend):
    """Test scale bar for epochs."""
    fig = epochs.plot()
    texts = fig._get_scale_bar_texts()
    # mag & grad in this instance
    if browser_backend.name == 'pyqtgraph':
        assert len(texts) == 2
        wants = ('800.0 fT/cm', '2000.0 fT')
    elif browser_backend.name == 'matplotlib':
        assert len(texts) == 4
        wants = ('800.0 fT/cm', '0.55 sec', '2000.0 fT', '0.55 sec')
    assert texts == wants


def test_plot_epochs_clicks(epochs, epochs_full, capsys,
                            browser_backend):
    """Test plot_epochs mouse interaction."""
    fig = epochs.plot(events=epochs.events)
    x = fig.mne.traces[0].get_xdata()[3]
    y = fig.mne.traces[0].get_ydata()[3]
    n_epochs = len(epochs)
    epoch_num = fig.mne.inst.selection[0]
    # test (un)marking bad epochs
    fig._fake_click((x, y), xform='data')  # mark a bad epoch
    assert epoch_num in fig.mne.bad_epochs
    fig._fake_click((x, y), xform='data')  # unmark it
    assert epoch_num not in fig.mne.bad_epochs
    fig._fake_click((x, y), xform='data')  # mark it bad again
    assert epoch_num in fig.mne.bad_epochs
    # test vline
    fig._fake_keypress('escape')  # close and drop epochs
    fig._close_event()  # XXX workaround, MPL Agg doesn't trigger close event
    assert n_epochs - 1 == len(epochs)
    # test marking bad channels
    # need more than 1 epoch this time
    fig = epochs_full.plot(n_epochs=3)
    first_ch = fig._get_ticklabels('y')[0]
    assert first_ch not in fig.mne.info['bads']
    fig._click_ch_name(ch_index=0, button=1)  # click ch name to mark bad
    assert first_ch in fig.mne.info['bads']
    # test clicking scrollbars
    fig._fake_click((0.5, 0.5), ax=fig.mne.ax_vscroll)
    fig._fake_click((0.5, 0.5), ax=fig.mne.ax_hscroll)
    # test moving bad epoch offscreen
    fig._fake_keypress('right')  # move right
    x = fig.mne.traces[0].get_xdata()[-3]
    y = fig.mne.traces[0].get_ydata()[-3]
    fig._fake_click((x, y), xform='data')  # mark a bad epoch
    fig._fake_keypress('left')  # move back
    out, err = capsys.readouterr()
    assert 'out of bounds' not in out
    assert 'out of bounds' not in err
    fig._fake_keypress('escape')
    fig._close_event()  # XXX workaround, MPL Agg doesn't trigger close event
    assert len(epochs_full) == 6
    # test rightclick → image plot
    fig = epochs_full.plot()
    fig._click_ch_name(ch_index=0, button=3)  # show image plot
    assert len(fig.mne.child_figs) == 1
    # test scroll wheel
    fig._fake_scroll(0.5, 0.5, -0.5)  # scroll down
    fig._fake_scroll(0.5, 0.5, 0.5)  # scroll up


def test_plot_epochs_keypresses(epochs_full, browser_backend):
    """Test plot_epochs keypress interaction."""
    # we need more than 1 epoch
    epochs_full.drop_bad(dict(mag=4e-12))  # for histogram plot coverage
    fig = epochs_full.plot(n_epochs=3)
    # make sure green vlines are visible first (for coverage)
    sample_idx = len(epochs_full.times) // 2  # halfway through the first epoch
    x = fig.mne.traces[0].get_xdata()[sample_idx]
    y = (fig.mne.traces[0].get_ydata()[sample_idx]
         + fig.mne.traces[1].get_ydata()[sample_idx]) / 2
    fig._fake_click([x, y], xform='data')  # click between traces
    # test keys
    keys = ('pagedown', 'down', 'up', 'down', 'right', 'left', '-', '+', '=',
            'd', 'd', 'pageup', 'home', 'shift+right', 'end', 'shift+left',
            'z', 'z', 's', 's', '?', 'h', 'j', 'b')
    for key in keys * 2:  # test twice → once in normal, once in butterfly view
        fig._fake_keypress(key)
    fig._fake_click([x, y], xform='data', button=3)  # remove vlines


def test_plot_overlapping_epochs_with_events(browser_backend):
    """Test drawing of event lines in overlapping epochs."""
    data = np.zeros(shape=(3, 2, 100))  # 3 epochs, 2 channels, 100 samples
    sfreq = 100
    info = create_info(
        ch_names=('a', 'b'), ch_types=('misc', 'misc'), sfreq=sfreq)
    # 90% overlap, so all 3 events should appear in all 3 epochs when plotted:
    events = np.column_stack(([50, 60, 70], [0, 0, 0], [1, 2, 3]))
    epochs = EpochsArray(data, info, tmin=-0.5, events=events)
    fig = epochs.plot(events=events, picks='misc')
    if browser_backend.name == 'matplotlib':
        n_event_lines = len(fig.mne.event_lines.get_segments())
    else:
        n_event_lines = len(fig.mne.event_lines)
    assert n_event_lines == 9


def test_epochs_plot_sensors(epochs):
    """Test sensor plotting."""
    epochs.plot_sensors()


def test_plot_epochs_nodata(browser_backend):
    """Test plotting of epochs when no data channels are present."""
    data = np.random.RandomState(0).randn(10, 2, 1000)
    info = create_info(2, 1000., 'stim')
    epochs = EpochsArray(data, info)
    with pytest.raises(ValueError, match='consider passing picks explicitly'):
        epochs.plot()


@pytest.mark.slowtest
def test_plot_epochs_image(epochs):
    """Test plotting of epochs image.

    Note that some of these tests that should pass are triggering MPL
    UserWarnings about tight_layout not being applied ("tight_layout cannot
    make axes width small enough to accommodate all axes decorations"). Calling
    `plt.close('all')` just before the offending test seems to prevent this
    warning, though it's unclear why.
    """
    figs = epochs.plot_image()
    assert len(figs) == 2  # one fig per ch_type (test data has mag, grad)
    assert len(plt.get_fignums()) == 2
    figs = epochs.plot_image()
    assert len(figs) == 2
    assert len(plt.get_fignums()) == 4  # should create new figures
    epochs.plot_image(picks='mag', sigma=0.1)
    epochs.plot_image(picks=[0, 1], combine='mean',
                      ts_args=dict(show_sensors=False))
    epochs.plot_image(picks=[1], order=[0], overlay_times=[0.1], vmin=0.01,
                      title='test')
    plt.close('all')
    epochs.plot_image(picks=[1], overlay_times=[0.1], vmin=-0.001, vmax=0.001)
    plt.close('all')
    epochs.plot_image(picks=[1], vmin=lambda x: x.min())
    # test providing figure
    fig, axs = plt.subplots(3, 1)
    epochs.plot_image(picks=[1], fig=fig)
    # test providing axes instance
    epochs.plot_image(picks=[1], axes=axs[0], evoked=False, colorbar=False)
    plt.close('all')
    # test order=callable
    epochs.plot_image(picks=[0, 1],
                      order=lambda times, data: np.arange(len(data))[::-1])
    # test warning
    with pytest.warns(RuntimeWarning, match='Only one channel in group'):
        epochs.plot_image(picks=[1], combine='mean')
    # group_by should be a dict
    with pytest.raises(TypeError, match="dict or None"):
        epochs.plot_image(group_by='foo')
    # units and scalings keys must match
    with pytest.raises(ValueError, match='Scalings and units must have the'):
        epochs.plot_image(units=dict(hi=1), scalings=dict(ho=1))
    plt.close('all')
    # test invert_y
    epochs.plot_image(ts_args=dict(invert_y=True))
    # can't combine different sensor types
    with pytest.raises(ValueError, match='Cannot combine sensors of differ'):
        epochs.plot_image(group_by=dict(foo=[0, 1, 2]))
    # can't pass both fig and axes
    with pytest.raises(ValueError, match='one of "fig" or "axes" must be'):
        epochs.plot_image(fig='foo', axes='bar')
    # wrong number of axes in fig
    with pytest.raises(ValueError, match='"fig" must contain . axes, got .'):
        epochs.plot_image(fig=plt.figure())
    # only 1 group allowed when fig is passed
    with pytest.raises(ValueError, match='"group_by" can only have one group'):
        fig, axs = plt.subplots(3, 1)
        epochs.plot_image(fig=fig, group_by=dict(foo=[0, 1], bar=[5, 6]))
        del fig, axs
    plt.close('all')
    # must pass correct number of axes (1, 2, or 3)
    with pytest.raises(ValueError, match='is a list, can only plot one group'):
        fig, axs = plt.subplots(1, 3)
        epochs.plot_image(axes=axs)
    for length, kwargs in ([3, dict()],
                           [2, dict(evoked=False)],
                           [2, dict(colorbar=False)],
                           [1, dict(evoked=False, colorbar=False)]):
        fig, axs = plt.subplots(1, length + 1)
        epochs.plot_image(picks='mag', axes=axs[:length], **kwargs)
        with pytest.raises(ValueError, match='"axes" must be length ., got .'):
            epochs.plot_image(picks='mag', axes=axs, **kwargs)
    plt.close('all')
    # mismatch between axes dict keys and group_by dict keys
    with pytest.raises(ValueError, match='must match the keys in "group_by"'):
        epochs.plot_image(axes=dict())
    # wrong number of axes in dict
    match = 'each value in "axes" must be a list of . axes, got .'
    with pytest.raises(ValueError, match=match):
        epochs.plot_image(axes=dict(foo=axs[:2], bar=axs[:3]),
                          group_by=dict(foo=[0, 1], bar=[5, 6]))
    # bad value of "combine"
    with pytest.raises(ValueError, match='"combine" must be None, a callable'):
        epochs.plot_image(combine='foo')
    # mismatched picks and overlay_times
    with pytest.raises(ValueError, match='size of overlay_times parameter'):
        epochs.plot_image(picks=[1], overlay_times=[0.1, 0.2])
    # bad overlay times
    with pytest.warns(RuntimeWarning, match='fall outside'):
        epochs.plot_image(overlay_times=[999.])
    # mismatched picks and order
    with pytest.raises(ValueError, match='must match the length of the data'):
        epochs.plot_image(picks=[1], order=[0, 1])
    # with a ref MEG channel (that we "convert" from a grad channel)
    with pytest.warns(RuntimeWarning, match='.* from T/m to T.$'):
        epochs.set_channel_types({epochs.ch_names[0]: 'ref_meg'})
    epochs.plot_image()
    plt.close('all')


def test_plot_epochs_image_emg():
    """Test plotting epochs image with EMG."""
    info = create_info(['EMG 001'], sfreq=100, ch_types='emg')
    data = np.ones((2, 1, 10))
    epochs = EpochsArray(data=data, info=info)
    epochs.plot_image('EMG 001', ts_args={"show_sensors": False})


def test_plot_drop_log(epochs_unloaded):
    """Test plotting a drop log."""
    with pytest.raises(ValueError, match='bad epochs have not yet been'):
        epochs_unloaded.plot_drop_log()
    epochs_unloaded.drop_bad()
    epochs_unloaded.plot_drop_log()
    plot_drop_log((('One',), (), ()))
    plot_drop_log((('One',), ('Two',), ()))
    plot_drop_log((('One',), ('One', 'Two'), ()))
    for arg in ([], ([],), (1,)):
        with pytest.raises(TypeError, match='tuple of tuple of str'):
            plot_drop_log(arg)
    plt.close('all')


def test_plot_psd_epochs(epochs):
    """Test plotting epochs psd (+topomap)."""
    epochs.plot_psd(average=True, spatial_colors=False)
    epochs.plot_psd(average=False, spatial_colors=True)
    epochs.plot_psd(average=False, spatial_colors=False)
    # test plot_psd_topomap errors
    with pytest.raises(RuntimeError, match='No frequencies in band'):
        epochs.plot_psd_topomap(bands=dict(foo=(0, 0.01)))
    plt.close('all')
    # test defaults
    fig = epochs.plot_psd_topomap()
    assert len(fig.axes) == 10  # default: 5 bands (δ, θ, α, β, γ) + colorbars
    # test joint vlim
    fig = epochs.plot_psd_topomap(vlim='joint')
    vmin_0 = fig.axes[0].images[0].norm.vmin
    vmax_0 = fig.axes[0].images[0].norm.vmax
    assert all(vmin_0 == ax.images[0].norm.vmin for ax in fig.axes[1:5])
    assert all(vmax_0 == ax.images[0].norm.vmax for ax in fig.axes[1:5])
    # test support for single-bin bands and old-style list-of-tuple input
    fig = epochs.plot_psd_topomap(bands=[(20, '20 Hz'), (15, 25, '15-25 Hz')])
    # test with a flat channel
    err_str = 'for channel %s' % epochs.ch_names[2]
    epochs.get_data()[0, 2, :] = 0
    for dB in [True, False]:
        with pytest.warns(UserWarning, match=err_str):
            epochs.plot_psd(dB=dB)


def test_plot_psdtopo_nirs(fnirs_epochs):
    """Test plotting of PSD topography for nirs data."""
    bands = {'0.2 Hz': 0.2, '0.4 Hz': 0.4, '0.8 Hz': 0.8}
    fig = fnirs_epochs.plot_psd_topomap(bands=bands)
    assert len(fig.axes) == 6  # 3 band x (plot + cmap)


@testing.requires_testing_data
def test_plot_epochs_ctf(raw_ctf, browser_backend):
    """Test of basic CTF plotting."""
    raw_ctf.pick_channels(['UDIO001', 'UPPT001', 'SCLK01-177',
                           'BG1-4304', 'MLC11-4304', 'MLC11-4304',
                           'EEG058', 'UADC007-4302'])
    evts = make_fixed_length_events(raw_ctf)
    epochs = Epochs(raw_ctf, evts, preload=True)
    epochs.plot()
    browser_backend._close_all()

    # test butterfly
    fig = epochs.plot(butterfly=True)
    # leave fullscreen testing to Raw / _figure abstraction (too annoying here)
    keys = ('b', 'b', 'pagedown', 'down', 'up', 'down', 'right', 'left', '-',
            '+', '=', 'd', 'd', 'pageup', 'home', 'end', 'z', 'z', 's', 's',
            '?', 'h', 'j')
    for key in keys:
        fig._fake_keypress(key)
    fig._fake_scroll(0.5, 0.5, -0.5)  # scroll down
    fig._fake_scroll(0.5, 0.5, 0.5)  # scroll up
    fig._resize_by_factor(1)
    fig._fake_keypress('escape')  # close and drop epochs


@pytest.mark.slowtest
@testing.requires_testing_data
def test_plot_psd_epochs_ctf(raw_ctf):
    """Test plotting CTF epochs psd (+topomap)."""
    evts = make_fixed_length_events(raw_ctf)
    epochs = Epochs(raw_ctf, evts, preload=True)
    # EEG060 is flat in this dataset
    for dB in [True, False]:
        with pytest.warns(UserWarning, match='for channel EEG060'):
            epochs.plot_psd(dB=dB)
    epochs.drop_channels(['EEG060'])
    epochs.plot_psd(spatial_colors=False, average=False)
    with pytest.raises(RuntimeError, match='No frequencies in band'):
        epochs.plot_psd_topomap(bands=[(0, 0.01, 'foo')])
    epochs.plot_psd_topomap()


def test_plot_epochs_selection_butterfly(raw, browser_backend):
    """Test that using selection and butterfly works."""
    events = make_fixed_length_events(raw)[:1]
    epochs = Epochs(raw, events, tmin=0, tmax=0.5, preload=True, baseline=None)
    assert len(epochs) == 1
    epochs.plot(group_by='selection', butterfly=True)
