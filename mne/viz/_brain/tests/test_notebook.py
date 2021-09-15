# -*- coding: utf-8 -*-
#
# Authors: Guillaume Favelier <guillaume.favelier@gmail.com>
#          Eric Larson <larson.eric.d@gmail.com>

# NOTE: Tests in this directory must be self-contained because they are
# executed in a separate IPython kernel.

import sys
import pytest
from mne.datasets import testing


# This will skip all tests in this scope
pytestmark = pytest.mark.skipif(
    sys.platform.startswith('win'), reason='nbexec does not work on Windows')


@testing.requires_testing_data
def test_notebook_alignment(renderer_notebook, brain_gc, nbexec):
    """Test plot alignment in a notebook."""
    import mne
    data_path = mne.datasets.testing.data_path()
    raw_fname = data_path + '/MEG/sample/sample_audvis_trunc_raw.fif'
    subjects_dir = data_path + '/subjects'
    subject = 'sample'
    trans = data_path + '/MEG/sample/sample_audvis_trunc-trans.fif'
    info = mne.io.read_info(raw_fname)
    mne.viz.set_3d_backend('notebook')
    fig = mne.viz.plot_alignment(
        info, trans, subject=subject, dig=True,
        meg=['helmet', 'sensors'], subjects_dir=subjects_dir,
        surfaces=['head-dense'])
    assert fig.display is not None


@pytest.mark.slowtest  # ~3 min on GitHub macOS
@testing.requires_testing_data
def test_notebook_interactive(renderer_notebook, brain_gc, nbexec):
    """Test interactive modes."""
    import os
    import tempfile
    from contextlib import contextmanager
    from numpy.testing import assert_allclose
    from ipywidgets import Button
    import matplotlib.pyplot as plt
    import mne
    from mne.datasets import testing
    data_path = testing.data_path()
    sample_dir = os.path.join(data_path, 'MEG', 'sample')
    subjects_dir = os.path.join(data_path, 'subjects')
    fname_stc = os.path.join(sample_dir, 'sample_audvis_trunc-meg')
    stc = mne.read_source_estimate(fname_stc, subject='sample')
    stc.crop(0.1, 0.11)
    initial_time = 0.13
    mne.viz.set_3d_backend('notebook')
    brain_class = mne.viz.get_brain_class()

    @contextmanager
    def interactive(on):
        old = plt.isinteractive()
        plt.interactive(on)
        try:
            yield
        finally:
            plt.interactive(old)

    with interactive(False):
        brain = stc.plot(subjects_dir=subjects_dir, initial_time=initial_time,
                         clim=dict(kind='value', pos_lims=[3, 6, 9]),
                         time_viewer=True,
                         show_traces=True,
                         hemi='lh', size=300)
        assert isinstance(brain, brain_class)
        assert brain._renderer.figure.notebook
        assert brain._renderer.figure.display is not None
        brain._renderer._update()
        tmp_path = tempfile.mkdtemp()
        movie_path = os.path.join(tmp_path, 'test.gif')
        screenshot_path = os.path.join(tmp_path, 'test.png')
        brain._renderer.actions['movie_field'].value = movie_path
        brain._renderer.actions['screenshot_field'].value = screenshot_path
        total_number_of_buttons = sum(
            '_field' not in k for k in brain._renderer.actions.keys())
        assert 'play' in brain._renderer.actions
        # play is not a button widget, it does not have a click() method
        number_of_buttons = 1
        for action in brain._renderer.actions.values():
            if isinstance(action, Button):
                action.click()
                number_of_buttons += 1
        assert number_of_buttons == total_number_of_buttons
        assert os.path.isfile(movie_path)
        assert os.path.isfile(screenshot_path)
        img_nv = brain.screenshot()
        assert img_nv.shape == (300, 300, 3), img_nv.shape
        img_v = brain.screenshot(time_viewer=True)
        assert img_v.shape[1:] == (300, 3), img_v.shape
        # XXX This rtol is not very good, ideally would be zero
        assert_allclose(
            img_v.shape[0], img_nv.shape[0] * 1.25, err_msg=img_nv.shape,
            rtol=0.1)
        brain.close()


@testing.requires_testing_data
def test_notebook_button_counts(renderer_notebook, brain_gc, nbexec):
    """Test button counts."""
    import mne
    from ipywidgets import Button
    mne.viz.set_3d_backend('notebook')
    rend = mne.viz.create_3d_figure(size=(100, 100), scene=False)
    fig = rend.scene()
    mne.viz.set_3d_title(fig, 'Notebook testing')
    mne.viz.set_3d_view(fig, 200, 70, focalpoint=[0, 0, 0])
    assert fig.display is None
    rend.show()
    total_number_of_buttons = sum(
        '_field' not in k for k in rend.actions.keys())
    number_of_buttons = 0
    for action in rend.actions.values():
        if isinstance(action, Button):
            action.click()
            number_of_buttons += 1
    assert number_of_buttons == total_number_of_buttons
    assert fig.display is not None
