# Authors: Guillaume Favelier <guillaume.favelier@gmail.com>
#
# License: Simplified BSD


def test_gui_api(renderer_notebook, nbexec):
    """Test GUI API."""
    import contextlib
    import mne
    # nbexec does not expose renderer_notebook so I use a
    # temporary variable to synchronize the tests
    try:
        assert mne.MNE_PYVISTAQT_BACKEND_TEST
    except AttributeError:
        mne.viz.set_3d_backend('notebook')
    renderer = mne.viz.backends.renderer._get_renderer(size=(300, 300))

    from unittest.mock import Mock
    mock = Mock()

    @contextlib.contextmanager
    def _check_widget_trigger(widget, mock, before, after, call_count=True,
                              get_value=True, idx=None):
        if get_value:
            if idx is None:
                assert widget.get_value() == before
            else:
                assert widget.get_value(idx) == before
        old_call_count = mock.call_count
        try:
            yield
        finally:
            if get_value:
                if idx is None:
                    assert widget.get_value() == after
                else:
                    assert widget.get_value(idx) == after
            if call_count:
                assert mock.call_count == old_call_count + 1

    # --- BEGIN: dock ---
    renderer._dock_initialize(name='', area='left')

    # label
    widget = renderer._dock_add_label('', align=True)
    widget.update()
    widget.set_enabled(False)

    # XXX: it is not stable yet
    # button
    # widget = renderer._dock_add_button('', mock)
    # with _check_widget_trigger(widget, mock, None, None, get_value=False):
    #     widget.set_value(True)

    # slider
    widget = renderer._dock_add_slider('', 0, [0, 10], mock)
    with _check_widget_trigger(widget, mock, 0, 5):
        widget.set_value(5)

    # check box
    widget = renderer._dock_add_check_box('', False, mock)
    with _check_widget_trigger(widget, mock, False, True):
        widget.set_value(True)

    # spin box
    renderer._dock_add_spin_box('', 0, [0, 1], mock, step=0.1)
    widget = renderer._dock_add_spin_box('', 0, [0, 1], mock, step=None)
    with _check_widget_trigger(widget, mock, 0, 0.5):
        widget.set_value(0.5)

    # combo box
    widget = renderer._dock_add_combo_box('', 'foo', ['foo', 'bar'], mock)
    with _check_widget_trigger(widget, mock, 'foo', 'bar'):
        widget.set_value('bar')

    # XXX: it is not stable yet
    # # radio buttons
    # widget = renderer._dock_add_radio_buttons('foo', ['foo', 'bar'], mock)
    # with _check_widget_trigger(widget, mock, 'foo', 'bar', idx=0):
    #     widget.set_value(1, 'bar')
    # assert widget.get_value(1)

    # text field
    widget = renderer._dock_add_text('', 'foo', '')
    with _check_widget_trigger(widget, mock, 'foo', 'bar', call_count=False):
        widget.set_value('bar')

    # file button
    renderer._dock_add_file_button('', '', mock, directory=True)
    renderer._dock_add_file_button('', '', mock, save=True)
    widget = renderer._dock_add_file_button('', '', mock,
                                            input_text_widget=False)
    # XXX: the internal file dialogs may hang without signals

    renderer._dock_initialize(name='', area='right')
    renderer._dock_named_layout(name='')
    renderer._dock_add_group_box(name='')
    renderer._dock_add_stretch()
    renderer._dock_add_layout()
    renderer._dock_finalize()
    renderer._dock_hide()
    renderer._dock_show()
    # --- END: dock ---

    renderer.show()
    renderer.close()


def test_gui_api_qt(renderer_interactive_pyvistaqt):
    """Test GUI API with the Qt backend."""
    import mne
    mne.MNE_PYVISTAQT_BACKEND_TEST = True
    test_gui_api(None, None)
