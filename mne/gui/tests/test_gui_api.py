# Authors: Guillaume Favelier <guillaume.favelier@gmail.com>
#
# License: Simplified BSD


def test_gui_api(renderer_pyvistaqt):
    """Test GUI API."""
    from unittest.mock import Mock
    renderer = renderer_pyvistaqt._get_renderer(size=(300, 300))

    # --- BEGIN: dock ---
    renderer._dock_initialize(name='', area='left')

    # label
    renderer._dock_add_label('', align=True)

    # text field
    renderer._dock_add_text('', '', '')

    # button
    mock = Mock()
    renderer._dock_add_button('', mock)

    # file button
    mock = Mock()
    renderer._dock_add_file_button('', '', mock, directory=True)
    renderer._dock_add_file_button('', '', mock, save=True)
    renderer._dock_add_file_button('', '', mock,
                                   input_text_widget=False)
    # spin box
    mock = Mock()
    renderer._dock_add_spin_box('', 0, [0, 1], mock, step=None)
    renderer._dock_add_spin_box('', 0, [0, 1], mock, step=0.1)

    # check box
    mock = Mock()
    widget = renderer._dock_add_check_box('', False, mock)
    assert not widget.get_value()
    assert mock.call_count == 0
    widget.set_value(True)  # change value to trigger the callback
    assert mock.call_count == 1

    # radio button
    mock = Mock()
    widget = renderer._dock_add_radio_buttons('foo', ['foo', 'bar'], mock)
    widget.set_value(0, True)
    widget.set_enabled(False)

    renderer._dock_initialize(name='', area='right')
    renderer._dock_finalize()
    renderer._dock_hide()
    renderer._dock_show()
    # --- END: dock ---

    renderer.show()
    renderer.close()
