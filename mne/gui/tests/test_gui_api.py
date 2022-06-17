# Authors: Guillaume Favelier <guillaume.favelier@gmail.com>
#
# License: Simplified BSD

import sys
import pytest

from mne.utils import _check_qt_version

# This will skip all tests in this scope
pytestmark = pytest.mark.skipif(
    sys.platform.startswith('win'), reason='nbexec does not work on Windows')


def test_gui_api(renderer_notebook, nbexec, n_warn=0):
    """Test GUI API."""
    import contextlib
    import mne
    import warnings
    import sys
    try:
        # Function
        n_warn  # noqa
    except Exception:
        # Notebook standalone mode
        n_warn = 0
    # nbexec does not expose renderer_notebook so I use a
    # temporary variable to synchronize the tests
    try:
        assert mne.MNE_PYVISTAQT_BACKEND_TEST
    except AttributeError:
        mne.viz.set_3d_backend('notebook')
        backend = 'notebook'
    else:
        backend = 'qt'
    renderer = mne.viz.backends.renderer._get_renderer(size=(300, 300))

    # theme
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        renderer._window_set_theme('/does/not/exist')
    if backend == 'qt':
        assert len(w) == 1
        assert 'not found' in str(w[0].message), str(w[0].message)
    else:
        assert len(w) == 0
    with mne.utils._record_warnings() as w:
        renderer._window_set_theme('dark')
    if sys.platform != 'darwin':  # sometimes this is fine
        assert len(w) == n_warn

    # window without 3d plotter
    if backend == 'qt':
        window = renderer._window_create()
        widget = renderer._window_create()
        central_layout = renderer._layout_create(orientation='grid')
        renderer._layout_add_widget(central_layout, widget, row=0, col=0)
        renderer._window_initialize(window=window,
                                    central_layout=central_layout)

    from unittest.mock import Mock
    mock = Mock()

    @contextlib.contextmanager
    def _check_widget_trigger(widget, mock, before, after, call_count=True,
                              get_value=True):
        if get_value:
            assert widget.get_value() == before
        old_call_count = mock.call_count
        try:
            yield
        finally:
            if get_value:
                assert widget.get_value() == after
            if call_count:
                assert mock.call_count == old_call_count + 1

    # --- BEGIN: dock ---
    renderer._dock_initialize(name='', area='left')

    # label (not interactive)
    widget = renderer._dock_add_label(
        value='',
        align=False,
        selectable=True,
    )
    widget = renderer._dock_add_label(
        value='',
        align=True,
    )
    widget.update()
    # labels are disabled by default with the notebook backend
    widget.set_enabled(False)
    assert not widget.is_enabled()
    widget.set_enabled(True)
    assert widget.is_enabled()

    # ToolButton
    widget = renderer._dock_add_button(
        name='',
        callback=mock,
        style='toolbutton',
        tooltip='button',
    )
    with _check_widget_trigger(widget, mock, None, None, get_value=False):
        widget.set_value(True)

    # PushButton
    widget = renderer._dock_add_button(
        name='',
        callback=mock,
    )
    with _check_widget_trigger(widget, mock, None, None, get_value=False):
        widget.set_value(True)

    # slider
    widget = renderer._dock_add_slider(
        name='',
        value=0,
        rng=[0, 10],
        callback=mock,
        tooltip='slider',
    )
    with _check_widget_trigger(widget, mock, 0, 5):
        widget.set_value(5)

    # check box
    widget = renderer._dock_add_check_box(
        name='',
        value=False,
        callback=mock,
        tooltip='check box',
    )
    with _check_widget_trigger(widget, mock, False, True):
        widget.set_value(True)

    # spin box
    renderer._dock_add_spin_box(
        name='',
        value=0,
        rng=[0, 1],
        callback=mock,
        step=0.1,
        tooltip='spin box',
    )
    widget = renderer._dock_add_spin_box(
        name='',
        value=0,
        rng=[0, 1],
        callback=mock,
        step=None,
    )
    with _check_widget_trigger(widget, mock, 0, 0.5):
        widget.set_value(0.5)

    # combo box
    widget = renderer._dock_add_combo_box(
        name='',
        value='foo',
        rng=['foo', 'bar'],
        callback=mock,
        tooltip='combo box',
    )
    with _check_widget_trigger(widget, mock, 'foo', 'bar'):
        widget.set_value('bar')

    # radio buttons
    widget = renderer._dock_add_radio_buttons(
        value='foo',
        rng=['foo', 'bar'],
        callback=mock,
    )
    with _check_widget_trigger(widget, mock, None, None, get_value=False):
        widget.set_value(1, 'bar')
    assert widget.get_value(0) == 'foo'
    assert widget.get_value(1) == 'bar'
    widget.set_enabled(False)

    # text field
    widget = renderer._dock_add_text(
        name='',
        value='foo',
        placeholder='',
        callback=mock,
    )
    with _check_widget_trigger(widget, mock, 'foo', 'bar'):
        widget.set_value('bar')
    widget.set_style(dict(border="2px solid #ff0000"))

    # file button
    renderer._dock_add_file_button(
        name='',
        desc='',
        func=mock,
        is_directory=True,
        tooltip='file button',
    )
    renderer._dock_add_file_button(
        name='',
        desc='',
        func=mock,
        initial_directory='',
    )
    renderer._dock_add_file_button(
        name='',
        desc='',
        func=mock,
    )
    widget = renderer._dock_add_file_button(
        name='',
        desc='',
        func=mock,
        save=True
    )
    # XXX: the internal file dialogs may hang without signals
    widget.set_enabled(False)

    renderer._dock_initialize(name='', area='right')
    renderer._dock_named_layout(name='')
    for collapse in (None, True, False):
        renderer._dock_add_group_box(name='', collapse=collapse)
    renderer._dock_add_stretch()
    renderer._dock_add_layout()
    renderer._dock_finalize()
    renderer._dock_hide()
    renderer._dock_show()
    # --- END: dock ---

    # --- BEGIN: tool bar ---
    renderer._tool_bar_initialize(
        name="default",
        window=None,
    )

    # button
    assert 'reset' not in renderer.actions
    renderer._tool_bar_add_button(
        name='reset',
        desc='',
        func=mock,
        icon_name='help',
    )
    assert 'reset' in renderer.actions

    # icon
    renderer._tool_bar_update_button_icon(
        name='reset',
        icon_name='reset',
    )

    # text
    renderer._tool_bar_add_text(
        name='',
        value='',
        placeholder='',
    )

    # spacer
    renderer._tool_bar_add_spacer()

    # file button
    assert 'help' not in renderer.actions
    renderer._tool_bar_add_file_button(
        name='help',
        desc='',
        func=mock,
        shortcut=None,
    )
    renderer.actions['help'].trigger()

    # play button
    assert 'play' not in renderer.actions
    renderer._tool_bar_add_play_button(
        name='play',
        desc='',
        func=mock,
        shortcut=None,
    )
    assert 'play' in renderer.actions
    # --- END: tool bar ---

    # --- BEGIN: menu bar ---
    renderer._menu_initialize()

    # submenu
    renderer._menu_add_submenu(name='foo', desc='foo')
    assert 'foo' in renderer._menus
    assert 'foo' in renderer._menu_actions

    # button
    renderer._menu_add_button(
        menu_name='foo',
        name='bar',
        desc='bar',
        func=mock,
    )
    assert 'bar' in renderer._menu_actions['foo']
    with _check_widget_trigger(None, mock, '', '', get_value=False):
        renderer._menu_actions['foo']['bar'].trigger()

    # --- END: menu bar ---

    # --- BEGIN: status bar ---
    renderer._status_bar_initialize()
    renderer._status_bar_update()

    # label
    widget = renderer._status_bar_add_label(value='foo', stretch=0)
    assert widget.get_value() == 'foo'

    # progress bar
    widget = renderer._status_bar_add_progress_bar(stretch=0)
    # by default, get_value() is -1 for Qt and 0 for Ipywidgets
    widget.set_value(0)
    assert widget.get_value() == 0
    # --- END: status bar ---

    # --- BEGIN: tooltips ---
    widget = renderer._dock_add_button(
        name='',
        callback=mock,
        tooltip='foo'
    )
    assert widget.get_tooltip() == 'foo'
    # Change it …
    widget.set_tooltip('bar')
    assert widget.get_tooltip() == 'bar'
    # --- END: tooltips ---

    # --- BEGIN: dialog ---
    # dialogs are not supported yet on notebook
    if renderer._kind == 'qt':
        # warning
        buttons = ["Save", "Cancel"]
        widget = renderer._dialog_create(
            title='',
            text='',
            info_text='',
            callback=mock,
            buttons=buttons,
            modal=False,
        )
        widget.show()
        for button in buttons:
            with _check_widget_trigger(None, mock, '', '', get_value=False):
                widget.trigger(button=button)
            assert mock.call_args.args == (button,)

        # buttons list empty means OK button (default)
        button = 'Ok'
        widget = renderer._dialog_create(
            title='',
            text='',
            info_text='',
            callback=mock,
            icon='NoIcon',
            modal=False,
        )
        widget.show()
        with _check_widget_trigger(None, mock, '', '', get_value=False):
            widget.trigger(button=button)
        assert mock.call_args.args == (button,)
    # --- END: dialog ---

    # --- BEGIN: keypress ---
    renderer._keypress_initialize()
    renderer._keypress_add('a', mock)
    # keypress is not supported yet on notebook
    if renderer._kind == 'qt':
        with _check_widget_trigger(None, mock, '', '', get_value=False):
            renderer._keypress_trigger('a')
    # --- END: keypress ---

    renderer.show()

    renderer._window_close_connect(lambda: mock('first'), after=False)
    renderer._window_close_connect(lambda: mock('last'))
    old_call_count = mock.call_count
    renderer.close()
    if renderer._kind == 'qt':
        assert mock.call_count == old_call_count + 2
        assert mock.call_args_list[-1].args == ('last',)
        assert mock.call_args_list[-2].args == ('first',)


def test_gui_api_qt(renderer_interactive_pyvistaqt):
    """Test GUI API with the Qt backend."""
    import mne
    mne.MNE_PYVISTAQT_BACKEND_TEST = True
    _, api = _check_qt_version(return_api=True)
    n_warn = int(api in ('PySide6', 'PyQt6'))
    test_gui_api(None, None, n_warn=n_warn)
