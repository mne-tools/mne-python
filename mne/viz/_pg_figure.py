# -*- coding: utf-8 -*-
"""Base classes and functions for 2D browser backends."""

# Authors: Martin Schulz <dev@earthman-music.de>
#
# License: Simplified BSD

import datetime
import math
import platform
from contextlib import contextmanager
from functools import partial

import numpy as np
from scipy.stats import zscore

from PyQt5.QtCore import (QEvent, QPointF, Qt, pyqtSignal, QRunnable,
                          QObject, QThreadPool, QRectF)
from PyQt5.QtGui import (QFont, QIcon, QPixmap, QTransform,
                         QMouseEvent, QPainter, QImage, QPen)
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import (QAction, QColorDialog, QComboBox, QDialog,
                             QDockWidget, QDoubleSpinBox, QFormLayout,
                             QGridLayout, QHBoxLayout, QInputDialog,
                             QLabel, QMainWindow, QMessageBox,
                             QPushButton, QScrollBar, QSizePolicy,
                             QWidget, QStyleOptionSlider, QStyle,
                             QApplication, QGraphicsView, QProgressBar,
                             QVBoxLayout, QLineEdit, QCheckBox, QScrollArea)
from pyqtgraph import (AxisItem, GraphicsView, InfLineLabel, InfiniteLine,
                       LinearRegionItem,
                       PlotCurveItem, PlotItem, TextItem, ViewBox, functions,
                       mkBrush, mkPen, setConfigOption, mkQApp, mkColor)
try:
    from pytestqt.exceptions import capture_exceptions
except ModuleNotFoundError:
    @contextmanager
    def capture_exceptions():
        yield [None]


from ._figure import BrowserBase
from ..annotations import _sync_onset
from ..io.pick import _DATA_CH_TYPES_ORDER_DEFAULT
from ..utils import logger

name = 'pyqtgraph'


class RawTraceItem(PlotCurveItem):
    """Graphics-Object for single data trace."""

    def __init__(self, mne, ch_idx):
        super().__init__(clickable=True)
        # ToDo: Does it affect performance, if the mne-object is referenced
        #  to in every RawTraceItem?
        self.mne = mne
        self.check_nan = self.mne.check_nan

        # Set default z-value to 1 to be before other items in scene
        self.setZValue(1)

        self.set_ch_idx(ch_idx)
        self._update_bad_color()
        self.update_data()

    def _update_bad_color(self):
        if self.isbad:
            self.setPen(self.mne.ch_color_bad)
        else:
            self.setPen(self.color)

    def update_range_idx(self):
        """Should be updated when view-range or ch_idx changes."""
        self.range_idx = np.argwhere(self.mne.picks == self.ch_idx)[0][0]

    def update_ypos(self):
        """Should be updated when butterfly is toggled or ch_idx changes."""
        if self.mne.butterfly:
            self.ypos = self.mne.butterfly_type_order.index(self.ch_type) + 1
        else:
            self.ypos = self.order_idx + 1

    def set_ch_idx(self, ch_idx):
        """Sets the channel index and all deriving indices."""
        # The ch_idx is the index of the channel represented by this trace
        # in the channel-order from the unchanged instance (which also picks
        # refer to).
        self.ch_idx = ch_idx
        # The range_idx is the index of the channel represented by this trace
        # in the shown range.
        self.update_range_idx()
        # The order_idx is the index of the channel represented by this trace
        # in the channel-order (defined e.g. by group_by).
        self.order_idx = np.argwhere(self.mne.ch_order == self.ch_idx)[0][0]
        self.ch_name = self.mne.inst.ch_names[ch_idx]
        self.isbad = self.ch_name in self.mne.info['bads']
        self.ch_type = self.mne.ch_types[ch_idx]
        self.color = self.mne.ch_color_dict[self.ch_type]
        self.update_ypos()

    def update_data(self):
        """Update data (fetch data from self.mne according to self.ch_idx)."""
        if self.check_nan:
            connect = 'finite'
            skip = False
        else:
            connect = 'all'
            skip = True

        if self.mne.data_preloaded:
            data = self.mne.data[self.order_idx]
        else:
            data = self.mne.data[self.range_idx]

        # Apply decim
        if all([i is not None for i in [self.mne.decim_times,
                                        self.mne.decim_data]]):
            times = self.mne.decim_times[self.mne.decim_data[self.range_idx]]
            data = data[..., ::self.mne.decim_data[self.range_idx]]
        else:
            times = self.mne.times

        self.setData(times, data, connect=connect, skipFiniteCheck=skip)

        self.setPos(0, self.ypos)

    def mouseClickEvent(self, ev):
        """Customize mouse click events."""
        if not self.clickable or ev.button() != Qt.MouseButton.LeftButton:
            ev.ignore()
            return
        if self.mouseShape().contains(ev.pos()):
            ev.accept()
            self.sigClicked.emit(self, ev)

    def get_xdata(self):
        """Get xdata for testing."""
        return self.xData

    def get_ydata(self):
        """Get ydata for testing."""
        return self.yData + self.ypos


class TimeAxis(AxisItem):
    """The X-Axis displaying the time."""

    def __init__(self, mne):
        self.mne = mne
        super().__init__(orientation='bottom')

    def tickValues(self, minVal, maxVal, size):
        """Customize creation of axis values from visible axis range."""
        if self.mne.is_epochs:
            values = self.mne.midpoints[np.argwhere(
                minVal <= self.mne.midpoints <= maxVal)]
            tick_values = [(len(self.mne.inst.times), values)]
            return tick_values
        else:
            return super().tickValues(minVal, maxVal, size)

    def tickStrings(self, values, scale, spacing):
        """Customize strings of axis values."""
        if self.mne.is_epochs:
            epoch_nums = self.mne.inst.selection
            ts = epoch_nums[np.in1d(self.mne.midpoints, values).nonzero()[0]]
            tick_strings = [str(v) for v in ts]

        elif self.mne.time_format == 'clock':
            meas_date = self.mne.info['meas_date']
            first_time = datetime.timedelta(seconds=self.mne.inst.first_time)
            digits = np.ceil(-np.log10(spacing) + 1).astype(int)
            tick_strings = list()
            for val in values:
                val_time = datetime.timedelta(seconds=val) + \
                           first_time + meas_date
                val_str = val_time.strftime('%H:%M:%S')
                if int(val_time.microsecond):
                    val_str += \
                        f'{round(val_time.microsecond * 1e-6, digits)}'[1:]
                tick_strings.append(val_str)
        else:
            tick_strings = super().tickStrings(values, scale, spacing)

        return tick_strings

    def repaint(self):
        """Repaint Time Axis."""
        self.picture = None
        self.update()

    def get_labels(self):
        """Get labels for testing."""
        values = self.tickValues(*self.mne.viewbox.viewRange()[0], None)
        labels = self.tickStrings(values, None, None)

        return labels


class ChannelAxis(AxisItem):
    """The Y-Axis displaying the channel-names."""

    def __init__(self, main):
        self.main = main
        self.mne = main.mne
        self.ch_texts = dict()
        super().__init__(orientation='left')

    def tickValues(self, minVal, maxVal, size):
        """Customize creation of axis values from visible axis range."""
        minVal, maxVal = sorted((minVal, maxVal))
        values = list(range(round(minVal) + 1, round(maxVal)))
        tick_values = [(1, values)]
        return tick_values

    def tickStrings(self, values, scale, spacing):
        """Customize strings of axis values."""
        # Get channel-names
        if self.mne.butterfly and self.mne.fig_selection is not None:
            exclude = ('Vertex', 'Custom')
            ticklabels = list(self.mne.ch_selections)
            keep_mask = np.in1d(ticklabels, exclude, invert=True)
            ticklabels = [t.replace('Left-', 'L-').replace('Right-', 'R-')
                          for t in ticklabels]  # avoid having to rotate labels
            tick_strings = np.array(ticklabels)[keep_mask]
        elif self.mne.butterfly:
            _, ixs, _ = np.intersect1d(_DATA_CH_TYPES_ORDER_DEFAULT,
                                       self.mne.ch_types, return_indices=True)
            ixs.sort()
            tick_strings = np.array(_DATA_CH_TYPES_ORDER_DEFAULT)[ixs]
        else:
            ch_idxs = [v - 1 for v in values]
            tick_strings = self.mne.ch_names[self.mne.ch_order[ch_idxs]]

        return tick_strings

    def drawPicture(self, p, axisSpec, tickSpecs, textSpecs):
        """Customize drawing of axis items."""
        super().drawPicture(p, axisSpec, tickSpecs, textSpecs)
        for rect, flags, text in textSpecs:
            if text in self.mne.info['bads']:
                p.setPen(functions.mkPen(self.mne.ch_color_bad))
            else:
                p.setPen(functions.mkPen('k'))
            self.ch_texts[text] = ((rect.left(), rect.left() + rect.width()),
                                   (rect.top(), rect.top() + rect.height()))
            p.drawText(rect, int(flags), text)

    def repaint(self):
        """Repaint Channel Axis."""
        self.picture = None
        self.update()

    def mouseClickEvent(self, event):
        """Customize mouse click events."""
        # Clean up channel-texts
        self.ch_texts = {k: v for k, v in self.ch_texts.items()
                         if k in [tr.ch_name for tr in self.mne.traces]}
        # Get channel-name from position of channel-description
        ypos = event.scenePos().y()
        for ch_name in self.ch_texts:
            ymin, ymax = self.ch_texts[ch_name][1]
            if ymin < ypos < ymax:
                print(f'{ch_name} clicked!')
                line = [li for li in self.mne.traces
                        if li.ch_name == ch_name][0]
                self.main._bad_ch_clicked(line)
                break
        # return super().mouseClickEvent(event)

    def get_labels(self):
        """Get labels for testing."""
        values = self.tickValues(*self.mne.viewbox.viewRange()[1], None)
        labels = self.tickStrings(values[0][1], None, None)

        return labels


class BaseScrollBar(QScrollBar):
    """Base Class for scrolling directly to the clicked position."""

    def __init__(self, parent=None):
        super().__init__(parent)

    def mousePressEvent(self, event):
        """Customize mouse click events.

        Taken from: https://stackoverflow.com/questions/29710327/
        how-to-override-qscrollbar-onclick-default-behaviour
        """
        if event.button() == Qt.LeftButton:
            opt = QStyleOptionSlider()
            self.initStyleOption(opt)
            control = self.style().hitTestComplexControl(
                QStyle.CC_ScrollBar, opt,
                event.pos(), self)
            if (control == QStyle.SC_ScrollBarAddPage or
                    control == QStyle.SC_ScrollBarSubPage):
                # scroll here
                gr = self.style().subControlRect(QStyle.CC_ScrollBar,
                                                 opt,
                                                 QStyle.SC_ScrollBarGroove,
                                                 self)
                sr = self.style().subControlRect(QStyle.CC_ScrollBar,
                                                 opt,
                                                 QStyle.SC_ScrollBarSlider,
                                                 self)
                if self.orientation() == Qt.Horizontal:
                    pos = event.pos().x()
                    sliderLength = sr.width()
                    sliderMin = gr.x()
                    sliderMax = gr.right() - sliderLength + 1
                    if (self.layoutDirection() == Qt.RightToLeft):
                        opt.upsideDown = not opt.upsideDown
                else:
                    pos = event.pos().y()
                    sliderLength = sr.height()
                    sliderMin = gr.y()
                    sliderMax = gr.bottom() - sliderLength + 1
                self.setValue(QStyle.sliderValueFromPosition(
                    self.minimum(), self.maximum(), pos - sliderMin,
                                                    sliderMax - sliderMin,
                    opt.upsideDown))
                return

        return super().mousePressEvent(event)


class TimeScrollBar(BaseScrollBar):
    """Scrolls through time."""

    def __init__(self, mne):
        super().__init__(Qt.Horizontal)
        self.mne = mne
        self.step_factor = None

        self.setMinimum(0)
        self.setSingleStep(1)
        self.setPageStep(self.mne.tsteps_per_window)
        self._update_duration()
        self.setFocusPolicy(Qt.WheelFocus)
        # Because valueChanged is needed (captures every input to scrollbar,
        # not just sliderMoved), there has to be made a differentiation
        # between internal and external changes.
        self.external_change = False
        self.valueChanged.connect(self._time_changed)

    def _time_changed(self, value):
        if not self.external_change:
            value /= self.step_factor
            self.mne.plt.setXRange(value, value + self.mne.duration,
                                   padding=0)

    def update_value(self, value):
        """Update value of the ScrollBar."""
        # Mark change as external to avoid setting
        # XRange again in _time_changed.
        self.external_change = True
        self.setValue(int(value * self.step_factor))
        self.external_change = False
        self._update_duration()

    def _update_duration(self):
        new_step_factor = self.mne.tsteps_per_window / self.mne.duration
        if new_step_factor != self.step_factor:
            self.step_factor = new_step_factor
            new_maximum = int((self.mne.xmax - self.mne.duration)
                              * self.step_factor)
            self.setMaximum(new_maximum)

    def keyPressEvent(self, event):
        """Customize key press events."""
        # Let main handle the keypress
        event.ignore()


class ChannelScrollBar(BaseScrollBar):
    """Scrolls through channels."""

    def __init__(self, mne):
        super().__init__(Qt.Vertical)
        self.mne = mne

        self.setMinimum(0)
        self._update_nchan()
        self.setSingleStep(1)
        self.setFocusPolicy(Qt.WheelFocus)
        # Because valueChanged is needed (captures every input to scrollbar,
        # not just sliderMoved), there has to be made a differentiation
        # between internal and external changes.
        self.external_change = False
        self.valueChanged.connect(self._channel_changed)

    def _channel_changed(self, value):
        value = min(value, self.mne.ymax - self.mne.n_channels)
        if not self.external_change:
            self.mne.plt.setYRange(value, value + self.mne.n_channels + 1,
                                   padding=0)

    def update_value(self, value):
        """Update value of the ScrollBar."""
        # Mark change as external to avoid setting YRange again in
        # _channel_changed.
        self.external_change = True
        self.setValue(value)
        self.external_change = False
        self._update_nchan()

    def _update_nchan(self):
        self.setPageStep(self.mne.n_channels)
        self.setMaximum(self.mne.ymax - self.mne.n_channels - 1)

    def keyPressEvent(self, event):
        """Customize key press events."""
        # Let main handle the keypress
        event.ignore()


class OverviewBar(QLabel):
    """
    Provides overview over channels and current visible range.

    Has different modes:
    - channels: Display channel-types
    - zscore: Display channel-wise zscore across time
    """

    def __init__(self, browser):
        super().__init__()
        self.browser = browser
        self.mne = browser.mne
        self.bg_img = None
        # Set minimum Size to 1/10 of display size
        min_h = int(QApplication.desktop().screenGeometry().height() / 10)
        self.setMinimumSize(1, 1)
        self.setFixedHeight(min_h)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.setStyleSheet("QLabel {background-color : white}")
        self.set_overview()

    def paintEvent(self, event):
        """Customize painting of this label."""
        super().paintEvent(event)

        painter = QPainter(self)

        # Paint bad-channels
        for line_idx, ch_idx in enumerate(self.mne.ch_order):
            if self.mne.ch_names[ch_idx] in self.mne.info['bads']:
                painter.setPen(mkColor(self.mne.ch_color_bad))
                start = self._mapFromData(0, line_idx)
                stop = self._mapFromData(self.mne.inst.times[-1], line_idx)
                painter.drawLine(start, stop)

        # Paint Annotations
        for annot in self.mne.inst.annotations:
            des = annot['description']
            if self.mne.visible_annotations[des]:
                plot_onset = _sync_onset(self.mne.inst, annot['onset'])
                duration = annot['duration']
                color_name = self.mne.annotation_segment_colors[des]
                color = mkColor(color_name)
                color.setAlpha(200)
                painter.setPen(color)
                painter.setBrush(color)
                top_left = self._mapFromData(plot_onset, 0)
                bottom_right = self._mapFromData(plot_onset + duration,
                                                 len(self.mne.ch_order))

                painter.drawRect(QRectF(top_left, bottom_right))

        # Paint view range
        view_pen = QPen(mkColor('g'))
        view_pen.setWidth(2)
        painter.setPen(view_pen)
        painter.setBrush(Qt.NoBrush)  # Clear previous brush
        top_left = self._mapFromData(self.mne.t_start, self.mne.ch_start)
        bottom_right = self._mapFromData(self.mne.t_start
                                         + self.mne.duration,
                                         self.mne.ch_start
                                         + self.mne.n_channels)
        painter.drawRect(QRectF(top_left, bottom_right))

    def _set_range_from_pos(self, pos):
        x, y = self._mapToData(pos)
        if x and y:
            # Move middle of view range to click position
            x = x - self.mne.duration / 2
            y = y - self.mne.n_channels / 2
            self.mne.plt.setXRange(x, x + self.mne.duration, padding=0)
            self.mne.plt.setYRange(y, y + self.mne.n_channels + 1, padding=0)

    def mousePressEvent(self, event):
        """Customize mouse press events."""
        self._set_range_from_pos(event.pos())

    def mouseMoveEvent(self, event):
        """Customize mouse move events."""
        self._set_range_from_pos(event.pos())

    def _fit_bg_img(self):
        # Resize Pixmap
        if self.bg_img:
            p = QPixmap.fromImage(self.bg_img)
            p = p.scaled(self.width(), self.height(),
                         Qt.IgnoreAspectRatio)
            self.setPixmap(p)

    def resizeEvent(self, event):
        """Customize resize event."""
        super().resizeEvent(event)

        self._fit_bg_img()

    def set_overview(self):
        """Set the background-image for the selected overview-mode."""
        # Add Overview-Pixmap
        if self.mne.overview_mode == 'channels' or not self.mne.preload:
            channel_rgba = np.empty((len(self.mne.ch_order),
                                     2, 4))
            for line_idx, ch_idx in enumerate(self.mne.ch_order):
                ch_type = self.mne.ch_types[ch_idx]
                color = mkColor(self.mne.ch_color_dict[ch_type])
                channel_rgba[line_idx, :] = color.getRgb()

            channel_rgba = np.require(channel_rgba, np.uint8, 'C')
            self.bg_img = QImage(channel_rgba,
                                 channel_rgba.shape[1],
                                 channel_rgba.shape[0],
                                 QImage.Format_RGBA8888)
            self.setPixmap(QPixmap.fromImage(self.bg_img))

        elif self.mne.overview_mode == 'zscore' \
                and hasattr(self.mne, 'zscore_rgba'):
            self.bg_img = QImage(self.mne.zscore_rgba,
                                 self.mne.zscore_rgba.shape[1],
                                 self.mne.zscore_rgba.shape[0],
                                 QImage.Format_RGBA8888)
            self.setPixmap(QPixmap.fromImage(self.bg_img))

        self._fit_bg_img()

    def _mapFromData(self, x, y):
        # Include padding from black frame
        point_x = self.width() * x / self.mne.inst.times[-1]
        point_y = self.height() * y / len(self.mne.ch_order)

        return QPointF(point_x, point_y)

    def _mapToData(self, point):
        # Include padding from black frame
        time_idx = int(len(self.mne.inst.times) * point.x() / self.width())
        if time_idx < len(self.mne.inst.times):
            x = self.mne.inst.times[time_idx]
            y = len(self.mne.ch_order) * point.y() / self.height()
        else:
            x, y = None, None

        return x, y


class RawViewBox(ViewBox):
    """PyQtGraph-Wrapper for interaction with the View."""

    def __init__(self, main):
        super().__init__(invertY=True)
        self.enableAutoRange(enable=False, x=False, y=False)
        self.main = main
        self.mne = main.mne
        self._drag_start = None
        self._drag_region = None

    def mouseDragEvent(self, event, axis=None):
        """Customize mouse drag events."""
        event.accept()

        if event.button() == Qt.LeftButton \
                and self.mne.annotation_mode:
            if self.mne.current_description:
                description = self.mne.current_description
                if event.isStart():
                    self._drag_start = self.mapSceneToView(
                        event.scenePos()).x()
                    self._drag_region = AnnotRegion(self.mne,
                                                    description=description,
                                                    values=(self._drag_start,
                                                            self._drag_start))
                    self.mne.plt.addItem(self._drag_region)
                    self.mne.plt.addItem(self._drag_region.label_item)
                elif event.isFinish():
                    drag_stop = self.mapSceneToView(event.scenePos()).x()
                    self._drag_region.setRegion((self._drag_start, drag_stop))
                    plot_onset = min(self._drag_start, drag_stop)
                    duration = abs(self._drag_start - drag_stop)
                    self.main._add_annotation(plot_onset, duration,
                                              region=self._drag_region)
                else:
                    self._drag_region.setRegion((self._drag_start,
                                                 self.mapSceneToView(
                                                     event.scenePos()).x()))
            elif event.isFinish():
                QMessageBox.warning(self.main, 'No description!',
                                    'No description is given, add one!')

    def mouseClickEvent(self, event):
        """Customize mouse click events."""
        # If we want the context-menu back, uncomment following line
        # super().mouseClickEvent(event)
        if event.button() == Qt.LeftButton:
            self.main._add_vline(self.mapSceneToView(event.scenePos()).x())
        elif event.button() == Qt.RightButton:
            self.main._remove_vline()

    def wheelEvent(self, ev, axis=None):
        """Customize mouse wheel/trackpad-scroll events."""
        ev.accept()
        scroll = -1 * ev.delta() / 120
        if ev.orientation() == Qt.Horizontal:
            self.main.hscroll(scroll * 10)
        elif ev.orientation() == Qt.Vertical:
            self.main.vscroll(scroll)


class VLineLabel(InfLineLabel):
    """Label of the vline displaying the time."""

    def __init__(self, vline):
        super().__init__(vline, text='{value:.3f} s', position=0.975,
                         fill='g', color='b', movable=True)
        self.vline = vline
        self.cursorOffset = None

    def mouseDragEvent(self, ev):
        """Customize mouse drag events."""
        if self.movable and ev.button() == Qt.LeftButton:
            if ev.isStart():
                self.vline.moving = True
                self.cursorOffset = (self.vline.pos() -
                                     self.mapToView(ev.buttonDownPos()))
            ev.accept()

            if not self.vline.moving:
                return

            self.vline.setPos(self.cursorOffset + self.mapToView(ev.pos()))
            self.vline.sigDragged.emit(self)
            if ev.isFinish():
                self.vline.moving = False
                self.vline.sigPositionChangeFinished.emit(self)


class VLine(InfiniteLine):
    """Marker to be placed inside the Data-Trace-Plot."""

    def __init__(self, pos, bounds):
        super().__init__(pos, pen='g', hoverPen='y',
                         movable=True, bounds=bounds)
        self.line = VLineLabel(self)


class HelpDialog(QDialog):
    """Shows all keyboard-shortcuts."""

    def __init__(self, main):
        super().__init__(main)
        self.mne = main.mne
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self._init_ui()
        self.open()

    def _init_ui(self):
        layout = QFormLayout()
        for key in self.mne.keyboard_shortcuts:
            key_dict = self.mne.keyboard_shortcuts[key]
            if 'alias' in key_dict:
                key = key_dict['alias']
            for idx, key_des in enumerate(key_dict['description']):
                if 'modifier' in key_dict:
                    mod = key_dict['modifier'][idx]
                    if mod is not None:
                        key = mod + ' + ' + key
                layout.addRow(key, QLabel(key_des))
        self.setLayout(layout)


class AnnotRegion(LinearRegionItem):
    """Graphics-Oobject for Annotations."""

    regionChangeFinished = pyqtSignal(object)
    gotSelected = pyqtSignal(object)
    removeRequested = pyqtSignal(object)

    def __init__(self, mne, description, values):

        super().__init__(values=values, orientation='vertical',
                         movable=True, swapMode='sort')
        # Set default z-value to 0 to be behind other items in scene
        self.setZValue(0)

        self.sigRegionChangeFinished.connect(self._region_changed)
        self.mne = mne
        self.description = description
        self.old_onset = values[0]
        self.selected = False

        self.label_item = TextItem(text=description, anchor=(0.5, 0.5))
        self.label_item.setFont(QFont('AnyStyle', 10, QFont.Bold))
        self.sigRegionChanged.connect(self.update_label_pos)

        self.update_color()

    def _region_changed(self):
        self.regionChangeFinished.emit(self)
        self.old_onset = self.getRegion()[0]

    def update_color(self):
        """Update color of annotation-region."""
        color = self.mne.annotation_segment_colors[self.description]
        color = mkColor(color)
        hover_color = mkColor(color)
        text_color = mkColor(color)
        color.setAlpha(75)
        hover_color.setAlpha(150)
        text_color.setAlpha(255)
        self.setBrush(color)
        self.setHoverBrush(hover_color)
        self.label_item.setColor(text_color)
        self.update()

    def update_description(self, description):
        """Update description of annoation-region."""
        self.description = description
        self.label_item.setText(description)
        self.label_item.update()

    def update_visible(self, visible):
        """Update if annotation-region is visible."""
        self.setVisible(visible)
        self.label_item.setVisible(visible)

    def paint(self, p, *args):
        """Customize painting of annotation-region (add selection-rect)."""
        super().paint(p, *args)

        if self.selected:
            # Draw selection rectangle
            p.setBrush(mkBrush(None))
            p.setPen(mkPen(color='c', width=3))
            p.drawRect(self.boundingRect())

    def remove(self):
        """Remove annotation-region."""
        self.removeRequested.emit(self)
        vb = self.mne.viewbox
        if vb and self.label_item in vb.addedItems:
            vb.removeItem(self.label_item)

    def select(self, selected):
        """Update select-state of annotation-region."""
        self.selected = selected
        if selected:
            self.gotSelected.emit(self)
        self.update()

    def mouseClickEvent(self, event):
        """Customize mouse click events."""
        event.accept()
        if event.button() == Qt.LeftButton and self.movable:
            self.select(True)
        elif event.button() == Qt.RightButton and self.movable:
            self.remove()

    def update_label_pos(self):
        """Update position of description-label from annotation-region."""
        rgn = self.getRegion()
        vb = self.mne.viewbox
        if vb:
            ymax = vb.viewRange()[1][1]
            self.label_item.setPos(sum(rgn) / 2, ymax - 0.3)


class AnnotationDock(QDockWidget):
    """Dock-Window for Management of annotations."""

    def __init__(self, main):
        super().__init__('Annotations')
        self.main = main
        self.mne = main.mne
        self._init_ui()

        self.setFeatures(QDockWidget.DockWidgetMovable |
                         QDockWidget.DockWidgetFloatable)

    def _init_ui(self):
        widget = QWidget()
        layout = QHBoxLayout()
        layout.setAlignment(Qt.AlignLeft)

        self.description_cmbx = QComboBox()
        self.description_cmbx.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.description_cmbx.activated.connect(self._description_changed)
        self._update_description_cmbx()
        layout.addWidget(self.description_cmbx)

        add_bt = QPushButton('Add Description')
        add_bt.clicked.connect(self._add_description)
        layout.addWidget(add_bt)

        rm_bt = QPushButton('Remove Description')
        rm_bt.clicked.connect(self._remove_description)
        layout.addWidget(rm_bt)

        edit_bt = QPushButton('Edit Description')
        edit_bt.clicked.connect(self._edit_description)
        layout.addWidget(edit_bt)

        color_bt = QPushButton('Edit Color')
        color_bt.clicked.connect(self._set_color)
        layout.addWidget(color_bt)

        select_bt = QPushButton('Select Visible')
        select_bt.clicked.connect(self._select_annotations)
        layout.addWidget(select_bt)

        # Determine reasonable time decimals from sampling frequency.
        time_decimals = int(np.ceil(np.log10(self.mne.info['sfreq'])))

        layout.addWidget(QLabel('Start:'))
        self.start_bx = QDoubleSpinBox()
        self.start_bx.setDecimals(time_decimals)
        self.start_bx.editingFinished.connect(self._start_changed)
        layout.addWidget(self.start_bx)

        layout.addWidget(QLabel('Stop:'))
        self.stop_bx = QDoubleSpinBox()
        self.stop_bx.setDecimals(time_decimals)
        self.stop_bx.editingFinished.connect(self._stop_changed)
        layout.addWidget(self.stop_bx)

        widget.setLayout(layout)
        self.setWidget(widget)

    def _add_description_to_cmbx(self, description):
        color_pixmap = QPixmap(25, 25)
        color = mkColor(self.mne.annotation_segment_colors[description])
        color.setAlpha(75)
        color_pixmap.fill(color)
        color_icon = QIcon(color_pixmap)
        self.description_cmbx.addItem(color_icon, description)

    def _add_description(self):
        new_description, ok = QInputDialog.getText(self,
                                                   'Set new description!',
                                                   'New description: ')
        if ok and new_description \
                and new_description not in self.mne.new_annotation_labels:
            self.mne.new_annotation_labels.append(new_description)
            self.mne.visible_annotations[new_description] = True
            self.main._setup_annotation_colors()
            self._add_description_to_cmbx(new_description)
        self.mne.current_description = self.description_cmbx.currentText()

    def _edit_description(self):
        curr_des = self.description_cmbx.currentText()

        # This is a inline approach of creating the dialog and thus preventing
        # an additional class.
        def get_edited_values():
            new_des = input_w.text()
            if mode_cmbx:
                mode = mode_cmbx.currentText()
            else:
                mode = 'group'
            if new_des:
                if mode == 'group' or self.mne.selected_region is None:
                    edit_regions = [r for r in self.mne.regions
                                    if r.description == curr_des]
                    for ed_region in edit_regions:
                        idx = self.main._get_onset_idx(
                            ed_region.getRegion()[0])
                        self.mne.inst.annotations.description[idx] = new_des
                        ed_region.update_description(new_des)
                    self.mne.new_annotation_labels.remove(curr_des)
                    self.mne.new_annotation_labels = \
                        self.main._get_annotation_labels()
                    self.mne.visible_annotations[new_des] = \
                        self.mne.visible_annotations.pop(curr_des)
                    self.mne.annotation_segment_colors[new_des] = \
                        self.mne.annotation_segment_colors.pop(curr_des)
                else:
                    idx = self.main._get_onset_idx(
                        self.mne.selected_region.getRegion()[0])
                    self.mne.inst.annotations.description[idx] = new_des
                    self.mne.selected_region.update_description(new_des)
                    if new_des not in self.mne.new_annotation_labels:
                        self.mne.new_annotation_labels.append(new_des)
                    self.mne.visible_annotations[new_des] = \
                        self.mne.visible_annotations[curr_des]
                    self.mne.annotation_segment_colors[new_des] = \
                        self.mne.annotation_segment_colors[curr_des]
                    if curr_des not in \
                            self.mne.inst.annotations.description:
                        self.mne.new_annotation_labels.remove(curr_des)
                        self.mne.visible_annotations.pop(curr_des)
                        self.mne.annotation_segment_colors.pop(curr_des)
                self.mne.current_description = new_des
                self.main._setup_annotation_colors()
                self._update_description_cmbx()
                self._update_regions_colors()

            edit_dlg.close()

        if len(self.mne.inst.annotations.description) > 0:
            edit_dlg = QDialog()
            layout = QVBoxLayout()
            if self.mne.selected_region:
                mode_cmbx = QComboBox()
                mode_cmbx.addItems(['group', 'current'])
                layout.addWidget(QLabel('Edit Scope:'))
                layout.addWidget(mode_cmbx)
            else:
                mode_cmbx = None
            layout.addWidget(QLabel(f'Change "{curr_des}" to:'))
            input_w = QLineEdit()
            layout.addWidget(input_w)
            bt_layout = QHBoxLayout()
            ok_bt = QPushButton('Ok')
            ok_bt.clicked.connect(get_edited_values)
            bt_layout.addWidget(ok_bt)
            cancel_bt = QPushButton('Cancel')
            cancel_bt.clicked.connect(edit_dlg.close)
            bt_layout.addWidget(cancel_bt)
            layout.addLayout(bt_layout)
            edit_dlg.setLayout(layout)
            edit_dlg.exec()
        else:
            QMessageBox.information(self, 'No Annotations!',
                                    'There are no annotations yet to edit!')

    def _remove_description(self):
        rm_description = self.description_cmbx.currentText()
        existing_annot = list(self.mne.inst.annotations.description).count(
            rm_description)
        if existing_annot > 0:
            ans = QMessageBox.question(self,
                                       f'Remove annotations '
                                       f'with {rm_description}?',
                                       f'There exist {existing_annot} '
                                       f'annotations with '
                                       f'"{rm_description}".\n'
                                       f'Do you really want to remove them?')
            if ans == QMessageBox.Yes:
                rm_idxs = np.where(
                    self.mne.inst.annotations.description == rm_description)
                for idx in rm_idxs:
                    self.mne.inst.annotations.delete(idx)
                for rm_region in [r for r in self.mne.regions
                                  if r.description == rm_description]:
                    rm_region.remove()

                # Remove from descriptions
                self.mne.new_annotation_labels.remove(rm_description)
                self._update_description_cmbx()

                # Remove from visible annotations
                self.mne.visible_annotations.pop(rm_description)

                # Remove from color-mapping
                if rm_description in self.mne.annotation_segment_colors:
                    self.mne.annotation_segment_colors.pop(rm_description)

                # Set first description in Combo-Box to current description
                if self.description_cmbx.count() > 0:
                    self.description_cmbx.setCurrentIndex(0)
                    self.mne.current_description = \
                        self.description_cmbx.currentText()

    def _select_annotations(self):
        def _set_visible_region(state, description):
            self.mne.visible_annotations[description] = bool(state)

        def _select_all():
            for chkbx in chkbxs:
                chkbx.setChecked(True)

        def _clear_all():
            for chkbx in chkbxs:
                chkbx.setChecked(False)

        select_dlg = QDialog(self)
        chkbxs = list()
        layout = QVBoxLayout()
        layout.addWidget(QLabel('Select visible labels:'))

        # Add descriptions to scroll-area to be scalable.
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout()

        for des in self.mne.visible_annotations:
            chkbx = QCheckBox(des)
            chkbx.setChecked(self.mne.visible_annotations[des])
            chkbx.stateChanged.connect(partial(_set_visible_region,
                                               description=des))
            chkbxs.append(chkbx)
            scroll_layout.addWidget(chkbx)

        scroll_widget.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_widget)
        layout.addWidget(scroll_area)

        bt_layout = QGridLayout()

        all_bt = QPushButton('All')
        all_bt.clicked.connect(_select_all)
        bt_layout.addWidget(all_bt, 0, 0)

        clear_bt = QPushButton('Clear')
        clear_bt.clicked.connect(_clear_all)
        bt_layout.addWidget(clear_bt, 0, 1)

        ok_bt = QPushButton('Ok')
        ok_bt.clicked.connect(select_dlg.close)
        bt_layout.addWidget(ok_bt, 1, 0, 1, 2)

        layout.addLayout(bt_layout)

        select_dlg.setLayout(layout)
        select_dlg.exec()

        self._update_regions_visible()

    def _description_changed(self, descr_idx):
        new_descr = self.description_cmbx.itemText(descr_idx)
        self.mne.current_description = new_descr

    def _start_changed(self):
        start = self.start_bx.value()
        sel_region = self.mne.selected_region
        if sel_region:
            stop = sel_region.getRegion()[1]
            if start < stop:
                sel_region.setRegion((start, stop))
            else:
                QMessageBox.warning(self, 'Invalid value!',
                                    'Start can\'t be bigger or equal to Stop!')
                self.start_bx.setValue(sel_region.getRegion()[0])

    def _stop_changed(self):
        stop = self.stop_bx.value()
        sel_region = self.mne.selected_region
        if sel_region:
            start = sel_region.getRegion()[0]
            if start < stop:
                sel_region.setRegion((start, stop))
            else:
                QMessageBox.warning(self, 'Invalid value!',
                                    'Stop can\'t be smaller '
                                    'or equal to Start!')
                self.stop_bx.setValue(sel_region.getRegion()[1])

    def _set_color(self):
        curr_descr = self.description_cmbx.currentText()
        if curr_descr in self.mne.annotation_segment_colors:
            curr_col = self.mne.annotation_segment_colors[curr_descr]
        else:
            curr_col = None
        color = QColorDialog.getColor(mkColor(curr_col), self,
                                      f'Choose color for {curr_descr}!')
        if color.isValid():
            self.mne.annotation_segment_colors[curr_descr] = color
            self._update_description_cmbx()
            self._update_regions_colors()

    def update_values(self, region):
        """Update spinbox-values from region."""
        rgn = region.getRegion()
        self.description_cmbx.setCurrentText(region.description)
        self.start_bx.setValue(rgn[0])
        self.stop_bx.setValue(rgn[1])

    def _update_description_cmbx(self):
        self.description_cmbx.clear()
        descriptions = self.main._get_annotation_labels()
        for description in descriptions:
            self._add_description_to_cmbx(description)
        self.description_cmbx.setCurrentText(self.mne.current_description)

    def _update_regions_visible(self):
        for region in self.mne.regions:
            region.update_visible(
                self.mne.visible_annotations[region.description])

    def _update_regions_colors(self):
        for region in self.mne.regions:
            region.update_color()

    def reset(self):
        """Reset to default state."""
        if self.description_cmbx.count() > 0:
            self.description_cmbx.setCurrentIndex(0)
            self.mne.current_description = self.description_cmbx.currentText()
        self.start_bx.setValue(0)
        self.stop_bx.setValue(0)


class BrowserView(GraphicsView):
    """Customized View as part of GraphicsView-Framework."""

    def __init__(self, plot, **kwargs):
        super().__init__(**kwargs)
        self.setCentralItem(plot)
        self.setSizePolicy(QSizePolicy.MinimumExpanding,
                           QSizePolicy.MinimumExpanding)
        self.viewport().setAttribute(Qt.WA_AcceptTouchEvents, True)

        self.viewport().grabGesture(Qt.PinchGesture)
        self.viewport().grabGesture(Qt.SwipeGesture)

    def viewportEvent(self, event):
        """Customize viewportEvent for touch-gestures (WIP)."""
        if event.type() in [QEvent.TouchBegin, QEvent.TouchUpdate,
                            QEvent.TouchEnd]:
            if event.touchPoints() == 2:
                pass
        elif event.type() == QEvent.Gesture:
            print('Gesture')
        return super().viewportEvent(event)

    def mouseMoveEvent(self, ev):
        """Customize MouseMoveEvent."""
        super().mouseMoveEvent(ev)
        self.sigSceneMouseMoved.emit(ev.pos())


class LoadRunnerSignals(QObject):
    """Signals for the LoadRunner (QRunnables aren't QObjects)."""

    loadProgress = pyqtSignal(int)
    processText = pyqtSignal(str)
    loadingFinished = pyqtSignal()


class LoadRunner(QRunnable):
    """A QRunnable for preloading in a separate QThread."""

    def __init__(self, browser):
        super().__init__()
        self.browser = browser
        self.mne = browser.mne
        self.sigs = LoadRunnerSignals()

    def run(self):
        """Load and process data in a separate QThread."""
        # Split data loading into 10 chunks to show user progress.
        # Testing showed that e.g. n_chunks=100 extends loading time
        # (at least for the sample dataset)
        # because of the frequent gui-update-calls.
        # Thus n_chunks = 10 should suffice.
        data = None
        times = None
        n_chunks = 10
        if not self.mne.is_epochs:
            chunk_size = len(self.browser.mne.inst) // n_chunks
            for n in range(n_chunks):
                start = n * chunk_size
                if n == n_chunks - 1:
                    # Get last chunk which may be larger due to rounding above
                    stop = None
                else:
                    stop = start + chunk_size
                # Load data
                data_chunk, times_chunk = self.browser._load_data(start, stop)
                if data is None:
                    data = data_chunk
                    times = times_chunk
                else:
                    data = np.concatenate((data, data_chunk), axis=1)
                    times = np.concatenate((times, times_chunk), axis=0)
                self.sigs.loadProgress.emit(n + 1)
        else:
            self.browser._load_data()
            self.sigs.loadProgress.emit(n_chunks)

        picks = self.browser.mne.ch_order
        # Deactive remove dc because it will be removed for visible range
        stashed_remove_dc = self.mne.remove_dc
        self.mne.remove_dc = False
        data = self.browser._process_data(data, start, stop, picks,
                                          self.sigs)
        self.mne.remove_dc = stashed_remove_dc

        # Invert Data to be displayed from top on inverted Y-Axis.
        data *= -1

        self.browser.mne.global_data = data
        self.browser.mne.global_times = times

        # Calculate Z-Scores
        if self.mne.overview_mode == 'zscore':
            self.sigs.processText.emit('Calculating Z-Scores...')
            self.browser._get_zscore(data)

        self.sigs.loadingFinished.emit()


class _PGMetaClass(type(BrowserBase), type(QMainWindow)):
    """Class is necessary to prevent a metaclass conflict.

    The conflict arises due to the different types of QMainWindow and
    BrowserBase.
    """

    pass


class PyQtGraphBrowser(BrowserBase, QMainWindow, metaclass=_PGMetaClass):
    """A PyQtGraph-backend for 2D data browsing."""

    def __init__(self, **kwargs):
        """
        Those are pyqtgraph-specific parameters.

        They might be added to the .plot()-call or will be integrated
        into a settings-gui.

        Parameters
        ----------
        ds : int | str
            The downsampling-factor. Either 'auto' to get the downsampling-rate
            from the visible range or an integer (1 means no downsampling).
            Defaults to 'auto'.
        ds_method : str
            The downsampling-method to use (from pyqtgraph).
            See here under "Optimization-Keywords" for more detail:
            https://pyqtgraph.readthedocs.io/en/latest/graphicsItems/
            plotdataitem.html?#
        ds_chunk_size : int | None
            Chunk size for downsampling. No chunking if None (default).
        antialiasing : bool
            Enable Antialiasing.
        use_opengl : bool
            Use OpenGL.
        enable_ds_cache : bool
            If True cache the downsampled arrays inside RawCurveItems
            per downsampling-factor.
        tsteps_per_window : int
            Set how many single scrolling-steps are done in time
            for the shown time-window.
        check_nan : bool
            If to check for NaN-values.
        preload : str
            If True, preprocessing steps are applied on all data
            and are repeated only if necessary. If False (default),
            preprocessing is applied only on the visible data.
        overview_mode : str | None
            Set the mode for the display of an overview over the data.
            Currently available is "zscore" to display the zscore for
            each channel across time. This only works if preload=True.
            Defaults to "zscore".
        """
        self.pg_kwarg_defaults = dict(duration=20,
                                      n_channels=30,
                                      highpass=None,
                                      lowpass=None,
                                      ds='auto',
                                      ds_method='peak',
                                      antialiasing=False,
                                      use_opengl=False,
                                      enable_ds_cache=True,
                                      tsteps_per_window=100,
                                      check_nan=False,
                                      remove_dc=True,
                                      preload=True,
                                      show_overview_bar=True,
                                      overview_mode='channels')
        for kw in [k for k in self.pg_kwarg_defaults if k not in kwargs]:
            kwargs[kw] = self.pg_kwarg_defaults[kw]

        BrowserBase.__init__(self, **kwargs)
        QMainWindow.__init__(self)

        # Initialize attributes which are only used by pyqtgraph, not by
        # matplotlib and add them to MNEBrowseParams.
        self.mne.ds_cache = dict()
        self.mne.data_preloaded = False

        # Add Load-Progressbar for loading in a thread
        self.mne.load_prog_label = QLabel('Loading...')
        self.statusBar().addWidget(self.mne.load_prog_label)
        self.mne.load_prog_label.hide()
        self.mne.load_progressbar = QProgressBar()
        # Set to n_chunks of LoadRunner
        self.mne.load_progressbar.setMaximum(10)
        self.statusBar().addWidget(self.mne.load_progressbar, stretch=1)
        self.mne.load_progressbar.hide()

        self.mne.traces = list()
        self.mne.scale_factor = 1
        self.mne.butterfly_type_order = [tp for tp in
                                         _DATA_CH_TYPES_ORDER_DEFAULT
                                         if tp in self.mne.ch_types]

        # Initialize annotations (ToDo: Adjust to MPL)
        self.mne.annotation_mode = False
        self.mne.new_annotation_labels = self._get_annotation_labels()
        if len(self.mne.new_annotation_labels) > 0:
            self.mne.current_description = self.mne.new_annotation_labels[0]
        else:
            self.mne.current_description = None
        self._setup_annotation_colors()
        self.mne.regions = list()
        self.mne.selected_region = None

        setConfigOption('antialias', self.mne.antialiasing)

        # Start preloading if enabled
        if self.mne.preload:
            self._preload_in_thread()

        # Create centralWidget and layout
        widget = QWidget()
        layout = QGridLayout()

        # Initialize Axis-Items
        time_axis = TimeAxis(self.mne)
        time_axis.setLabel(text='Time', units='s')
        channel_axis = ChannelAxis(self)
        viewbox = RawViewBox(self)
        vars(self.mne).update(time_axis=time_axis, channel_axis=channel_axis,
                              viewbox=viewbox)

        # Initialize data (needed in RawTraceItem.update_data).
        # This could be optimized.
        self._update_data()

        # Initialize Trace-Plot
        plt = PlotItem(viewBox=viewbox,
                       axisItems={'bottom': time_axis, 'left': channel_axis})
        # Hide AutoRange-Button
        plt.hideButtons()
        # Configure XY-Range
        self.mne.xmax = self.mne.inst.times[-1]
        plt.setXRange(0, self.mne.duration, padding=0)
        # Add one empty line as padding at top (y=0).
        # Negative Y-Axis to display channels from top.
        self.mne.ymax = len(self.mne.ch_order) + 1
        plt.setYRange(0, self.mne.n_channels + 1, padding=0)
        plt.setLimits(xMin=0, xMax=self.mne.xmax,
                      yMin=0, yMax=self.mne.ymax)
        # Connect Signals from PlotItem
        plt.sigXRangeChanged.connect(self._xrange_changed)
        plt.sigYRangeChanged.connect(self._yrange_changed)
        vars(self.mne).update(plt=plt)

        # Add traces
        for ch_idx in self.mne.picks:
            self._add_trace(ch_idx)

        # Check for OpenGL
        try:
            import OpenGL
            logger.info(f'Using pyopengl with version {OpenGL.__version__}')
        except ModuleNotFoundError:
            logger.warning('pyopengl was not found on this device.\n'
                           'Defaulting to plot without OpenGL with reduced '
                           'performance.')
            self.mne.use_opengl = False

        # Initialize BrowserView (inherits QGraphicsView)
        view = BrowserView(plt, background='w',
                           useOpenGL=self.mne.use_opengl)
        layout.addWidget(view, 0, 0)

        # Initialize Scroll-Bars
        ax_hscroll = TimeScrollBar(self.mne)
        layout.addWidget(ax_hscroll, 1, 0)

        ax_vscroll = ChannelScrollBar(self.mne)
        layout.addWidget(ax_vscroll, 0, 1)

        # OverviewBar
        overview_bar = OverviewBar(self)
        layout.addWidget(overview_bar, 2, 0)

        widget.setLayout(layout)
        self.setCentralWidget(widget)

        # Initialize Annotation-Dock
        fig_annotation = AnnotationDock(self)
        self.addDockWidget(Qt.TopDockWidgetArea, fig_annotation)
        fig_annotation.setVisible(False)
        vars(self.mne).update(fig_annotation=fig_annotation)

        # Add annotations as regions
        for annot in self.mne.inst.annotations:
            plot_onset = _sync_onset(self.mne.inst, annot['onset'])
            duration = annot['duration']
            description = annot['description']
            self._add_region(plot_onset, duration, description)

        # Initialize annotations
        self._change_annot_mode()

        # Initialize VLine
        self.mne.vline = None
        self.mne.vline_visible = False

        # Initialize crosshair (as in pyqtgraph example)
        self.mne.crosshair_enabled = False
        self.mne.crosshair_h = None
        self.mne.crosshair_v = None
        view.sigSceneMouseMoved.connect(self._mouse_moved)

        # Initialize Toolbar
        toolbar = self.addToolBar('Tools')

        adecr_time = QAction('-Time', parent=self)
        adecr_time.triggered.connect(partial(self.change_duration,
                                             -self.mne.tsteps_per_window / 10))
        toolbar.addAction(adecr_time)

        aincr_time = QAction('+Time', parent=self)
        aincr_time.triggered.connect(partial(self.change_duration,
                                             self.mne.tsteps_per_window / 10))
        toolbar.addAction(aincr_time)

        adecr_nchan = QAction('-Channels', parent=self)
        adecr_nchan.triggered.connect(partial(self.change_nchan, -10))
        toolbar.addAction(adecr_nchan)

        aincr_nchan = QAction('+Channels', parent=self)
        aincr_nchan.triggered.connect(partial(self.change_nchan, 10))
        toolbar.addAction(aincr_nchan)

        atoggle_annot = QAction('Toggle Annotations', parent=self)
        atoggle_annot.triggered.connect(self._toggle_annotation_fig)
        toolbar.addAction(atoggle_annot)

        ahelp = QAction('Help', parent=self)
        ahelp.triggered.connect(self._toggle_help_fig)
        toolbar.addAction(ahelp)

        # Add GUI-Elements to MNEBrowserParams-Instance
        vars(self.mne).update(
            plt=plt, view=view, ax_hscroll=ax_hscroll, ax_vscroll=ax_vscroll,
            overview_bar=overview_bar, fig_annotation=fig_annotation,
            toolbar=toolbar
        )

        # Initialize Keyboard-Shortcuts
        is_mac = platform.system() == 'Darwin'
        dur_keys = ('fn + ←', 'fn + →') if is_mac else ('Home', 'End')
        ch_keys = ('fn + ↑', 'fn + ↓') if is_mac else ('Page up', 'Page down')
        self.mne.keyboard_shortcuts = {
            'left': {
                'alias': '←',
                'qt_key': Qt.Key_Left,
                'modifier': [None, 'Ctrl'],
                'slot': self.hscroll,
                'parameter': [-10, -1],
                'description': ['Move left', 'Move left (tiny step)']
            },
            'right': {
                'alias': '→',
                'qt_key': Qt.Key_Right,
                'modifier': [None, 'Ctrl'],
                'slot': self.hscroll,
                'parameter': [10, 1],
                'description': ['Move right', 'Move left (tiny step)']
            },
            'up': {
                'alias': '↑',
                'qt_key': Qt.Key_Up,
                'modifier': [None, 'Ctrl'],
                'slot': self.vscroll,
                'parameter': [-10, -1],
                'description': ['Move up', 'Move up (tiny step)']
            },
            'down': {
                'alias': '↓',
                'qt_key': Qt.Key_Down,
                'modifier': [None, 'Ctrl'],
                'slot': self.vscroll,
                'parameter': [10, 1],
                'description': ['Move down', 'Move down (tiny step)']
            },
            'home': {
                'alias': dur_keys[0],
                'qt_key': Qt.Key_Home,
                'modifier': [None, 'Ctrl'],
                'slot': self.change_duration,
                'parameter': [-10, -1],
                'description': ['Decrease duration',
                                'Decrease duration (tiny step)']
            },
            'end': {
                'alias': dur_keys[1],
                'qt_key': Qt.Key_End,
                'modifier': [None, 'Ctrl'],
                'slot': self.change_duration,
                'parameter': [10, 1],
                'description': ['Increase duration',
                                'Increase duration (tiny step)']
            },
            'pagedown': {
                'alias': ch_keys[1],
                'qt_key': Qt.Key_PageUp,
                'modifier': [None, 'Ctrl'],
                'slot': self.change_nchan,
                'parameter': [-10, -1],
                'description': ['Decrease shown channels',
                                'Decrease shown channels (tiny step)']
            },
            'pageup': {
                'alias': ch_keys[0],
                'qt_key': Qt.Key_PageDown,
                'modifier': [None, 'Ctrl'],
                'slot': self.change_nchan,
                'parameter': [10, 1],
                'description': ['Increase shown channels',
                                'Increase shown channels (tiny step)']
            },
            '-': {
                'qt_key': Qt.Key_Minus,
                'slot': self.scale_all,
                'parameter': [0.5],
                'description': ['Decrease Scale']
            },
            '+': {
                'qt_key': Qt.Key_Plus,
                'slot': self.scale_all,
                'parameter': [2],
                'description': ['Increase Scale']
            },
            'a': {
                'qt_key': Qt.Key_A,
                'slot': self._toggle_annotation_fig,
                'description': ['Toggle Annotation-Tool']
            },
            'b': {
                'qt_key': Qt.Key_B,
                'slot': self._toggle_butterfly,
                'description': ['Toggle Annotation-Tool']
            },
            'd': {
                'qt_key': Qt.Key_D,
                'slot': self._toggle_dc,
                'description': ['Toggle DC-Correction']
            },
            'o': {
                'qt_key': Qt.Key_O,
                'slot': self._toggle_overview_bar,
                'description': ['Toggle Overview-Bar']
            },
            't': {
                'qt_key': Qt.Key_T,
                'slot': self._toggle_time_format,
                'description': ['Toggle Time-Format']
            },
            'x': {
                'qt_key': Qt.Key_X,
                'slot': self._toggle_crosshair,
                'description': ['Toggle Crosshair']
            },
            'z': {
                'qt_key': Qt.Key_Z,
                'slot': self._toggle_zenmode,
                'description': ['Toggle Zen-Mode']
            },
            '?': {
                'qt_key': Qt.Key_Question,
                'slot': self._toggle_help_fig,
                'description': ['Show Help']
            },
            'f11': {
                'qt_key': Qt.Key_F11,
                'slot': self._toggle_fullscreen,
                'description': ['Toggle Full-Screen']
            }
        }

    def _get_scale_transform(self):
        transform = QTransform()
        transform.scale(1, self.mne.scale_factor)

        return transform

    def _bad_ch_clicked(self, line):
        """Slot for bad channel click."""
        self._toggle_bad_channel(line.range_idx)

        # Update line color
        line.isbad = not line.isbad
        line._update_bad_color()

        # Update Channel-Axis
        self.mne.channel_axis.repaint()

        # Update Overview-Bar
        self.mne.overview_bar.update()

    def _add_trace(self, ch_idx):
        trace = RawTraceItem(self.mne, ch_idx)

        # Apply scaling
        transform = self._get_scale_transform()
        trace.setTransform(transform)

        # Add Item early to have access to viewBox
        self.mne.plt.addItem(trace)
        self.mne.traces.append(trace)

        trace.sigClicked.connect(lambda tr, _: self._bad_ch_clicked(tr))

    def _remove_trace(self, trace):
        self.mne.plt.removeItem(trace)
        self.mne.traces.remove(trace)

    def scale_all(self, step):
        """Scale all traces by multiplying with step."""
        self.mne.scale_factor *= step
        transform = self._get_scale_transform()

        for line in self.mne.traces:
            line.setTransform(transform)

    def hscroll(self, step):
        """Scroll horizontally by step."""
        rel_step = step * self.mne.duration / self.mne.tsteps_per_window
        # Get current range and add step to it
        xmin, xmax = [i + rel_step for i in self.mne.viewbox.viewRange()[0]]

        if xmin < 0:
            xmin = 0
            xmax = xmin + self.mne.duration
        elif xmax > self.mne.xmax:
            xmax = self.mne.xmax
            xmin = xmax - self.mne.duration

        self.mne.plt.setXRange(xmin, xmax, padding=0)

    def vscroll(self, step):
        """Scroll vertically by step."""
        # Get current range and add step to it
        ymin, ymax = [i + step for i in self.mne.viewbox.viewRange()[1]]

        if ymin < 0:
            ymin = 0
            ymax = self.mne.n_channels + 1
        elif ymax > self.mne.ymax:
            ymax = self.mne.ymax
            ymin = ymax - self.mne.n_channels - 1

        self.mne.plt.setYRange(ymin, ymax, padding=0)

    def change_duration(self, step):
        """Change duration by step."""
        rel_step = (self.mne.duration * step) / (
                self.mne.tsteps_per_window * 2)
        xmin, xmax = self.mne.viewbox.viewRange()[0]
        xmax += rel_step
        xmin -= rel_step

        if self.mne.is_epochs:
            # use the length of one epoch as duration change
            min_dur = len(self.mne.inst.times) / self.mne.info['sfreq']
        else:
            # never show fewer than 3 samples
            min_dur = 3 * np.diff(self.mne.inst.times[:2])[0]

        if xmax - xmin < min_dur:
            xmax = xmin + min_dur

        if xmax > self.mne.xmax:
            xmax = self.mne.xmax

        if xmin < 0:
            xmin = 0

        self.mne.plt.setXRange(xmin, xmax, padding=0)

    def change_nchan(self, step):
        """Change number of channels by step."""
        ymin, ymax = self.mne.viewbox.viewRange()[1]
        ymax += step
        if ymax > self.mne.ymax:
            ymax = self.mne.ymax
            ymin -= step

        if ymin < 0:
            ymin = 0

        if ymax - ymin <= 2:
            ymax = ymin + 2

        self.mne.plt.setYRange(ymin, ymax, padding=0)

    def _remove_vline(self):
        if self.mne.vline:
            if self.mne.is_epochs:
                for vline in self.mne.vline:
                    self.mne.plt.removeItem(vline)
            else:
                self.mne.plt.removeItem(self.mne.vline)

        self.mne.vline = None
        self.mne.vline_visible = False

    def _add_vline(self, pos):
        # Remove vline if already shown
        self._remove_vline()

        self.mne.vline = VLine(pos, bounds=(0, self.mne.xmax))
        self.mne.plt.addItem(self.mne.vline)
        self.mne.vline_visible = True

    def _mouse_moved(self, pos):
        """Show Crosshair if enabled at mouse move."""
        if self.mne.crosshair_enabled:
            if self.mne.plt.sceneBoundingRect().contains(pos):
                mousePoint = self.mne.viewbox.mapSceneToView(pos)
                x, y = mousePoint.x(), mousePoint.y()
                if (0 <= x <= self.mne.xmax and
                        0 <= y <= self.mne.ymax):
                    if not self.mne.crosshair_v:
                        self.mne.crosshair_v = InfiniteLine(angle=90,
                                                            movable=False,
                                                            pen='g')
                        self.mne.plt.addItem(self.mne.crosshair_v,
                                             ignoreBounds=True)
                    if not self.mne.crosshair_h:
                        self.mne.crosshair_h = InfiniteLine(angle=0,
                                                            movable=False,
                                                            pen='g')
                        self.mne.plt.addItem(self.mne.crosshair_h,
                                             ignoreBounds=True)

                    # Get ypos from trace
                    trace = [tr for tr in self.mne.traces if
                             tr.ypos - 0.5 < y < tr.ypos + 0.5]
                    if len(trace) == 1:
                        trace = trace[0]
                        idx = np.argmin(np.abs(trace.xData - x))
                        y = trace.get_ydata()[idx]

                        self.mne.crosshair_v.setPos(x)
                        self.mne.crosshair_h.setPos(y)

                        self.statusBar().showMessage(f'x={x:.3f} s, y={y:.3f}')

    def _toggle_crosshair(self):
        self.mne.crosshair_enabled = not self.mne.crosshair_enabled
        if self.mne.crosshair_v:
            self.mne.plt.removeItem(self.mne.crosshair_v)
            self.mne.crosshair_v = None
        if self.mne.crosshair_h:
            self.mne.plt.removeItem(self.mne.crosshair_h)
            self.mne.crosshair_h = None

    def _xrange_changed(self, _, xrange):
        # Update data
        self.mne.t_start = xrange[0]
        self.mne.duration = xrange[1] - xrange[0]
        self._redraw(update_data=True)

        # Update Time-Bar
        self.mne.ax_hscroll.update_value(xrange[0])

        # Update Overview-Bar
        self.mne.overview_bar.update()

    def _yrange_changed(self, _, yrange):
        if not self.mne.butterfly:
            # Update picks and data
            self.mne.ch_start = np.clip(round(yrange[0]), 0,
                                        len(self.mne.ch_order)
                                        - self.mne.n_channels)
            self.mne.n_channels = round(yrange[1] - yrange[0] - 1)
            self._update_picks()
            self._update_data()

            # Update Channel-Bar
            self.mne.ax_vscroll.update_value(self.mne.ch_start)

            # Update Overview-Bar
            self.mne.overview_bar.update()

        off_traces = [tr for tr in self.mne.traces
                      if tr.ch_idx not in self.mne.picks]
        add_idxs = [p for p in self.mne.picks
                    if p not in [tr.ch_idx for tr in self.mne.traces]]

        # Update number of traces.
        trace_diff = len(self.mne.picks) - len(self.mne.traces)

        # Remove unnecessary traces.
        if trace_diff < 0:
            # Only remove from traces not in picks.
            remove_traces = off_traces[:abs(trace_diff)]
            for trace in remove_traces:
                self._remove_trace(trace)
                off_traces.remove(trace)

        # Add new traces if necessary.
        if trace_diff > 0:
            # Make copy to avoid skipping iteration.
            idxs_copy = add_idxs.copy()
            for aidx in idxs_copy:
                self._add_trace(aidx)
                add_idxs.remove(aidx)

        # Update range_idx for traces which just shifted in y-position
        for trace in [tr for tr in self.mne.traces if tr not in off_traces]:
            trace.update_range_idx()

        # Update data of traces outside of yrange (reuse remaining trace-items)
        for trace, ch_idx in zip(off_traces, add_idxs):
            trace.set_ch_idx(ch_idx)
            trace._update_bad_color()
            trace.update_data()

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # DATA HANDLING
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    def _apply_downsampling(self):
        """
        Get ds-factor and apply ds with one of multiple methods.

        The methods are taken from PlotDataItem in pyqtgraph
        and adjusted to multi-channel data.
        """
        # Get Downsampling-Factor
        # Auto-Downsampling from pyqtgraph
        if (self.mne.ds == 'auto' and
                all([hasattr(self.mne, a) for a in ['viewbox', 'times']])):
            vb = self.mne.viewbox
            if vb is not None:
                view_range = vb.viewRect()
            else:
                view_range = None
            if view_range is not None and len(self.mne.times) > 1:
                dx = float(self.mne.times[-1] - self.mne.times[0]) / (
                        len(self.mne.times) - 1)
                if dx != 0.0:
                    x0 = view_range.left() / dx
                    x1 = view_range.right() / dx
                    width = vb.width()
                    if width != 0.0:
                        # Auto-Downsampling with 5 samples per pixel
                        self.mne.ds = int(max(1, (x1 - x0) / (width * 5)))

        if not isinstance(self.mne.ds, int):
            self.mne.ds = 1

        # Apply Downsampling
        if self.mne.ds not in [None, 1]:
            ds = self.mne.ds
            times = self.mne.times
            data = self.mne.data
            n_ch = data.shape[0]

            if self.mne.enable_ds_cache and ds in self.mne.ds_cache:
                # Caching is only activated if downsampling is applied
                # on the preloaded data.
                times, data = self.mne.ds_cache[ds]
            else:
                if self.mne.ds_method == 'subsample':
                    times = times[::ds]
                    data = data[:, ::ds]

                elif self.mne.ds_method == 'mean':
                    n = len(times) // ds
                    # start of x-values
                    # try to select a somewhat centered point
                    stx = ds // 2
                    times = times[stx:stx + n * ds:ds]
                    rs_data = data[:, n * ds].reshape(n_ch, n, ds)
                    data = rs_data.mean(axis=2)

                elif self.mne.ds_method == 'peak':
                    n = len(times) // ds
                    # start of x-values
                    # try to select a somewhat centered point
                    stx = ds // 2

                    x1 = np.empty((n, 2))
                    x1[:] = times[stx:stx + n * ds:ds, np.newaxis]
                    times = x1.reshape(n * 2)

                    y1 = np.empty((n_ch, n, 2))
                    y2 = data[:n * ds].reshape((n_ch, n, ds))
                    y1[:, :, 0] = y2.max(axis=2)
                    y1[:, :, 1] = y2.min(axis=2)
                    data = y1.reshape((n_ch, n * 2))

                # Only cache downsampled data if cache is enabled
                # (may be not with big datasets)
                if self.mne.enable_ds_cache and \
                        self.mne.preload and self.mne.data_preloaded:
                    self.mne.ds_cache[ds] = times, data

            self.mne.times, self.mne.data = times, data

    def _show_process(self, message):
        if self.mne.load_progressbar.isVisible():
            self.mne.load_progressbar.hide()
            self.mne.load_prog_label.hide()
        self.statusBar().showMessage(message)

    def _preload_finished(self):
        self.statusBar().showMessage('Loading Finished', 5)
        self.mne.data_preloaded = True

        if self.mne.overview_mode == 'zscore':
            # Show loaded overview image
            self.mne.overview_bar.set_overview()

    def _preload_in_thread(self):
        self.mne.data_preloaded = False
        # Remove previously loaded data
        if all([hasattr(self.mne, st)
                for st in ['global_data', 'global_times']]):
            del self.mne.global_data, self.mne.global_times
        # Start preload thread
        self.mne.load_progressbar.show()
        self.mne.load_prog_label.show()
        load_runner = LoadRunner(self)
        load_runner.sigs.loadProgress.connect(self.mne.
                                              load_progressbar.setValue)
        load_runner.sigs.processText.connect(self._show_process)
        load_runner.sigs.loadingFinished.connect(self._preload_finished)
        QThreadPool.globalInstance().start(load_runner)

    def _get_decim(self):
        if self.mne.decim != 1:
            self.mne.decim_data = np.ones_like(self.mne.picks)
            data_picks_mask = np.in1d(self.mne.picks, self.mne.picks_data)
            self.mne.decim_data[data_picks_mask] = self.mne.decim
            # decim can vary by channel type,
            # so compute different `times` vectors
            self.mne.decim_times = {decim_value: self.mne.times[::decim_value]
                                                 + self.mne.first_time for
                                    decim_value in
                                    set(self.mne.decim_data)}

    def _update_data(self):
        if self.mne.data_preloaded:
            # get start/stop-samples
            start, stop = self._get_start_stop()
            self.mne.times = self.mne.global_times[start:stop]
            self.mne.data = self.mne.global_data[:, start:stop]

            # remove DC locally
            if self.mne.remove_dc:
                self.mne.data = self.mne.data - \
                                self.mne.data.mean(axis=1, keepdims=True)
        else:
            super()._update_data()

            # Invert Data to be displayed from top on inverted Y-Axis.
            self.mne.data *= -1

        # Get decim
        self._get_decim()

        # Apply Downsampling (if enabled)
        self._apply_downsampling()

    def _get_zscore(self, data):
        # Reshape data to reasonable size for display
        max_pixel_width = QApplication.desktop().screenGeometry().width()
        collapse_by = data.shape[1] // max_pixel_width
        data = data[:, :max_pixel_width * collapse_by]
        data = data.reshape(data.shape[0], max_pixel_width, collapse_by)
        data = data.mean(axis=2)
        z = zscore(data, axis=1)

        zmin = np.min(z, axis=1)
        zmax = np.max(z, axis=1)

        # Convert into RGBA
        zrgba = np.empty((*z.shape, 4))
        for row_idx, row in enumerate(z):
            for col_idx, value in enumerate(row):
                if math.isnan(value):
                    value = 0
                if value == 0:
                    rgba = [0, 0, 0, 0]
                elif value < 0:
                    alpha = int(255 * value / abs(zmin[row_idx]))
                    rgba = [0, 0, 255, alpha]
                else:
                    alpha = int(255 * value / zmax[row_idx])
                    rgba = [255, 0, 0, alpha]

                zrgba[row_idx, col_idx] = rgba

        zrgba = np.require(zrgba, np.uint8, 'C')

        self.mne.zscore_rgba = zrgba

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # ANNOTATIONS
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    def _add_region(self, plot_onset, duration, description, region=None):
        if not region:
            region = AnnotRegion(self.mne, description=description,
                                 values=(plot_onset, plot_onset + duration))
            self.mne.plt.addItem(region)
            # Found no better way yet to initialize the region-labels
            self.mne.plt.addItem(region.label_item)
        region.regionChangeFinished.connect(self._region_changed)
        region.gotSelected.connect(self._region_selected)
        region.removeRequested.connect(self._remove_region)
        self.mne.viewbox.sigYRangeChanged.connect(
            region.update_label_pos)
        self.mne.regions.append(region)

        region.update_label_pos()

    def _remove_region(self, region):
        # Remove from shown regions
        if region.label_item in self.mne.viewbox.addedItems:
            self.mne.viewbox.removeItem(region.label_item)
        if region in self.mne.plt.items:
            self.mne.plt.removeItem(region)

        # Remove from all regions
        if region in self.mne.regions:
            self.mne.regions.remove(region)

        # Reset selected region
        if region == self.mne.selected_region:
            self.mne.selected_region = None

        # Remove from annotations
        idx = self._get_onset_idx(region.getRegion()[0])
        self.mne.inst.annotations.delete(idx)

        # Update Overview-Bar
        self.mne.overview_bar.update()

    def _region_selected(self, region):
        old_region = self.mne.selected_region
        # Remove selected-status from old region
        if old_region and old_region != region:
            old_region.selected = False
            old_region.update()
        self.mne.selected_region = region
        self.mne.current_description = region.description
        self.mne.fig_annotation.update_values(region)

    def _get_onset_idx(self, plot_onset):
        onset = _sync_onset(self.mne.inst, plot_onset, inverse=True)
        idx = np.where(self.mne.inst.annotations.onset == onset)
        return idx

    def _region_changed(self, region):
        rgn = region.getRegion()
        region.select(True)
        idx = self._get_onset_idx(region.old_onset)

        # Update Spinboxes of Annot-Dock
        self.mne.fig_annotation.update_values(region)

        # Change annotations
        self.mne.inst.annotations.onset[idx] = _sync_onset(self.mne.inst,
                                                           rgn[0],
                                                           inverse=True)
        self.mne.inst.annotations.duration[idx] = rgn[1] - rgn[0]

    def _draw_annotations(self):
        # All regions are constantly added to the Scene and handled by Qt
        # which is faster than handling adding/removing in Python.
        pass

    def _add_annotation(self, plot_onset, duration, region=None):
        """Add annotation to Annotations."""
        onset = _sync_onset(self.mne.inst, plot_onset, inverse=True)
        self.mne.inst.annotations.append(onset, duration,
                                         self.mne.current_description)
        self._add_region(plot_onset, duration, self.mne.current_description,
                         region)
        region.select(True)

        # Update Overview-Bar
        self.mne.overview_bar.update()

    def _change_annot_mode(self):
        if not self.mne.annotation_mode:
            # Reset Widgets in Annotation-Figure
            self.mne.fig_annotation.reset()

        # Show Annotation-Dock if activated.
        self.mne.fig_annotation.setVisible(self.mne.annotation_mode)

        # Make Regions movable if activated and move into foreground
        for region in self.mne.regions:
            region.setMovable(self.mne.annotation_mode)
            if self.mne.annotation_mode:
                region.setZValue(2)
            else:
                region.setZValue(0)

        # Remove selection-rectangle.
        if not self.mne.annotation_mode and self.mne.selected_region:
            self.mne.selected_region.select(False)
            self.mne.selected_region = None

    def _toggle_annotation_fig(self):
        self.mne.annotation_mode = not self.mne.annotation_mode
        self._change_annot_mode()

    def _toggle_help_fig(self):
        if self.mne.fig_help is None:
            self.mne.fig_help = HelpDialog(self)
        else:
            self.mne.fig_help.close()
            self.mne.fig_help = None

    def _toggle_butterfly(self):
        self.mne.butterfly = not self.mne.butterfly
        self.mne.ax_vscroll.setVisible(not self.mne.butterfly)
        self.mne.overview_bar.setVisible(not self.mne.butterfly)

        self._update_picks()
        self._update_data()

        if self.mne.butterfly:
            # ToDo: Butterfly + Selection
            ymax = len(self.mne.butterfly_type_order) + 1
            self.mne.plt.setLimits(yMax=ymax)
            self.mne.plt.setYRange(0, ymax, padding=0)
        else:
            self.mne.plt.setLimits(yMax=self.mne.ymax)
            self.mne.plt.setYRange(self.mne.ch_start,
                                   self.mne.ch_start + self.mne.n_channels + 1,
                                   padding=0)

        # update ypos for butterfly-mode
        for trace in self.mne.traces:
            trace.update_ypos()

        self._draw_traces()

    def _toggle_dc(self):
        self.mne.remove_dc = not self.mne.remove_dc
        self._redraw()

    def _toggle_time_format(self):
        if self.mne.time_format == 'float':
            self.mne.time_format = 'clock'
            self.mne.time_axis.setLabel(text='Time')
        else:
            self.mne.time_format = 'float'
            self.mne.time_axis.setLabel(text='Time', units='s')
        self.mne.time_axis.repaint()

    def _toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def _toggle_overview_bar(self):
        self.mne.show_overview_bar = not self.mne.show_overview_bar
        self.mne.overview_bar.setVisible(self.mne.show_overview_bar)

    def _toggle_zenmode(self):
        self.mne.scrollbars_visible = not self.mne.scrollbars_visible
        for bar in [self.mne.ax_hscroll, self.mne.ax_vscroll]:
            bar.setVisible(self.mne.scrollbars_visible)
        self.mne.toolbar.setVisible(self.mne.scrollbars_visible)
        self.mne.overview_bar.setVisible(self.mne.scrollbars_visible)

    def _update_trace_offsets(self):
        pass

    def _create_selection_fig(self):
        pass

    def _toggle_proj_fig(self):
        pass

    def keyPressEvent(self, event):
        """Customize key press events."""
        # On MacOs additionally KeypadModifier is set when arrow-keys
        # are pressed.
        # On Unix GroupSwitchModifier is set when ctrl is pressed.
        # To preserve cross-platform consistency the following comparison
        # of the modifier-values is done.
        # modifiers need to be exclusive
        modifiers = {
            'Ctrl': '4' in hex(int(event.modifiers()))
        }

        for key_name in self.mne.keyboard_shortcuts:
            key_dict = self.mne.keyboard_shortcuts[key_name]
            if key_dict['qt_key'] == event.key():
                slot = key_dict['slot']

                param_idx = 0
                # Get modifier
                if 'modifier' in key_dict:
                    mods = [modifiers[mod] for mod in modifiers]
                    if any(mods):
                        mod = [mod for mod in modifiers if modifiers[mod]][0]
                        if mod in key_dict['modifier']:
                            param_idx = key_dict['modifier'].index(mod)

                if 'parameter' in key_dict:
                    slot(key_dict['parameter'][param_idx])
                else:
                    slot()

                break

    def _draw_traces(self):
        # Update data in traces (=drawing traces)
        for trace in self.mne.traces:
            # Update data
            trace.update_data()

    def _close_event(self, fig=None):
        fig = fig or self
        fig.close()

    def _get_size(self):
        inch_width = self.width() / self.physicalDpiX()
        inch_height = self.height() / self.physicalDpiY()

        return inch_width, inch_height

    def _fake_keypress(self, key, fig=None):
        fig = fig or self
        # Use pytest-qt's exception-hook
        with capture_exceptions() as exceptions:
            QTest.keyPress(fig, self.mne.keyboard_shortcuts[key]['qt_key'])

        for exc in exceptions:
            raise RuntimeError(f'There as been an {exc[0]} inside the Qt '
                               f'event loop (look above for traceback).')

    def _fake_click(self, point, fig=None, ax=None,
                    xform='ax', button=1, kind='press'):

        # Wait until Window is fully shown.
        QTest.qWaitForWindowExposed(self)
        # Scene-Dimensions still seem to change to final state when waiting
        # for a short time.
        QTest.qWait(10)

        # Qt: right-button=2, matplotlib: right-button=3
        button = 2 if button == 3 else button

        # For Qt, fig or ax both would be the widget to test interaction on.
        # If View
        fig = ax or fig or self.mne.view

        if xform == 'ax':
            # For Qt, the equivalent of matplotlibs transAxes
            # would be a transformation to View Coordinates.
            # But for the View top-left is (0, 0) and bottom-right is
            # (view-width, view-height).

            view_width = fig.width()
            view_height = fig.height()

            x = view_width * point[0]
            y = view_height * (1 - point[1])

            point = QPointF(x, y)

        elif xform == 'data':
            # For Qt, the equivalent of matplotlibs transData
            # would be a transformation to
            # the coordinate system of the ViewBox.
            # This only works on the View (self.mne.view)
            fig = self.mne.view
            point = self.mne.viewbox.mapViewToScene(QPointF(*point))
        elif xform == 'none':
            point = QPointF(*point)

        # Use pytest-qt's exception-hook
        with capture_exceptions() as exceptions:
            if kind == 'press':
                _mouseClick(widget=fig, pos=point, button=button)
            elif kind == 'release':
                _mouseRelease(widget=fig, pos=point, button=button)
            elif kind == 'motion':
                _mouseMove(widget=fig, pos=point)

        for exc in exceptions:
            raise RuntimeError(f'There as been an {exc[0]} inside the Qt '
                               f'event loop (look above for traceback).')

        # Waiting some time for events to be processed.
        QTest.qWait(10)

    def _fake_scroll(self, x, y, step, fig=None):
        pass

    def _click_ch_name(self, ch_index, button):
        if not self.mne.butterfly:
            ch_name = self.mne.ch_names[self.mne.picks[ch_index]]
            xrange, yrange = self.mne.channel_axis.ch_texts[ch_name]
            x = np.mean(xrange)
            y = np.mean(yrange)

            self._fake_click((x, y), fig=self.mne.view, button=button,
                             xform='none')

    def _resize_by_factor(self, factor):
        pass

    def _get_ticklabels(self, orientation):
        if orientation == 'x':
            ax = self.mne.time_axis
        else:
            ax = self.mne.channel_axis

        return list(ax.get_labels())

    def closeEvent(self, event):
        """Customize close event."""
        event.accept()

        self._close(event)


qt_key_mapping = {
    'escape': Qt.Key_Escape,
    'down': Qt.Key_Down,
    'up': Qt.Key_Up,
    'left': Qt.Key_Left,
    'right': Qt.Key_Right,
    '-': Qt.Key_Minus,
    '+': Qt.Key_Plus,
    '=': Qt.Key_Equal,
    'pageup': Qt.Key_PageUp,
    'pagedown': Qt.Key_PageDown,
    'home': Qt.Key_Home,
    'end': Qt.Key_End,
    '?': Qt.Key_Question,
    'f11': Qt.Key_F11
}
for char in 'abcdefghijklmnopyqrstuvwxyz0123456789':
    qt_key_mapping[char] = getattr(Qt, f'Key_{char.upper() or char}')


def _get_n_figs():
    return len(QApplication.topLevelWindows())


def _close_all():
    QApplication.closeAllWindows()


# mouse testing functions copied from pyqtgraph (pyqtgraph.tests.ui_testing.py)
def _mousePress(widget, pos, button, modifier=None):
    if isinstance(widget, QGraphicsView):
        widget = widget.viewport()
    if modifier is None:
        modifier = Qt.KeyboardModifier.NoModifier
    event = QMouseEvent(QEvent.Type.MouseButtonPress, pos, button,
                        Qt.MouseButton.NoButton, modifier)
    QApplication.sendEvent(widget, event)


def _mouseRelease(widget, pos, button, modifier=None):
    if isinstance(widget, QGraphicsView):
        widget = widget.viewport()
    if modifier is None:
        modifier = Qt.KeyboardModifier.NoModifier
    event = QMouseEvent(QEvent.Type.MouseButtonRelease, pos,
                        button, Qt.MouseButton.NoButton, modifier)
    QApplication.sendEvent(widget, event)


def _mouseMove(widget, pos, buttons=None, modifier=None):
    if isinstance(widget, QGraphicsView):
        widget = widget.viewport()
    if modifier is None:
        modifier = Qt.KeyboardModifier.NoModifier
    if buttons is None:
        buttons = Qt.MouseButton.NoButton
    event = QMouseEvent(QEvent.Type.MouseMove, pos,
                        Qt.MouseButton.NoButton, buttons, modifier)
    QApplication.sendEvent(widget, event)


def _mouseDrag(widget, pos1, pos2, button, modifier=None):
    _mouseMove(widget, pos1)
    _mousePress(widget, pos1, button, modifier)
    _mouseMove(widget, pos2, button, modifier)
    _mouseRelease(widget, pos2, button, modifier)


def _mouseClick(widget, pos, button, modifier=None):
    _mouseMove(widget, pos)
    _mousePress(widget, pos, button, modifier)
    _mouseRelease(widget, pos, button, modifier)


def _init_browser(inst, figsize, **kwargs):
    setConfigOption('enableExperimental', True)

    mkQApp()
    browser = PyQtGraphBrowser(inst=inst, figsize=figsize, **kwargs)
    width = int(figsize[0] * browser.physicalDpiX())
    height = int(figsize[1] * browser.physicalDpiY())
    browser.resize(width, height)

    return browser
