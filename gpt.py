# realtime_plot_api.py
import sys
import queue
import numpy as np
import pyqtgraph as pg
from PyQt6 import QtWidgets, QtCore


class RealTimePlotAPI:
    """
    Multi-curve real-time XY plotting API with fixed-width scrolling x-window.

    Key methods:
      - start()                 # blocking GUI loop
      - stop()
      - push(channel, x, y)     # push scalar or arrays for a named channel

    Notes:
      - GUI should run in main thread (Qt best practice).
      - push() is thread-safe.
    """
    def __init__(
        self,
        title="Real-Time Multi-Curve Plot",
        window_size=(1000, 600),
        target_fps=60,
        x_window=10.0,          # seconds (or x-units) shown
        y_range=None,           # tuple(min, max) or None for auto
        line_width=2,
        max_batch_drain=1000,   # safety cap per frame while draining queue
        legend=True,
        grid=True,
    ):
        self.title = title
        self.window_size = window_size
        self.target_fps = int(target_fps)
        self.x_window = float(x_window)
        self.y_range = y_range
        self.line_width = line_width
        self.max_batch_drain = int(max_batch_drain)
        self.legend = legend
        self.grid = grid

        self._q = queue.Queue()
        self._running = False

        # Channel state:
        # channel_name -> dict(x=np.ndarray, y=np.ndarray, curve=PlotDataItem, color=...)
        self._channels = {}

        # Round-robin color palette
        self._palette = [
            "#00D1FF", "#FFDD00", "#FF5C8A", "#7CFF6B", "#B388FF",
            "#FF9F1C", "#2EC4B6", "#E71D36", "#A1C181", "#5E60CE"
        ]
        self._color_idx = 0

        # Qt handles
        self._app = None
        self._win = None
        self._plot = None
        self._timer = None
        self._legend_item = None

    # ---------------------- Public API ---------------------- #
    def push(self, channel, x, y):
        """
        Push data for a channel (thread-safe).

        channel: str
        x, y: scalar or array-like (same shape)
        """
        if not isinstance(channel, str) or not channel:
            raise ValueError("channel must be a non-empty string")

        x_arr = np.atleast_1d(np.asarray(x, dtype=np.float64))
        y_arr = np.atleast_1d(np.asarray(y, dtype=np.float64))

        if x_arr.shape != y_arr.shape:
            raise ValueError(f"x and y shape mismatch: {x_arr.shape} vs {y_arr.shape}")

        self._q.put((channel, x_arr, y_arr))

    def start(self):
        """
        Start GUI loop (blocking). Run in main thread.
        """
        if self._running:
            return

        self._running = True
        self._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

        self._win = pg.GraphicsLayoutWidget(show=True, title=self.title)
        self._win.resize(*self.window_size)

        self._plot = self._win.addPlot(title=self.title)
        if self.grid:
            self._plot.showGrid(x=True, y=True)
        if self.legend:
            self._legend_item = self._plot.addLegend()

        if self.y_range is not None:
            self._plot.setYRange(self.y_range[0], self.y_range[1])

        # We control x-range manually as a scrolling window
        self._plot.setAutoVisible(x=False, y=True)

        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self._update)
        interval_ms = max(1, int(1000 / self.target_fps))
        self._timer.start(interval_ms)

        self._win.destroyed.connect(lambda: self.stop())

        self._app.exec()

    def stop(self):
        if not self._running:
            return

        self._running = False
        if self._timer is not None:
            self._timer.stop()
        if self._app is not None:
            self._app.quit()

    # ---------------------- Internal ---------------------- #
    def _next_color(self):
        c = self._palette[self._color_idx % len(self._palette)]
        self._color_idx += 1
        return c

    def _ensure_channel(self, channel):
        if channel in self._channels:
            return

        color = self._next_color()
        curve = self._plot.plot(
            name=channel,
            pen=pg.mkPen(color=color, width=self.line_width)
        )
        self._channels[channel] = {
            "x": np.empty(0, dtype=np.float64),
            "y": np.empty(0, dtype=np.float64),
            "curve": curve,
            "color": color
        }

    def _update(self):
        if not self._running:
            return

        # Drain queue (bounded per frame to avoid UI stalls)
        drained = 0
        got_any = False
        while drained < self.max_batch_drain:
            try:
                channel, x_new, y_new = self._q.get_nowait()
            except queue.Empty:
                break

            self._ensure_channel(channel)
            ch = self._channels[channel]

            # Append
            ch["x"] = np.concatenate((ch["x"], x_new))
            ch["y"] = np.concatenate((ch["y"], y_new))
            got_any = True
            drained += 1

        if not got_any:
            return

        # Determine global right edge from latest x across channels
        latest_x = None
        for ch in self._channels.values():
            if ch["x"].size > 0:
                lx = ch["x"][-1]
                if (latest_x is None) or (lx > latest_x):
                    latest_x = lx

        if latest_x is None:
            return

        x_min = latest_x - self.x_window
        x_max = latest_x

        # Update each channel: clip to current x-window, then draw
        for ch in self._channels.values():
            x = ch["x"]
            y = ch["y"]
            if x.size == 0:
                continue

            # Keep only points within window (plus tiny margin)
            mask = x >= (x_min - 1e-12)
            if not np.all(mask):
                x = x[mask]
                y = y[mask]
                ch["x"] = x
                ch["y"] = y

            ch["curve"].setData(x, y)

        # Set fixed-width scrolling x-range
        self._plot.setXRange(x_min, x_max, padding=0.0)