import os
import time
import subprocess
import io
import tempfile
from PIL import Image
import numpy as np
from PySide6 import QtCore
from mss import mss
from constants import CAPTURE_FPS_CAP
import dbus
import dbus.mainloop.glib
import gi
gi.require_version('GLib', '2.0')
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst
from helpers.logging import debug

dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)

DEBUG_OVERRIDE = True  # Set to True to force debug output regardless of DEBUG_LEVEL

class ScreenGrabber(QtCore.QThread):
    frameCaptured = QtCore.Signal(np.ndarray)

    def __init__(self, id=1):
        super().__init__()
        self.id = id
        self._run = True
        self.is_wayland = os.environ.get("WAYLAND_DISPLAY") is not None

        # Always get monitor geometries at init (for both X11 and Wayland)
        with mss() as s:
            self.monitors = s.monitors  # List of monitor dicts
            if id < len(self.monitors):
                mon = self.monitors[id]
                self.capture_x = mon['left']
                self.capture_y = mon['top']
                self.capture_w = mon['width']
                self.capture_h = mon['height']
            else:
                # Fallback to first monitor if id is out of range
                mon = self.monitors[1]
                self.capture_x = mon['left']
                self.capture_y = mon['top']
                self.capture_w = mon['width']
                self.capture_h = mon['height']

        if not self.is_wayland:
            self.setObjectName(f"ScreenGrabber-{id}")
            debug(f"ScreenGrabber initialized for X11 monitor {id}: {self.capture_x}, {self.capture_y}, {self.capture_w}, {self.capture_h}", debug_override=DEBUG_OVERRIDE)
        else:
            debug(f"ScreenGrabber initialized for Wayland (ScreenCast) monitor {id}: {self.capture_x}, {self.capture_y}, {self.capture_w}, {self.capture_h}", debug_override=DEBUG_OVERRIDE)

    def run(self):
        if self.is_wayland:
            debug("Wayland mode detected, initializing portal session...", debug_override=DEBUG_OVERRIDE)
            # 1. Connect to the session bus
            bus = dbus.SessionBus()
            portal = bus.get_object('org.freedesktop.portal.Desktop', '/org/freedesktop/portal/desktop')
            iface = dbus.Interface(portal, 'org.freedesktop.portal.ScreenCast')

            # 2. Create a session (wait for Response signal)
            token = "screencap" + str(int(time.time()))
            options = {'session_handle_token': token}
            session_path = None
            response = {}

            main_loop = GLib.MainLoop()
            def handle_response(response_id, results, **kwargs):
                debug(f"[Wayland] handle_response called: response_id={response_id}, results={results}", debug_override=DEBUG_OVERRIDE)
                nonlocal session_path, response
                if response_id == 0 and ('session_handle' in results or 'handle' in results):
                    session_path = results.get('session_handle') or results.get('handle')
                    response = results
                    main_loop.quit()

            bus.add_signal_receiver(
                handle_response,
                signal_name="Response",
                dbus_interface="org.freedesktop.portal.Request",
                path_keyword="path"
            )
            debug("[Wayland] Creating session...", debug_override=DEBUG_OVERRIDE)
            iface.CreateSession(options, dbus_interface='org.freedesktop.portal.ScreenCast')
            main_loop.run()
            for _ in range(100):
                if session_path:
                    break
                time.sleep(0.05)
            if not session_path:
                debug("[Wayland] Failed to create portal session.", debug_override=DEBUG_OVERRIDE)
                return

            # 3. Select sources (wait for Response)
            select_options = {'types': dbus.UInt32(1)}
            select_done = False
            def handle_select_response(response_id, results, **kwargs):
                debug(f"[Wayland] handle_select_response: response_id={response_id}, results={results}", debug_override=DEBUG_OVERRIDE)
                nonlocal select_done
                select_done = (response_id == 0)
            bus.add_signal_receiver(
                handle_select_response,
                signal_name="Response",
                dbus_interface="org.freedesktop.portal.Request",
                path_keyword="path"
            )
            debug("[Wayland] Selecting sources...", debug_override=DEBUG_OVERRIDE)
            req_path = iface.SelectSources(session_path, select_options, dbus_interface='org.freedesktop.portal.ScreenCast')
            for _ in range(100):
                if select_done:
                    break
                time.sleep(0.05)
            if not select_done:
                debug("[Wayland] Failed to select sources.", debug_override=DEBUG_OVERRIDE)
                return

            # 4. Start the session (wait for Response)
            start_done = False
            start_results = {}

            def handle_start_response(response_id, results, **kwargs):
                debug(f"[Wayland] handle_start_response: response_id={response_id}, results={results}", debug_override=DEBUG_OVERRIDE)
                nonlocal start_done, start_results
                start_done = (response_id == 0)
                if start_done:
                    start_results = results

            bus.add_signal_receiver(
                handle_start_response,
                signal_name="Response",
                dbus_interface="org.freedesktop.portal.Request",
                path_keyword="path"
            )
            app_id = "screen-cap-translate"
            debug(f"[Wayland] session_path before Start: {session_path!r}", debug_override=DEBUG_OVERRIDE)
            if not session_path or not isinstance(session_path, str):
                debug("[Wayland] Invalid session_path for Start!", debug_override=DEBUG_OVERRIDE)
                return
            debug("[Wayland] Starting session...", debug_override=DEBUG_OVERRIDE)
            iface.Start(session_path, app_id, {}, dbus_interface='org.freedesktop.portal.ScreenCast')
            for _ in range(100):
                if start_done:
                    break
                time.sleep(0.05)
            if not start_done:
                debug("[Wayland] Failed to start session.", debug_override=DEBUG_OVERRIDE)
                return

            # 5. Get PipeWire node ID from start_results
            pw_streams = start_results.get('streams', [])
            if not pw_streams:
                debug("[Wayland] No streams returned from portal.", debug_override=DEBUG_OVERRIDE)
                return

            # Use self.id to select the correct stream if multiple were returned
            if self.id < len(pw_streams):
                node_id, props = pw_streams[self.id]
            else:
                node_id, props = pw_streams[0]  # fallback to first

            size = props.get('size')
            position = props.get('position')
            debug(f"[Wayland] PipeWire node_id: {node_id}, size: {size}, position: {position}", debug_override=DEBUG_OVERRIDE)

            # Prefer portal's size/position, fallback to mss geometry from __init__
            if size:
                self.capture_w, self.capture_h = int(size[0]), int(size[1])
            if position:
                self.capture_x, self.capture_y = int(position[0]), int(position[1])
                debug(f"[Wayland] Using portal position for capture_x/capture_y: {self.capture_x}, {self.capture_y}", debug_override=DEBUG_OVERRIDE)
            else:
                mon = self.monitors[self.id]
                self.capture_x = mon['left']
                self.capture_y = mon['top']
                debug(f"[Wayland] Using mss fallback for capture_x/capture_y: {self.capture_x}, {self.capture_y}", debug_override=DEBUG_OVERRIDE)

            debug(f"[Wayland] Capture area set to x={self.capture_x}, y={self.capture_y}, w={self.capture_w}, h={self.capture_h}", debug_override=DEBUG_OVERRIDE)

            # Initialize GStreamer
            Gst.init(None)
            debug("[Wayland] Initializing GStreamer pipeline...", debug_override=DEBUG_OVERRIDE)
            pipeline = Gst.parse_launch(
                f"pipewiresrc path={node_id} ! videoconvert ! video/x-raw,format=BGR,width={self.capture_w},height={self.capture_h} ! appsink name=sink"
            )
            appsink = pipeline.get_by_name("sink")
            appsink.set_property("max-buffers", 1)
            appsink.set_property("drop", True)
            pipeline.set_state(Gst.State.PLAYING)

            prev = 0
            while self._run:
                if time.time() - prev < 1 / CAPTURE_FPS_CAP:
                    time.sleep(.002)
                    continue
                prev = time.time()
                sample = appsink.emit("pull-sample")
                if sample:
                    buf = sample.get_buffer()
                    arr = np.ndarray(
                        (self.capture_h, self.capture_w, 3),
                        buffer=buf.extract_dup(0, buf.get_size()),
                        dtype=np.uint8,
                    )
                    debug(f"[Wayland] Frame captured at {time.time():.3f}", debug_override=False)
                    self.frameCaptured.emit(arr.copy())
                else:
                    debug("[Wayland] No frame received from PipeWire.", debug_override=False)
                    time.sleep(0.1)
            pipeline.set_state(Gst.State.NULL)
        else:
            with mss() as s:
                mon = self.monitors[self.id]
                prev = 0
                while self._run:
                    if time.time() - prev < 1 / CAPTURE_FPS_CAP:
                        time.sleep(.002)
                        continue
                    prev = time.time()
                    f = np.asarray(s.grab(mon), dtype=np.uint8)[..., :3].copy()
                    self.frameCaptured.emit(f)

    def stop(self):
        self._run = False
        self.wait()