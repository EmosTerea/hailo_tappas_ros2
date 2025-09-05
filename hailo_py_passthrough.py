#!/usr/bin/env python3
"""
Minimal HailoPython postprocess used to ensure hailonet attaches tensors to
the ROI metadata. It does not modify the frame; it simply accesses the
Hailo tensors so downstream elements (e.g., our identity pad probe) can
retrieve them reliably without any .so post-processing.

Intended usage in pipeline:
  ... ! hailonet ... ! hailopython module=/path/hailo_py_passthrough.py function=run qos=false ! identity name=after_hailo ! fakesink
"""
from __future__ import annotations

import builtins
import numpy as np  # type: ignore
import hailo  # type: ignore
from gsthailo import VideoFrame  # type: ignore
from gi.repository import Gst  # type: ignore


def _tensor_to_numpy(t):
    # Prefer full precision float when available
    try:
        arr = np.array(t.get_full_percision(), copy=False)  # typo in API on some versions
    except Exception:
        try:
            arr = np.array(t.get_full_precision(), copy=False)
        except Exception:
            try:
                arr = np.array(t, copy=False)
            except Exception:
                arr = np.array([])
    # Try shape reconstruction from dims
    # Try reshape or fallback to element-wise extraction
    try:
        h = int(t.height()); w = int(t.width()); c = int(t.features())
        if arr.size == h * w * c and arr.ndim != 3:
            arr = arr.reshape((h, w, c))
        if arr.size != h * w * c or arr.ndim != 3:
            # Slow but robust: per-element get
            dtype = np.float32
            tmp = np.empty((h, w, c), dtype=dtype)
            for yy in range(h):
                for xx in range(w):
                    for cc in range(c):
                        try:
                            tmp[yy, xx, cc] = float(t.get(yy, xx, cc))
                        except Exception:
                            tmp[yy, xx, cc] = 0.0
            arr = tmp
    except Exception:
        pass
    return np.array(arr, copy=True)


def run(video_frame: VideoFrame):
    result = {}
    try:
        roi = video_frame.roi
        # Extract tensors (if any)
        try:
            tensors = list(roi.get_tensors())
        except Exception:
            tensors = []
        # Debug: print once the tensor type and available methods
        try:
            printed = getattr(run, "_printed_dbg", False)
            if not printed and tensors:
                t0 = tensors[0]
                print("[hailopython] tensor type:", type(t0))
                print("[hailopython] tensor dir sample:", [m for m in dir(t0) if not m.startswith('__')][:30])
                run._printed_dbg = True
        except Exception:
            pass
        for t in tensors:
            try:
                name = t.name()
            except Exception:
                name = f"tensor_{len(result)}"
            # Debug dims
            try:
                h = int(t.height()); w = int(t.width()); c = int(t.features())
                shp = None
                try:
                    shp = t.shape()
                except Exception:
                    pass
                print(f"[hailopython] {name} dims h,w,c=({h},{w},{c}) shape={shp}")
            except Exception:
                pass
            result[name] = _tensor_to_numpy(t)
        # Fallback to matrices
        if not result:
            try:
                mats = roi.get_objects_typed(hailo.HAILO_MATRIX)
            except Exception:
                mats = []
            for i, m in enumerate(mats):
                try:
                    data = m.get_data()
                    arr = np.array(data)
                    try:
                        shp = m.shape()
                        shp = tuple(int(x) for x in (list(shp) if not isinstance(shp, (list, tuple)) else shp))
                        if arr.size == int(np.prod(shp)):
                            arr = arr.reshape(shp)
                    except Exception:
                        pass
                    result[f"matrix_{i}"] = np.array(arr, copy=True)
                except Exception:
                    continue
    except Exception:
        # Ignore extraction errors; just emit empty result
        result = {}

    # Push into a global collector queue if provided by the host script
    try:
        collector = getattr(builtins, 'HAILO_PY_COLLECTOR', None)
        if collector is not None:
            collector.put(result)
    except Exception:
        pass

    return Gst.FlowReturn.OK


def finalize(video_frame: VideoFrame):  # optional finalize hook
    return Gst.FlowReturn.OK
