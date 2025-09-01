#!/usr/bin/env bash
set -euo pipefail

echo "== OS =="
if [ -f /etc/os-release ]; then cat /etc/os-release; fi
echo

echo "== Kernel =="
uname -a || true
echo

echo "== Python =="
command -v python3 && python3 --version || echo "python3 not found"
command -v pip3 && pip3 --version || echo "pip3 not found"
echo

echo "== GStreamer core =="
if command -v gst-inspect-1.0 >/dev/null 2>&1; then
  which gst-inspect-1.0; gst-inspect-1.0 --version
else
  echo "gst-inspect-1.0 not found"
fi
echo

echo "== Hailo CLIs =="
for c in hailortcli hailortctl hailo hailo_rt_info hailo_fw_update; do
  if command -v "$c" >/dev/null 2>&1; then
    echo "-- $c --"
    ("$c" --version 2>/dev/null || "$c" -v 2>/dev/null || echo "version flag unknown")
  fi
done
echo

echo "== APT packages (hailo*) =="
if command -v dpkg >/dev/null 2>&1; then dpkg -l | awk '/hailo|Hailo|gst.*hailo|hailort/ {print}'; fi
echo

echo "== Python packages (hailo*) =="
if command -v python3 >/dev/null 2>&1; then
python3 - <<'PY'
import pkgutil, pkg_resources, sys
print('discovered modules:', [m.name for m in pkgutil.iter_modules() if 'hailo' in m.name.lower()])
try:
    for d in sorted(pkg_resources.working_set, key=lambda d: d.project_name.lower()):
        if 'hailo' in d.project_name.lower():
            print(f"{d.project_name}=={d.version} @ {d.location}")
except Exception as e:
    print('pkg_resources error:', e)
PY
fi
echo

echo "== GStreamer Hailo elements =="
if command -v gst-inspect-1.0 >/dev/null 2>&1; then
  gst-inspect-1.0 | awk 'tolower($0) ~ /hailo/ {print}' || true
  echo
  for e in hailonet hailofilter hailooverlay hailocropper hailotracker; do
    echo "-- $e --"
    gst-inspect-1.0 "$e" 2>/dev/null | awk 'NR<=60{print}' || true
    echo
  done
fi

echo "== libhailort on system =="
ldconfig -p 2>/dev/null | awk '/hailort/ {print}' || true
find /usr/lib /usr/local/lib /lib /opt -maxdepth 4 -type f -name 'libhailo*.so*' 2>/dev/null | sed 's/^/ - /' || true
echo

echo "== Kernel module hailort =="
modinfo hailort 2>/dev/null | awk 'NR<=60{print}' || echo "no kernel module or insufficient permissions"
echo

echo "== PCI/USB Hailo devices =="
command -v lspci >/dev/null 2>&1 && lspci | awk '/Hailo|AI/ {print}' || true
command -v lsusb >/dev/null 2>&1 && lsusb | awk '/Hailo|HAILO/ {print}' || true

