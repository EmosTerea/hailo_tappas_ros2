#!/usr/bin/env bash
set -Eeuo pipefail

# Hailo-8 PCIe driver installer for Ubuntu 24.04 on Raspberry Pi (raspi kernel)
# - Installs prerequisites (build tools, DKMS, headers)
# - Clones hailort-drivers (default tag v4.22.0)
# - Installs the kernel module via DKMS (persistent across kernel updates)
# - Downloads and installs firmware to /lib/firmware/hailo/hailo8_fw.bin
# - Installs udev rules and reloads them
# - Loads the module and verifies binding
#
# Usage:
#   sudo bash ./install_hailo_dkms.sh [--version v4.22.0] [--repo-dir /path] [--no-autoload]
#
# Notes:
# - Do NOT mix this with a manual "make install" of the module; DKMS replaces that flow.
# - On raspi kernels, the build auto-enables the Pi quirk macro (no need to pass CFLAGS).

VERSION="v4.22.0"
REPO_DIR=""
AUTOLOAD=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --version)
      VERSION="$2"; shift 2 ;;
    --repo-dir)
      REPO_DIR="$2"; shift 2 ;;
    --no-autoload)
      AUTOLOAD=0; shift ;;
    -h|--help)
      grep -E "^#( |$)" "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)
      echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
  echo "Please run with sudo: sudo bash $0" >&2
  exit 1
fi

echo "[+] Hailo PCIe DKMS install starting (version: ${VERSION})"

# Resolve invoking user home for repo placement if not provided
INVOKER=${SUDO_USER:-$(id -un)}
USER_HOME=$(getent passwd "$INVOKER" | cut -d: -f6)
[[ -n "$REPO_DIR" ]] || REPO_DIR="${USER_HOME}/hailort-drivers"

echo "[+] Using repo dir: $REPO_DIR"

echo "[+] Installing prerequisites (this may take a minute)..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y \
  build-essential git bc flex bison libelf-dev libssl-dev \
  dkms linux-headers-raspi linux-headers-"$(uname -r)" \
  curl ca-certificates pciutils

# Clone or update the repository to the requested tag
if [[ -d "$REPO_DIR/.git" ]]; then
  echo "[+] Repo exists; fetching tags and checking out $VERSION"
  git -C "$REPO_DIR" fetch --tags --force
  git -C "$REPO_DIR" checkout -f "$VERSION"
else
  echo "[+] Cloning hailort-drivers@$VERSION"
  git clone --depth=1 --branch "$VERSION" https://github.com/hailo-ai/hailort-drivers.git "$REPO_DIR"
fi

echo "[+] Building and installing kernel module via DKMS"
cd "$REPO_DIR/linux/pcie"
# Be tolerant if uninstall target is missing in older versions
make -s clean >/dev/null 2>&1 || true
# Remove any prior non-DKMS installation to avoid duplicates in modules tree
make -s uninstall >/dev/null 2>&1 || true
make -s uninstall_dkms >/dev/null 2>&1 || true
make install_dkms

echo "[+] Running depmod"
depmod -a

if [[ "$AUTOLOAD" -eq 1 ]]; then
  echo "[+] Enabling autoload at boot (/etc/modules-load.d/hailo_pci.conf)"
  echo hailo_pci > /etc/modules-load.d/hailo_pci.conf
fi

echo "[+] Downloading firmware"
cd "$REPO_DIR"
./download_firmware.sh

echo "[+] Installing firmware to /lib/firmware/hailo/hailo8_fw.bin"
mkdir -p /lib/firmware/hailo
# Try to find the most recent hailo8_fw*.bin in repo root
FW_SRC=$(ls -1t hailo8_fw*.bin 2>/dev/null | head -n1 || true)
if [[ -z "$FW_SRC" ]]; then
  echo "[!] Firmware file not found in repo root after download_firmware.sh" >&2
  echo "    Please check the download script output." >&2
  exit 1
fi
install -m 0644 -T "$FW_SRC" /lib/firmware/hailo/hailo8_fw.bin

echo "[+] Installing udev rules"
install -m 0644 -T "$REPO_DIR/linux/pcie/51-hailo-udev.rules" /etc/udev/rules.d/51-hailo-udev.rules
udevadm control --reload-rules
udevadm trigger

echo "[+] Loading hailo_pci module"
modprobe -r hailo_pci 2>/dev/null || true
modprobe hailo_pci

echo
echo "=== DKMS status ==="
dkms status | grep -i hailo || true

echo
echo "=== Module path ==="
modinfo -n hailo_pci || true

echo
echo "=== PCI device (Hailo) ==="
# Print the device block for Hailo-8 if present
lspci -nnk | awk '/Hailo-8 AI Processor/{print;getline;print;getline;print;getline;print}' || true

echo
echo "[+] Install complete. The driver should now persist across reboots and kernel updates."