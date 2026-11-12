#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# System-installed rules that grant non-root USB access to each sensor family.
REALSENSE_RULE="/lib/udev/rules.d/60-librealsense2-udev-rules.rules"
ORBBEC_SOURCE="/opt/ros/humble/share/orbbec_camera/udev/99-obsensor-libusb.rules"
ORBBEC_TARGET="/etc/udev/rules.d/99-obsensor-libusb.rules"
XENSE_SOURCE="$SCRIPT_DIR/99-xense.rules"
XENSE_TARGET="/etc/udev/rules.d/99-xense.rules"

if [[ ! -f "$REALSENSE_RULE" ]]; then
  echo "RealSense udev rule not found; skipping RealSense."
else
  echo "RealSense udev rule is already installed."
fi

if [[ ! -f "$ORBBEC_SOURCE" ]]; then
  echo "Orbbec udev rule not found; skipping Orbbec."
else
  sudo install -m 0644 "$ORBBEC_SOURCE" "$ORBBEC_TARGET"
fi

sudo groupadd -f xense
sudo usermod -aG xense "$USER"
sudo install -m 0644 "$XENSE_SOURCE" "$XENSE_TARGET"
# Disable USB runtime power management for all three sensor vendors so idle
# sensors are never autosuspended mid-stream.
USB_POWER_SOURCE="$SCRIPT_DIR/99-sensor-usb-power.rules"
USB_POWER_TARGET="/etc/udev/rules.d/99-sensor-usb-power.rules"
sudo install -m 0644 "$USB_POWER_SOURCE" "$USB_POWER_TARGET"
sudo udevadm control --reload-rules
sudo udevadm trigger

echo "Sensor udev rules are ready. Reconnect the cameras and Xense sensor, then log in again."
