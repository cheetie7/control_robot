#!/usr/bin/env bash
# Check or set the FTDI USB latency timer used by the FACTR Dynamixel bus.

set -euo pipefail

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_directory}/../.." && pwd)"
factr_config="${project_root}/xarm_ros2_teleop/config/gello_grav_comp_demo.yaml"

set_latency=false
if [[ ${1:-} == "--set" ]]; then
    set_latency=true
    shift
fi
if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [--set] [FACTR_SERIAL_DEVICE]" >&2
    exit 2
fi

# An explicit device takes precedence. Otherwise read the quoted device path
# from the FACTR YAML used by the main launch.
device_path="${1:-}"
if [[ -z ${device_path} ]]; then
    device_path="$(
        sed -nE \
            's/^[[:space:]]*dynamixel_port:[[:space:]]*"([^"]+)".*/\1/p' \
            "${factr_config}" | head -n 1
    )"
fi
if [[ -z ${device_path} || ! -e ${device_path} ]]; then
    echo "FACTR serial device is not connected: ${device_path:-<empty>}" >&2
    exit 1
fi

actual_device="$(readlink -f -- "${device_path}")"
tty_name="${actual_device##*/}"
if [[ ${tty_name} != ttyUSB* ]]; then
    echo "FACTR device must resolve to ttyUSB*, got: ${actual_device}" >&2
    exit 1
fi

latency_file="/sys/bus/usb-serial/devices/${tty_name}/latency_timer"
if [[ ! -r ${latency_file} ]]; then
    echo "FTDI latency timer is unavailable: ${latency_file}" >&2
    exit 1
fi

latency_value="$(<"${latency_file}")"
if [[ ${latency_value} == "1" ]]; then
    echo "FACTR latency timer is ready: ${tty_name}=1"
    exit 0
fi
if [[ ${set_latency} != true ]]; then
    echo "FACTR latency timer is ${latency_value}; expected 1." >&2
    echo "Run: $0 --set ${device_path}" >&2
    exit 1
fi

printf '1\n' | sudo tee "${latency_file}" >/dev/null
echo "FACTR latency timer updated: ${tty_name}=1"
