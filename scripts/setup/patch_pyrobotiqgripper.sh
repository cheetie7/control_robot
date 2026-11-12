#!/usr/bin/env bash
# Patch the missing Python 3.10 Optional import in pyrobotiqgripper 3.2.8.

set -euo pipefail

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_directory}/../.." && pwd)"
gripper_file="${project_root}/.venv/lib/python3.10/site-packages/pyrobotiqgripper/gripper.py"

if [[ ! -f "${gripper_file}" ]]; then
    echo "pyrobotiqgripper is not installed at ${gripper_file}" >&2
    echo "Run 'uv sync --frozen' before applying this patch." >&2
    exit 1
fi

if grep -qFx "from typing import Optional" "${gripper_file}"; then
    echo "pyrobotiqgripper Optional import is already patched."
    exit 0
fi

sed -i '1i from typing import Optional' "${gripper_file}"
echo "Patched ${gripper_file}"
