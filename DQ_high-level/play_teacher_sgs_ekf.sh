#!/usr/bin/env bash
# Run from any directory after activating the dqwbc environment.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-python}"
# Isaac Gym dynamically loads libpython, which conda does not always expose.
python_lib_dir="$("$python_bin" -c 'import sys; print(sys.prefix + "/lib")')"
export LD_LIBRARY_PATH="${python_lib_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
exec "$python_bin" "$script_dir/play_teacher_sgs_ekf.py" "$@"
