#!/usr/bin/env bash
set -euo pipefail

libero_script_dir() {
  cd "$(dirname "${BASH_SOURCE[0]}")" && pwd
}

libero_repo_root() {
  cd "$(libero_script_dir)/../../../.." && pwd
}

sanitize_component() {
  local value="$1"
  value="${value//+/_}"
  value="${value// /_}"
  printf '%s' "${value}" | sed -E 's/[^A-Za-z0-9._-]+/_/g; s/^[-_.]+//; s/[-_.]+$//'
}

derive_ckpt_alias() {
  local ckpt_path="$1"
  local parent_dir grandparent_dir
  parent_dir="$(basename "$(dirname "${ckpt_path}")")"
  grandparent_dir="$(basename "$(dirname "$(dirname "${ckpt_path}")")")"

  if [[ "${grandparent_dir}" != "final_model" && -n "${grandparent_dir}" ]]; then
    sanitize_component "${grandparent_dir}"
    return
  fi

  if [[ -n "${parent_dir}" ]]; then
    sanitize_component "${parent_dir}"
    return
  fi

  sanitize_component "$(basename "${ckpt_path%.*}")"
}

ensure_libero_eval_config() {
  local libero_home="$1"
  local libero_config_path="$2"
  if [ -f "${libero_config_path}/config.yaml" ]; then
    return 0
  fi

  mkdir -p "${libero_config_path}"
  cat > "${libero_config_path}/config.yaml" <<EOF
benchmark_root: ${libero_home}/libero/libero
bddl_files: ${libero_home}/libero/libero/./bddl_files
init_states: ${libero_home}/libero/libero/./init_files
datasets: ${libero_home}/libero/libero/../datasets
assets: ${libero_home}/libero/libero/./assets
EOF
}

detect_num_gpus() {
  local python_bin="$1"
  if [ -n "${NUM_GPUS:-}" ]; then
    printf '%s\n' "${NUM_GPUS}"
    return
  fi

  "${python_bin}" - <<'PY'
try:
    import torch
    count = int(torch.cuda.device_count())
except Exception:
    count = 0
print(max(count, 1))
PY
}

resolve_gpu_id() {
  local run_index="$1"
  local num_gpus="$2"
  local gpu_list_raw="${3:-}"
  local normalized="${gpu_list_raw//,/ }"

  if [ -n "${normalized//[[:space:]]/}" ]; then
    local -a gpu_ids=()
    local gpu_id
    read -r -a gpu_ids <<< "${normalized}"
    if [ "${#gpu_ids[@]}" -eq 0 ]; then
      echo "[ERROR] GPU list is empty after parsing: ${gpu_list_raw}" >&2
      return 1
    fi
    for gpu_id in "${gpu_ids[@]}"; do
      if ! [[ "${gpu_id}" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] GPU list contains a non-integer entry: ${gpu_id}" >&2
        return 1
      fi
    done
    printf '%s\n' "${gpu_ids[$((run_index % ${#gpu_ids[@]}))]}"
    return
  fi

  printf '%s\n' "$((run_index % num_gpus))"
}

wait_for_port() {
  local python_bin="$1"
  local host="$2"
  local port="$3"
  local timeout_sec="$4"
  "${python_bin}" - "$host" "$port" "$timeout_sec" <<'PY'
import socket
import sys
import time

host = sys.argv[1]
port = int(sys.argv[2])
timeout_sec = float(sys.argv[3])
deadline = time.time() + timeout_sec

while time.time() < deadline:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1.0)
    try:
        sock.connect((host, port))
    except OSError:
        time.sleep(1.0)
    else:
        sock.close()
        sys.exit(0)
    finally:
        try:
            sock.close()
        except OSError:
            pass

print(f"Timed out waiting for {host}:{port} after {timeout_sec:.0f}s", file=sys.stderr)
sys.exit(1)
PY
}

find_available_port() {
  local python_bin="$1"
  local preferred_port="$2"
  local search_limit="${3:-200}"
  "${python_bin}" - "$preferred_port" "$search_limit" <<'PY'
import socket
import sys

preferred_port = int(sys.argv[1])
search_limit = int(sys.argv[2])

if preferred_port <= 0:
    raise SystemExit(f"Invalid preferred port: {preferred_port}")
if search_limit <= 0:
    raise SystemExit(f"Invalid port search limit: {search_limit}")

for offset in range(search_limit):
    port = preferred_port + offset
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("0.0.0.0", port))
    except OSError:
        continue
    else:
        print(port)
        raise SystemExit(0)
    finally:
        sock.close()

raise SystemExit(
    f"Failed to find an available port in range "
    f"[{preferred_port}, {preferred_port + search_limit - 1}]"
)
PY
}

RESERVED_PORT=""
RESERVED_PORT_LOCK_FD=""
RESERVED_PORT_LOCK_PATH=""

reserve_port() {
  local python_bin="$1"
  local preferred_port="$2"
  local search_limit="${3:-200}"
  local lock_root="${PORT_LOCK_ROOT:-/tmp/starvla_libero_port_locks}"
  local port lock_path lock_fd

  if ! command -v flock >/dev/null 2>&1; then
    echo "[ERROR] \`flock\` is required for coordinated LIBERO port reservation." >&2
    return 1
  fi

  mkdir -p "${lock_root}"
  for ((offset = 0; offset < search_limit; offset++)); do
    port=$((preferred_port + offset))
    lock_path="${lock_root}/port_${port}.lock"
    exec {lock_fd}> "${lock_path}"
    if ! flock -n "${lock_fd}"; then
      eval "exec ${lock_fd}>&-"
      continue
    fi

    if ! "${python_bin}" - "${port}" <<'PY'
import socket
import sys

port = int(sys.argv[1])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.bind(("0.0.0.0", port))
except OSError:
    raise SystemExit(1)
finally:
    sock.close()
PY
    then
      flock -u "${lock_fd}" || true
      eval "exec ${lock_fd}>&-"
      continue
    fi

    RESERVED_PORT="${port}"
    RESERVED_PORT_LOCK_FD="${lock_fd}"
    RESERVED_PORT_LOCK_PATH="${lock_path}"
    return 0
  done

  echo "[ERROR] Failed to reserve an available port in range [${preferred_port}, $((preferred_port + search_limit - 1))]." >&2
  return 1
}

release_reserved_port() {
  if [ -n "${RESERVED_PORT_LOCK_FD:-}" ]; then
    flock -u "${RESERVED_PORT_LOCK_FD}" 2>/dev/null || true
    eval "exec ${RESERVED_PORT_LOCK_FD}>&-"
  fi
  RESERVED_PORT=""
  RESERVED_PORT_LOCK_FD=""
  RESERVED_PORT_LOCK_PATH=""
}
