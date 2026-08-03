#!/bin/bash
# Push local branch to GitHub.
#
# Usage:
#   bash push_github.sh                     # push current branch
#   bash push_github.sh vlajepa-xxy           # push a specific branch
#   bash push_github.sh --force               # force push current branch
#   bash push_github.sh vlajepa-xxy --force    # force push a specific branch
#   bash push_github.sh --no-proxy            # push without proxy
#
# Environment overrides:
#   GITHUB_REMOTE   default: origin
#   HTTP_PROXY / HTTPS_PROXY   used when proxy is enabled

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

REMOTE="${GITHUB_REMOTE:-origin}"
FORCE=0
USE_PROXY=1
BRANCH=""

usage() {
  sed -n '2,12p' "$0"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    -f|--force)
      FORCE=1
      shift
      ;;
    --no-proxy)
      USE_PROXY=0
      shift
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
    *)
      if [[ -z "${BRANCH}" ]]; then
        BRANCH="$1"
      else
        echo "Unexpected argument: $1" >&2
        usage
        exit 1
      fi
      shift
      ;;
  esac
done

if [[ -z "${BRANCH}" ]]; then
  BRANCH="$(git branch --show-current)"
fi

if [[ -z "${BRANCH}" ]]; then
  echo "Error: not on a branch (detached HEAD). Pass branch name explicitly." >&2
  exit 1
fi

enable_proxy() {
  if declare -F proxy_on >/dev/null 2>&1; then
    proxy_on
    return
  fi

  export http_proxy="${HTTP_PROXY:-http://172.16.3.158:3128}"
  export https_proxy="${HTTPS_PROXY:-http://172.16.3.158:3128}"
  export no_proxy="${NO_PROXY:-localhost,127.0.0.1,.aliyuncs.com,.internal}"
  echo "Proxy enabled: ${https_proxy}"
}

disable_proxy() {
  if declare -F proxy_off >/dev/null 2>&1; then
    proxy_off
    return
  fi

  unset http_proxy https_proxy no_proxy
  echo "Proxy disabled"
}

if [[ "${USE_PROXY}" -eq 1 ]]; then
  enable_proxy
else
  disable_proxy
fi

echo "Repository: ${PROJECT_ROOT}"
echo "Remote:     ${REMOTE}"
echo "Branch:     ${BRANCH}"
echo "Force push: $([[ "${FORCE}" -eq 1 ]] && echo yes || echo no)"
echo

git status --short --branch

if ! git diff-index --quiet HEAD -- 2>/dev/null; then
  echo
  echo "Warning: working tree has uncommitted changes."
fi

echo
read -r -p "Continue push to ${REMOTE}/${BRANCH}? [y/N] " confirm
if [[ ! "${confirm}" =~ ^[Yy]$ ]]; then
  echo "Aborted."
  exit 0
fi

PUSH_ARGS=(-u "${REMOTE}" "${BRANCH}")
if [[ "${FORCE}" -eq 1 ]]; then
  PUSH_ARGS=(--force-with-lease "${PUSH_ARGS[@]}")
fi

git push "${PUSH_ARGS[@]}"

echo
echo "Done: ${REMOTE}/${BRANCH}"
