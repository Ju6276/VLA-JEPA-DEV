#!/usr/bin/env bash
set -euo pipefail

BRANCH="XXY-INFRA"
REMOTE_NAME="ju6276"
REMOTE_URL="git@github.com:Ju6276/VLA-JEPA-DEV.git"

EXCLUDED_DIRS=(
  "checkpoints"
  "merged_dataset_001"
  "Qwen3-VL-2B-Instruct"
  "vjepa2-vitl-fpc64-256"
  "VJEPA21"
)

EXCLUDE_PATHSPECS=(
  ":(exclude)checkpoints/**"
  ":(exclude)merged_dataset_001/**"
  ":(exclude)Qwen3-VL-2B-Instruct/**"
  ":(exclude)vjepa2-vitl-fpc64-256/**"
  ":(exclude)VJEPA21/**"
)

usage() {
  cat <<EOF
Usage:
  ./push_to_github.sh [commit message]

Default:
  Sync the current repo to GitHub branch ${BRANCH}.

Excluded directories:
$(printf '  - %s\n' "${EXCLUDED_DIRS[@]}")
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || {
  echo "ERROR: not inside a git repo." >&2
  exit 1
}
cd "$REPO_ROOT"

COMMIT_MESSAGE="${1:-Sync ${BRANCH} $(date +'%Y-%m-%d %H:%M:%S')}"

echo "Repo:   $REPO_ROOT"
echo "Branch: $BRANCH"
echo "Remote: $REMOTE_URL"
echo

if git remote get-url "$REMOTE_NAME" >/dev/null 2>&1; then
  git remote set-url "$REMOTE_NAME" "$REMOTE_URL"
else
  git remote add "$REMOTE_NAME" "$REMOTE_URL"
fi

CURRENT_BRANCH="$(git branch --show-current)"
if [[ "$CURRENT_BRANCH" != "$BRANCH" ]]; then
  if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
    git switch "$BRANCH"
  else
    git switch -c "$BRANCH"
  fi
fi

# Keep excluded heavy/local directories out of the commit even if they were staged earlier.
git restore --staged -- "${EXCLUDED_DIRS[@]}" >/dev/null 2>&1 || true

git add -A -- . "${EXCLUDE_PATHSPECS[@]}"

echo "Staged changes:"
git diff --cached --stat
echo

if git diff --cached --quiet; then
  echo "No changes to commit. Pushing current ${BRANCH} branch."
else
  git commit -m "$COMMIT_MESSAGE"
fi

git push -u "$REMOTE_NAME" "$BRANCH"

echo
echo "Done. Synced to ${REMOTE_NAME}/${BRANCH}."
