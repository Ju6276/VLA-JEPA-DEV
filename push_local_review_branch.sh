#!/usr/bin/env bash
#
# 用途：
#   把“本次修改”整理成一个新的本地 review 分支，提交成 commit，
#   并推送到本机 bare repo remote（默认 remote 名为 local）。
#   这样新的 agent 可以直接通过 `git diff main..review/xxx`
#   查看本次修改痕迹，判断代码是否存在问题。
#
# 最常用命令：
#   1. 只提交指定文件，推荐用于当前工作区已经比较乱的情况：
#      ./push_local_review_branch.sh AGENT.md starVLA/model/framework/VLA_JEPA.py \
#        -b review/vlajepa-inference-sop \
#        -m "Add VLA-JEPA inference SOP"
#
#   2. 提交当前仓库内所有普通代码改动，并自动排除权重/缓存/结果目录：
#      ./push_local_review_branch.sh --all \
#        -b review/my-change \
#        -m "Describe my change"
#
#   3. 先演练一次，不创建最终 commit、不 push：
#      ./push_local_review_branch.sh --dry-run AGENT.md
#
# 查看 review 分支改动：
#   git diff main..review/my-change
#   git log --oneline main..review/my-change
#
# 注意：
#   - 这个脚本默认推送到本机 bare repo `local`，不会推送到 GitHub `origin`。
#   - 脚本运行前不能有已经 staged 的文件；请先 commit 或 `git reset` 取消暂存。
#   - `--all` 会排除 checkpoints/results/logs/模型权重/cache 等大文件目录。
set -euo pipefail

REMOTE="local"
BRANCH=""
MESSAGE=""
ADD_MODE="tracked"
DRY_RUN="false"

EXCLUDES=(
  ":(exclude).venv/**"
  ":(exclude)__pycache__/**"
  ":(exclude)**/__pycache__/**"
  ":(exclude)*.pyc"
  ":(exclude)*.pt"
  ":(exclude)*.pth"
  ":(exclude)*.safetensors"
  ":(exclude)checkpoints/**"
  ":(exclude)results/**"
  ":(exclude)logs/**"
  ":(exclude)wandb/**"
  ":(exclude)Qwen3-VL-2B-Instruct/**"
  ":(exclude)vjepa2-vitl-fpc64-256/**"
  ":(exclude)VJEPA21/**"
  ":(exclude)merged_dataset_001/**"
  ":(exclude)starVLA.egg-info/**"
)

usage() {
  cat <<'EOF'
用法:
  ./push_local_review_branch.sh [options] [paths...]

作用:
  创建一个新的本地 review 分支，把指定修改提交成一个 commit，
  并推送到本机 bare repo remote。新的 agent 可以基于这个分支
  查看本次修改痕迹，例如 `git diff main..review/xxx`。

参数:
  -b, --branch NAME     新分支名。默认: review/YYYYMMDD-HHMMSS
  -m, --message TEXT    commit 信息。默认: "Review snapshot: <branch>"
  -r, --remote NAME     推送目标 remote。默认: local
      --all             提交全部仓库改动，但自动排除大文件/缓存/结果目录。
      --tracked         只提交已被 Git 跟踪文件的修改；无 paths 时默认如此。
      --dry-run         只展示将提交什么，然后退出，不创建最终 commit、不 push。
  -h, --help            显示帮助。

常用例子:
  # 推荐：只提交指定文件，最干净
  ./push_local_review_branch.sh AGENT.md starVLA/model/framework/VLA_JEPA.py

  # 指定分支名和 commit 信息
  ./push_local_review_branch.sh AGENT.md \
    -b review/update-agent-doc \
    -m "Update agent SOP"

  # 提交全部普通代码改动，脚本会排除权重/缓存/结果目录
  ./push_local_review_branch.sh --all -m "Prepare VLA-JEPA inference SOP"

  # 只演练，不真正提交或推送
  ./push_local_review_branch.sh --dry-run AGENT.md

  # 指定分支名
  ./push_local_review_branch.sh -b review/openfaucet-serve --all

review 常用命令:
  git diff main..review/openfaucet-serve
  git diff --stat main..review/openfaucet-serve
  git log --oneline main..review/openfaucet-serve

注意:
  - 默认推送到本机 bare repo remote `local`，不会推送到 GitHub `origin`。
  - 使用 --all 时，模型、checkpoint、result、cache 等目录会被排除。
  - 如果某个被排除的新文件确实需要提交，请显式传入该文件路径。
  - 脚本运行前不能有已经 staged 的文件，否则会中止以避免混入旧修改。
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

timestamp() {
  date +"%Y%m%d-%H%M%S"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -b|--branch)
      [[ $# -ge 2 ]] || die "Missing value for $1"
      BRANCH="$2"
      shift 2
      ;;
    -m|--message)
      [[ $# -ge 2 ]] || die "Missing value for $1"
      MESSAGE="$2"
      shift 2
      ;;
    -r|--remote)
      [[ $# -ge 2 ]] || die "Missing value for $1"
      REMOTE="$2"
      shift 2
      ;;
    --all)
      ADD_MODE="all"
      shift
      ;;
    --tracked)
      ADD_MODE="tracked"
      shift
      ;;
    --dry-run)
      DRY_RUN="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      die "Unknown option: $1"
      ;;
    *)
      break
      ;;
  esac
done

PATHS=("$@")

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || die "Not inside a git repo"
cd "$REPO_ROOT"

git remote get-url "$REMOTE" >/dev/null 2>&1 || die "Remote '$REMOTE' does not exist. Expected local bare repo remote."

if [[ -z "$BRANCH" ]]; then
  BRANCH="review/$(timestamp)"
fi

if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
  die "Local branch '$BRANCH' already exists. Choose another name with --branch."
fi

if git ls-remote --exit-code --heads "$REMOTE" "$BRANCH" >/dev/null 2>&1; then
  die "Remote branch '$REMOTE/$BRANCH' already exists. Choose another name with --branch."
fi

CURRENT_BRANCH="$(git branch --show-current)"
if [[ -z "$CURRENT_BRANCH" ]]; then
  die "Detached HEAD is not supported by this helper script."
fi

if ! git diff --cached --quiet; then
  die "There are already staged changes. Commit/unstage them first so this review snapshot stays clean."
fi

echo "Repo:    $REPO_ROOT"
echo "Base:    $CURRENT_BRANCH"
echo "Branch:  $BRANCH"
echo "Remote:  $REMOTE"

git switch -c "$BRANCH"

if [[ "${#PATHS[@]}" -gt 0 ]]; then
  echo "Staging explicit paths:"
  printf '  %s\n' "${PATHS[@]}"
  git add -A -- "${PATHS[@]}"
elif [[ "$ADD_MODE" == "all" ]]; then
  echo "Staging all changes with built-in excludes."
  git add -A -- . "${EXCLUDES[@]}"
else
  echo "Staging tracked-file changes only. New untracked files are not included."
  git add -u
fi

echo
echo "Staged diff summary:"
git diff --cached --stat

if git diff --cached --quiet; then
  echo
  echo "No staged changes. Returning to $CURRENT_BRANCH and deleting empty branch."
  git switch "$CURRENT_BRANCH"
  git branch -D "$BRANCH" >/dev/null
  exit 0
fi

if [[ "$DRY_RUN" == "true" ]]; then
  echo
  echo "Dry run only. Returning to $CURRENT_BRANCH and deleting temporary branch."
  git reset --quiet
  git switch "$CURRENT_BRANCH"
  git branch -D "$BRANCH" >/dev/null
  exit 0
fi

if [[ -z "$MESSAGE" ]]; then
  MESSAGE="Review snapshot: $BRANCH"
fi

git commit -m "$MESSAGE"
git push -u "$REMOTE" "$BRANCH"

echo
echo "Done."
echo "Review branch pushed: $REMOTE/$BRANCH"
echo
echo "Useful review commands:"
echo "  git log --oneline --decorate $CURRENT_BRANCH..$BRANCH"
echo "  git diff --stat $CURRENT_BRANCH..$BRANCH"
echo "  git diff $CURRENT_BRANCH..$BRANCH"
echo
echo "To return to the base branch:"
echo "  git switch $CURRENT_BRANCH"
