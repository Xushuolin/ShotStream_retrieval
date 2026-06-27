#!/usr/bin/env bash
set -euo pipefail

MODE="${1:---vendor}"
REPO_URL="${LONGLIVE_REPO_URL:-https://github.com/NVlabs/LongLive.git}"
BRANCH="${LONGLIVE_BRANCH:-main}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_DIR="$ROOT_DIR/third_party/longlive2/LongLive"
TMP_DIR="$(mktemp -d)"

cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

case "$MODE" in
  --vendor|--worktree)
    ;;
  *)
    echo "Usage: $0 [--vendor|--worktree]" >&2
    echo "  --vendor   clone LongLive2 and remove nested .git for vendoring" >&2
    echo "  --worktree clone LongLive2 as an independent nested git checkout" >&2
    exit 2
    ;;
esac

if [[ -e "$DEST_DIR" ]]; then
  echo "Destination already exists: $DEST_DIR" >&2
  echo "Move it away or remove it before fetching LongLive2." >&2
  exit 1
fi

mkdir -p "$(dirname "$DEST_DIR")"

echo "Cloning $REPO_URL branch $BRANCH..."
git clone --single-branch --branch "$BRANCH" --depth 1 "$REPO_URL" "$TMP_DIR/LongLive"

if [[ "$MODE" == "--vendor" ]]; then
  rm -rf "$TMP_DIR/LongLive/.git"
fi

mv "$TMP_DIR/LongLive" "$DEST_DIR"

echo "LongLive2 fetched to $DEST_DIR"
if [[ "$MODE" == "--vendor" ]]; then
  echo "Vendored mode: nested .git removed; run 'git add third_party/longlive2/LongLive' if you want to commit the source."
else
  echo "Worktree mode: nested .git preserved; do not git-add this directory unless you intentionally want a submodule/gitlink."
fi
