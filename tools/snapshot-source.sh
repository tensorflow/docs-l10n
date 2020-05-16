#!/usr/bin/env bash
## Pull the source of all TensorFlow doc projects into a single snapshot.
## Usage:
##  $ ./tools/snapshot-source.sh [-h]
##  $ ./tools/snapshot-source.sh -c  # Create git commit/branch after pull
##
## Docs are collected in the ./site/en-snapshot directory
set -e

usage() {
  echo "Usage: $(basename $0) [options]"
  echo "  Copy source docs from all projects into local directory."
  echo "Options:"
  echo " -c     Auto-create the git commit w/ status message"
  echo " -o=dir Set a different output directory (default: site/en-snapshot)"
  echo " -h     Print this help and exit"
}

while getopts "co:h" opt; do
  case $opt in
    c) COMMIT_FLAG=1;;
    o) SNAPSHOT_ROOT="$OPTARG";;
    h | *)
      usage
      exit 0
      ;;
  esac
done

# List of all TensorFlow doc projects to pull in.
# format: project = repo:branch:src-dir:dest-dir
declare -A PROJECTS=(
  [addons]="tensorflow/addons:master:docs:addons"
  [agents]="tensorflow/agents:master:docs:agents"
  [datasets]="tensorflow/datasets:master:docs:datasets"
  [docs]="tensorflow/docs:master:site/en:."
  [federated]="tensorflow/federated:master:docs:federated"
  [graphics]="tensorflow/graphics:master:tensorflow_graphics/g3doc:graphics"
  [hub]="tensorflow/hub:master:docs:hub"
  [hub_tutorials]="tensorflow/hub:master:examples/colab:hub/tutorials"
  [io]="tensorflow/io:master:docs:io"
  [js]="tensorflow/tfjs-website:master:docs:js"
  [lattice]="tensorflow/lattice:master:docs:lattice"
  [lite]="tensorflow/tensorflow:master:tensorflow/lite/g3doc:lite"
  [mlir]="tensorflow/tensorflow:master:tensorflow/compiler/mlir/g3doc:mlir"
  [model_optimization]="tensorflow/model-optimization:master:tensorflow_model_optimization/g3doc:model_optimization"
  [neural_structured_learning]="tensorflow/neural-structured-learning:master:g3doc:neural_structured_learning"
  [probability]="tensorflow/probability:master:tensorflow_probability/g3doc:probability"
  [quantum]="tensorflow/quantum:master:docs:quantum"
  [swift]="tensorflow/swift:master:docs/site:swift"
  [tensorboard]="tensorflow/tensorboard:master:docs:tensorboard"
  [tfx]="tensorflow/tfx:master:docs:tfx"
  [xla]="tensorflow/tensorflow:master:tensorflow/compiler/xla/g3doc:xla"
)

LOG_NAME="[$(basename $0)]"
REPO_ROOT="$(cd $(dirname ${BASH_SOURCE[0]}) >/dev/null 2>&1 && cd .. && pwd)"
TEMP_DIR=$(mktemp -d -t "$(basename $0 '.sh')")
TIMESTAMP=$(date '+%s')

declare -A LAST_COMMITS  # Last commit ID for each project

##
## CHECK SETUP
##

if [[ -z "$SNAPSHOT_ROOT" ]]; then
  SNAPSHOT_ROOT="${REPO_ROOT}/site/en-snapshot"
  mkdir -p "$SNAPSHOT_ROOT"
fi

if [[ ! -d "$SNAPSHOT_ROOT" ]]; then
  echo "${LOG_NAME} Output directory does not exist: ${SNAPSHOT_ROOT}" >&2
  exit 1
fi

# Git status
if [[ -n "$COMMIT_FLAG" ]]; then
  # Want a clean branch if making a commit.
  if [[ -n $(git status -s) ]]; then
    echo "${LOG_NAME} git status not clear, exiting." >&2
    echo "Can run without committing, see help: $(basename $0) -h" >&2
    exit 1
  fi

  # Create new branch if on master
  if [[ $(git rev-parse --abbrev-ref HEAD) == "master" ]]; then
    branch_name="en-snapshot-${TIMESTAMP}"
    echo "${LOG_NAME} Create new branch for snapshot: ${branch_name}"
    git checkout -b "${branch_name}"
  fi
fi

##
## DOWNLOAD PROJECTS
##

echo "${LOG_NAME} Download projects to: ${TEMP_DIR}"

for project in "${!PROJECTS[@]}"; do
  repo=$(echo "${PROJECTS[$project]}" | cut -f1 -d':')
  branch=$(echo "${PROJECTS[$project]}" | cut -f2 -d':')

  # Download shallow clone of each project in temp.
  cd "$TEMP_DIR"
  git clone "git@github.com:${repo}.git" \
      --branch "$branch" --single-branch --depth 1 "$project"

  # Store last commit id for project.
  cd "./${project}"
  last_commit=$(git log --format="%H" -n 1)
  LAST_COMMITS[$project]="$last_commit"
done

##
## SYNC PROJECTS
##

rsync_opts=(
  --checksum
  --recursive
  --archive
  --del
  --exclude='BUILD'
  --exclude='README.md'
  --exclude='*.gwsq'
  --exclude='*.py'
  --exclude='*.yaml'
  --exclude='_*.ipynb'
  --exclude='index.*'
  --exclude='operation_semantics.md'
  --exclude='api_docs/'
  --exclude='/catalog/'
  --exclude='images/'
  --exclude='r1/'
)

# Root-level excludes for the 'docs' project.
root_excludes=(
  --exclude='/install/'
)

for project in "${!PROJECTS[@]}"; do
  dest_path=$(echo "${PROJECTS[$project]}" | cut -f4 -d':')
  if [[ "$project" != "docs" ]]; then
    root_excludes+=("--exclude=/$dest_path/")
  fi
done

echo "${LOG_NAME} Copy projects to: ${SNAPSHOT_ROOT}"

for project in "${!PROJECTS[@]}"; do
  src_path=$(echo "${PROJECTS[$project]}" | cut -f3 -d':')
  dest_path=$(echo "${PROJECTS[$project]}" | cut -f4 -d':')
  src_path_full="$TEMP_DIR/$project/$src_path"
  dest_path_full="$SNAPSHOT_ROOT/$dest_path"

  if [[ "$project" == "docs" ]]; then
    rsync_opts+=( "${root_excludes[@]}" )
  fi

  mkdir -p "$dest_path_full"
  rsync "${rsync_opts[@]}" "${src_path_full}/" "${dest_path_full}/"
done

##
## STATUS REPORTING
##

COMMIT_MSG_LIST=""
README_MSG_LIST=""

for project in "${!LAST_COMMITS[@]}"; do
  last_commit="${LAST_COMMITS[$project]}"
  repo=$(echo "${PROJECTS[$project]}" | cut -f1 -d':')
  branch=$(echo "${PROJECTS[$project]}" | cut -f2 -d':')
  src_path=$(echo "${PROJECTS[$project]}" | cut -f3 -d':')

  project_url="https://github.com/${repo}/tree/${branch}/${src_path}"
  commit_url="https://github.com/${repo}/commit/${last_commit}"

  # Append to both logs
  COMMIT_MSG_LIST+="- ${repo}: ${commit_url}\n"
  README_MSG_LIST+="- [${repo}](${project_url}): [${last_commit}](${commit_url})\n"
done

# Order project list
COMMIT_MSG_LIST="$(echo -e $COMMIT_MSG_LIST | sort)"
README_MSG_LIST="$(echo -e $README_MSG_LIST | sort)"

print_timestamp() {
  local timestamp="$1" timestamp_str=""
  if [[ $(uname) == "Darwin" ]]; then
    timestamp_str=$(date -r "$timestamp")  # BSD style
  else
    timestamp_str=$(date -d "@$timestamp")  # Linux
  fi
  echo "$timestamp_str"
}

TIMESTAMP_STR="$(print_timestamp $TIMESTAMP)"

COMMIT_MSG="Snapshot of the English source documentation.\n
Updated: ${TIMESTAMP_STR}\n
Projects and last commit:\n
${COMMIT_MSG_LIST}\n"

README_STR="DO NOT EDIT

This is a snapshot of the English source documentation.

Updated: ${TIMESTAMP_STR}

Projects and last commit:
${README_MSG_LIST}\n"


echo -e "$README_STR" > "${SNAPSHOT_ROOT}/README.md"

# Commit change
if [[ -n "$COMMIT_FLAG" ]]; then
  cd "$REPO_ROOT"
  git add "$SNAPSHOT_ROOT"
  COMMIT_MSG=$(echo -e "$COMMIT_MSG")
  git commit --message "$COMMIT_MSG"
fi

# Cleanup
rm -rf "$TEMP_DIR"
