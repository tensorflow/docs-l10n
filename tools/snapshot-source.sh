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
  echo " -F     Do not format notebooks. Formatting requires the tensorflow-docs pip package."
  echo " -o=dir Set a different output directory (default: site/en-snapshot)"
  echo " -h     Print this help and exit"
}

while getopts "cFo:h" opt; do
  case $opt in
    c) COMMIT_FLAG=1;;
    F) NO_FORMAT_FLAG=1;;
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
  [federated]="tensorflow/federated:main:docs:federated"
  [graphics]="tensorflow/graphics:master:tensorflow_graphics/g3doc:graphics"
  [hub]="tensorflow/hub:master:docs:hub"
  [hub_tutorials]="tensorflow/hub:master:examples/colab:hub/tutorials"
  [io]="tensorflow/io:master:docs:io"
  [js]="tensorflow/tfjs-website:master:docs:js"
  [keras_guides]="tensorflow/docs:snapshot-keras:site/en/guide/keras:guide/keras"
  [lattice]="tensorflow/lattice:master:docs:lattice"
  [lite]="tensorflow/tensorflow:master:tensorflow/lite/g3doc:lite"
  [mlir]="tensorflow/tensorflow:master:tensorflow/compiler/mlir/g3doc:mlir"
  [model_optimization]="tensorflow/model-optimization:master:tensorflow_model_optimization/g3doc:model_optimization"
  [neural_structured_learning]="tensorflow/neural-structured-learning:master:g3doc:neural_structured_learning"
  [probability]="tensorflow/probability:main:tensorflow_probability/g3doc:probability"
  [probability_examples]="tensorflow/probability:main:tensorflow_probability/examples/jupyter_notebooks:probability/examples"
  [quantum]="tensorflow/quantum:master:docs:quantum"
  [tensorboard]="tensorflow/tensorboard:master:docs:tensorboard"
  [tfx]="tensorflow/tfx:master:docs:tfx"
  [xla]="tensorflow/tensorflow:master:tensorflow/compiler/xla/g3doc:xla"
)

LOG_NAME="[$(basename $0)]"
REPO_ROOT="$(cd $(dirname ${BASH_SOURCE[0]}) >/dev/null 2>&1 && cd .. && pwd)"
TEMP_DIR=$(mktemp -d -t "$(basename $0 '.sh').XXXXX")
TEMP_SITE_ROOT="$TEMP_DIR/_siteroot"
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

# Notebook formatting requires the tensorflow-docs package.
# https://github.com/tensorflow/docs/tree/master/tools/tensorflow_docs/tools
if [[ -z "$NO_FORMAT_FLAG" ]]; then
  if ! python3 -m pip list | grep "tensorflow-docs" > /dev/null 2>&1; then
    echo "${LOG_NAME} Error: Can't find the tensorflow-docs pip package for formatting. (Use -F to disable.)" >&2
    exit 1
  fi
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
  if [[ $(git branch --show-current) == "master" ]]; then
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
  src_path=$(echo "${PROJECTS[$project]}" | cut -f3 -d':')
  dest_path=$(echo "${PROJECTS[$project]}" | cut -f4 -d':')

  # Download shallow clone of each project in temp.
  cd "$TEMP_DIR"
  git clone "https://github.com/${repo}.git" \
      --branch "$branch" --single-branch --depth 1 "$project"

  # Store last commit id for project.
  cd "./${project}"
  last_commit=$(git log --format="%H" -n 1)
  LAST_COMMITS[$project]="$last_commit"

  # Assemble shadow site
  mkdir -p "$TEMP_SITE_ROOT/$dest_path"
  cp -R "$src_path"/* "$TEMP_SITE_ROOT/$dest_path"/
done

###
## PRUNE
##

# Keep only doc formats.
find "$TEMP_SITE_ROOT" \
     -type f \( ! -name "*.ipynb" ! -name "*.md" ! -name "*.html" \) \
  | xargs rm

# Remove files we don't publish or don't translate.
find "$TEMP_SITE_ROOT" \
     -type f \( -name "README*" -or -name "_*" -or -name "index.*" \) -or \
     -type f \( -path "*/api_docs/*" -or -path "*/r1/*" \) \
  | xargs rm

# Remove specific pages or sections.
rm -rf "$TEMP_SITE_ROOT/install/"  # Different process.
rm -rf "$TEMP_SITE_ROOT/datasets/catalog/"  # Reference
rm -rf "$TEMP_SITE_ROOT/tensorboard/design"  # Design docs
rm "$TEMP_SITE_ROOT/xla/operation_semantics.md"  # Reference
# Cloud integration not available here (b/197880392)
rm "$TEMP_SITE_ROOT/guide/keras/training_keras_models_on_cloud.ipynb"

##
## SYNC
##

echo "${LOG_NAME} Copy projects to: ${SNAPSHOT_ROOT}"

rsync --archive --del --checksum "$TEMP_SITE_ROOT/" "$SNAPSHOT_ROOT/"

##
## STATUS REPORT
##

COMMIT_MSG_LIST=""
README_MSG_LIST=""

for project in "${!LAST_COMMITS[@]}"; do
  last_commit="${LAST_COMMITS[$project]}"
  short_id=$(echo "$last_commit" | head -c 8)
  repo=$(echo "${PROJECTS[$project]}" | cut -f1 -d':')
  branch=$(echo "${PROJECTS[$project]}" | cut -f2 -d':')
  src_path=$(echo "${PROJECTS[$project]}" | cut -f3 -d':')

  project_url="https://github.com/${repo}/tree/${branch}/${src_path}"
  commit_url="https://github.com/${repo}/commit/${last_commit}"

  # Append to both logs
  COMMIT_MSG_LIST+="- ${project}: ${commit_url}\n"
  README_MSG_LIST+="- [${project}](${project_url}) @ <a href='${commit_url}'><code>${short_id}</code></a>\n"
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

COMMIT_MSG="Source snapshot: ${TIMESTAMP_STR}\n\n
Projects and last commit:\n
${COMMIT_MSG_LIST}\n"

README_STR="__DO NOT EDIT__

This snapshot of the English documentation is used for tensorflow.org
translations. Do not edit these files. The source-of-truth files are located in
the projects listed below.

Please submit translations from the GitLocalize project: https://gitlocalize.com/tensorflow/docs-l10n

Updated: ${TIMESTAMP_STR}

Projects and last commit:
${README_MSG_LIST}\n"


CHANGELOG_FILE="${SNAPSHOT_ROOT}/README.md"
echo -e "$README_STR" > "$CHANGELOG_FILE"

##
## FINISH OPTIONS
##

# Format notebooks
if [[ -z "$NO_FORMAT_FLAG" ]]; then
  echo "${LOG_NAME} Format notebooks ..."
  if ! python3 -m tensorflow_docs.tools.nbfmt "${SNAPSHOT_ROOT}" > /dev/null 2>&1; then
    echo "${LOG_NAME} nbfmt error, exiting." >&2
    exit 1
  fi
fi

# Commit change
if [[ -n "$COMMIT_FLAG" ]]; then
  cd "$REPO_ROOT"
  # Want to commit more than a timestamp update. (READMEs already excluded)
  modified_docs=$(git ls-files --modified | grep -v "README.md" | wc -l)
  if (( "$modified_docs" == 0 )); then
    echo "${LOG_NAME} No commit since there are no file changes."
    git restore "$CHANGELOG_FILE"
  else
    echo "${LOG_NAME} Create snapshot commit ..."
    git add "$SNAPSHOT_ROOT"
    COMMIT_MSG=$(echo -e "$COMMIT_MSG")
    git commit --message "$COMMIT_MSG"
  fi
fi

# Cleanup
rm -rf "$TEMP_DIR"
