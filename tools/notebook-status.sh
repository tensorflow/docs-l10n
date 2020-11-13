#!/usr/bin/env bash
## Generate reports for failed notebooks.
## Present each languages failed notebooks in `/site/<lang>/FAILURES.md`.
## Collected from: https://storage.googleapis.com/tensorflow_docs/failed_notebooks/<lang>/failed_notebooks.txt
## Usage:
##  $ ./tools/notebook-status.sh [-h]

usage() {
  echo "Usage: $(basename $0) [options]"
  echo "  Generate reports for failed notebooks."
  echo "Options:"
  echo " -c     Auto-create the git commit w/ status files"
  echo " -h     Print this help and exit"
}

while getopts "ch" opt; do
  case $opt in
    c) COMMIT_FLAG=1;;
    h | *)
      usage
      exit 0
      ;;
  esac
done

LOG_NAME="[$(basename $0)]"
TIMESTAMP=$(date '+%s')
REPO_ROOT="$(cd $(dirname ${BASH_SOURCE[0]}) >/dev/null 2>&1 && cd .. && pwd)"
TEMP_DIR=$(mktemp -d -t "$(basename $0 '.sh').XXXXX")

##
## CHECK SETUP
##

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
    branch_name="notebook-status-${TIMESTAMP}"
    echo "${LOG_NAME} Create new branch for notebook status: ${branch_name}"
    git checkout -b "${branch_name}"
  fi
fi

##
## DOWNLOAD LOGS
##

readarray -t langs < <(find "$REPO_ROOT"/site -mindepth 1 -maxdepth 1 -type d \
                         | xargs -n1 basename \
                         | grep -v "en-snapshot" \
                         | sort)

# Entry values contain a list of failed notebook GitHub paths, separated by a newline.
declare -A LANG_FILES
declare -A LANG_MODIFIED

for lang in "${langs[@]}"; do
  # Get file list. Save headers to parse for last-modified time.
  files=$(curl --silent --fail --dump-header "$TEMP_DIR/headers-${lang}" \
               "https://storage.googleapis.com/tensorflow_docs/failed_notebooks/${lang}/failed_notebooks.txt")

  if [[ $? != 0 ]]; then
    echo "${LOG_NAME} Unable to download log for '${lang}', skipping." >&2
    continue
  fi

  # Parse header for time of last file update. (xargs to trim whitespace)
  last_modified=$(cat "$TEMP_DIR/headers-${lang}" \
                    | grep -i "last-modified" \
                    | cut -d':' -f2- \
                    | xargs)
  LANG_MODIFIED["$lang"]="$last_modified"
  LANG_FILES["$lang"]="$files"
done

##
## STATUS REPORT
##

print_timestamp() {
  local timestamp="$1"
  local timestamp_str=""
  local fmt_rfc7231="%a, %d %b %Y %H:%M:%S %Z"
  if [[ $(uname) == "Darwin" ]]; then
    timestamp_str=$(TZ=GMT date -r "$timestamp" +"$fmt_rfc7231")  #BSD
  else
    timestamp_str=$(TZ=GMT date -d "@$timestamp" +"$fmt_rfc7231")  # Linux
  fi
  echo "$timestamp_str"
}

TIMESTAMP_STR="$(print_timestamp $TIMESTAMP)"

COMMIT_MSG="Notebook status: ${TIMESTAMP_STR}\n\n
Update notebook failures.\n"

README_STR="__DO NOT EDIT__

## Notebook failures

The notebooks listed below did *not* run to completion. This usually indicates
that a translated file is out-of-date and not synced to its
[source counterpart](../en-snapshot/). Notebooks can be run in Colab and synced
using the [GitLocalize project](https://gitlocalize.com/tensorflow/docs-l10n).

*Notebooks are tested on a periodic basis (usually weekly or bi-weekly) so the
following list may not reflect recent updates.*

Updated: ${TIMESTAMP_STR}<br/>"

# GitLocalize may use a different path name.
declare -A LANG_GITLOCALIZE
LANG_GITLOCALIZE["es-419"]="es"

echo -n "${LOG_NAME} Create status report files ... "

for lang in "${!LANG_FILES[@]}"; do
  STATUS_FILE="${REPO_ROOT}/site/${lang}/FAILURES.md"
  echo -e "$README_STR" > "$STATUS_FILE"
  echo -e "Last run: ${LANG_MODIFIED[$lang]}" >> "$STATUS_FILE"

  # Get file list and remove empty lines.
  readarray -t failed_notebooks < <(echo "${LANG_FILES[$lang]}" \
                                      | grep -v '^$' \
                                      | sort)

  if [[ ${#failed_notebooks[@]} == 0 ]]; then
    # No failures.
    echo -e "\nAll <code>site/${lang}/</code> notebooks pass :)\n" >> "$STATUS_FILE"
    continue
  else
    echo -e "\nFailures in <code>site/${lang}/</code>:\n" >> "$STATUS_FILE"
  fi

  if [[ -v LANG_GITLOCALIZE["$lang"] ]]; then
    gitlocalize_lang=${LANG_GITLOCALIZE["$lang"]}
  else
    gitlocalize_lang="$lang"
  fi

  # Full path like: https://github.com/tensorflow/docs-l10n/blob/master/site/<lang>/guide/autodiff.ipynb
  for github_path in "${failed_notebooks[@]}"; do
    rel_path="${github_path#*site/${lang}/}"
    gitlocalize_path="https://gitlocalize.com/repo/4592/${gitlocalize_lang}/site/en-snapshot/${rel_path}"
    colab_path="https://colab.research.google.com/github/tensorflow/docs-l10n/blob/master/site/${lang}/${rel_path}"

    # Create list entry.
    li_entry="* [$rel_path]($github_path) · [[Sync in GitLocalize]($gitlocalize_path)] · [[Test in Colab]($colab_path)]"
    echo "$li_entry" >> "$STATUS_FILE"
  done
  echo -n -e "\n" >> "$STATUS_FILE"  # Add final newline.
done

echo "done."

##
## FINISH OPTIONS
##

# Commit change
if [[ -n "$COMMIT_FLAG" ]]; then
  cd "$REPO_ROOT"
  echo "${LOG_NAME} Create status commit ..."
  git add "$REPO_ROOT/site/"
  COMMIT_MSG=$(echo -e "$COMMIT_MSG")
  git commit --message "$COMMIT_MSG"
fi

# Cleanup
rm -rf "$TEMP_DIR"
