#!/bin/sh

# Get target directory or file.
TARGET=${1}

# Other constants.
ERROR_LIMIT=20
RESULT_FORMAT="plain2"

# Find target files.
TARGET_FILES=$(find ${TARGET} -type f -name *.ipynb -or -name '*.md')

# Temporary directory. Convert notebooks to markdown and save them here.
TEMP_DIR=$(mktemp -d)

# Show configurations.
echo "TARGET: ${TARGET}"
echo "TEMP_DIR: ${TEMP_DIR}"
echo "ERROR_LIMIT: ${ERROR_LIMIT}"
echo ""

# Apply RedPen to target files.
for FILE in ${TARGET_FILES}; do

  if [ ${FILE##*.} = "md" ]; then
    TARGET_MARKDOWN=${FILE}
  elif [ ${FILE##*.} = "ipynb" ]; then
    # Convert ipynb to md and save it in temporary directory.
    jupyter nbconvert --to markdown --output-dir ${TEMP_DIR} ${FILE}
    TARGET_MARKDOWN="${TEMP_DIR}/$(basename ${FILE} .ipynb).md"
  fi

  redpen --result-format ${RESULT_FORMAT} \
         --limit ${ERROR_LIMIT} \
         --conf tools/ci/ja/redpen-conf.xml ${TARGET_MARKDOWN} \
         2>tools/ci/ja/redpen.log

done
