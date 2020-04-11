#!/bin/sh

# Get target directory or file.
TARGET=${1}

# Other constants.
ERROR_LIMIT=20

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

  echo ${FILE}

  if [ ${FILE##*.} = "md" ]; then
    TARGET_MARKDOWN=${FILE}
  elif [ ${FILE##*.} = "ipynb" ]; then
    # Convert ipynb to md and save it in temporary directory.
    jupyter nbconvert --to markdown --output-dir ${TEMP_DIR} ${FILE}
    TARGET_MARKDOWN="${TEMP_DIR}/$(basename ${FILE} .ipynb).md"
  fi

  echo ""
  echo "Apply RedPen to: ${TARGET_MARKDOWN}"
  # redpen-distribution-1.10.1/bin/redpen --limit ${ERROR_LIMIT} --conf tools/ja/redpen-conf.xml ${TARGET_MARKDOWN}
  redpen --limit ${ERROR_LIMIT} --conf tools/ja/redpen-conf.xml ${TARGET_MARKDOWN}

done
