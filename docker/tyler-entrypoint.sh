#!/bin/sh
set -e

# Simple entrypoint to support two modes:
# 1) Single Roofer CityJSONL input:
#       tyler-container /data/buildings-and-trees.jsonl [tyler-args...]
#    -> runs roofer2tyler.py to create metadata.city.json + features/,
#       then invokes Tyler with -m/-f and any extra arguments.
# 2) Direct Tyler invocation:
#       tyler-container tyler-args...
#    -> passes all arguments straight through to the tyler binary.

if [ $# -gt 0 ] && [ -f "$1" ] && echo "$1" | grep -qE '\.jsonl$'; then
    JSONL_INPUT="$1"
    shift

    JSONL_DIR="$(dirname "$JSONL_INPUT")"
    METADATA_PATH="${JSONL_DIR}/metadata.city.json"
    FEATURES_DIR="${JSONL_DIR}/features"

    echo "Preprocessing Roofer CityJSONL: ${JSONL_INPUT}"
    python3 /usr/local/bin/roofer2tyler.py "${JSONL_INPUT}"

    echo "Running Tyler with metadata: ${METADATA_PATH}, features: ${FEATURES_DIR}"
    exec tyler -m "${METADATA_PATH}" -f "${FEATURES_DIR}" "$@"
else
    # Assume the user is passing Tyler CLI arguments directly
    exec tyler "$@"
fi

