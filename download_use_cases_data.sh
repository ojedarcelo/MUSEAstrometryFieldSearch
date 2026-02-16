#!/bin/bash

set -e  # Stop if any command fails

# -------- Variables --------
URL="https://zenodo.org/records/18653307/files/Field_Search_Data.tar.gz?download=1"
ARCHIVE="Field_Search_Data.tar.gz"
TMP_DIR="Field_Search_Data"

# Get directory where this script lives
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

cd "$SCRIPT_DIR"

echo "Downloading data from Zenodo..."
wget -O "$ARCHIVE" "$URL"

echo "Extracting archive..."
tar -xzf "$ARCHIVE"

echo "Moving use case folders to script directory..."
mv "$TMP_DIR/nfm_use_case" "$SCRIPT_DIR/"
mv "$TMP_DIR/wfm_use_case" "$SCRIPT_DIR/"

echo "Cleaning up..."
rm -rf "$TMP_DIR"
rm "$ARCHIVE"

echo "Done!"
