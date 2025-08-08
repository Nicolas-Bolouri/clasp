#!/bin/bash

echo "Downloading CLASP datasets, embeddings, and models from internet archive..."
curl -L -o clasp_data.zip https://archive.org/compress/clasp_data

echo "Unzipping..."
unzip clasp_data.zip

echo "Cleaning up..."
rm -f clasp_data_archive.torrent
rm -f clasp_data_archive.zip
rm -f clasp_data_files.xml
rm -f clasp_data_meta.sqlite
rm -f clasp_data_meta.xml
rm -f clasp_data.zip

echo "Done. Data available in $(pwd)/clasp_data"
