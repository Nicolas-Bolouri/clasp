#!/bin/bash

echo "Downloading CLASP datasets, embeddings, and models from internet archive..."
curl -L -o clasp_data.zip https://archive.org/compress/clasp_data

echo "Unzipping..."
unzip clasp_data.zip

echo "Done. Data available in $(pwd)/clasp_data"
