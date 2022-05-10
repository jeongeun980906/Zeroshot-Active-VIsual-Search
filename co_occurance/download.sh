#!/bin/bash

wget https://storage.googleapis.com/ai2-mosaic-public/projects/mosaic-kgs/comet-atomic_2020_BART.zip
unzip comet-atomic_2020_BART.zip

rm -rf ./__MACOSX
rm -rf ./comet-atomic_2020_BART/.DS_Store
rm -rf ./comet-atomic_2020_BART/.*swp

python -m spacy download en_core_web_lg
python install.py
