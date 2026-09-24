#! /bin/bash

for dataset in "cqadupstack-gaming"
do
    echo "Downloading $dataset"
    download-pre-embedded $dataset ../../data/pre-embedded/${dataset}
done
