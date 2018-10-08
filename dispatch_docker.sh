#!/usr/bin/env bash
sudo nvidia-docker build . -f Dockerfile --tag convolutional_sequence_to_sequence_learning
sudo nvidia-docker run -v `realpath MT_Data`:/data/ \
 --rm convolutional_sequence_to_sequence_learning \
 --mode train \
 --config config.yml \
 --data /data/europarl-v7.tfrecords \
 --de_vocab /data/wiki.de_tokens.txt \
 --en_vocab /data/wiki.en_tokens.txt \
 --pre_trained_embedding_de /data/wiki.de_embeddings.npy \
 --pre_trained_embedding_en /data/wiki.en_embeddings.npy \
 --temp /data/
