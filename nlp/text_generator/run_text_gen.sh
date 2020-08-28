#!/bin/bash

# Training
#./TextGenerator.py \
#	--mode training \
#	--training_data data/republic.txt \
#	--model data/republic_model.h5 \
#	--map data/republic_word2int_map.pkl \
#	--seed_text data/republic_seed.txt

# Inference 
./TextGenerator.py \
	--mode inference \
	--training_data data/republic.txt \
	--model data/republic_model.h5 \
	--map data/republic_word2int_map.pkl \
	--seed_text data/republic_seed.txt
