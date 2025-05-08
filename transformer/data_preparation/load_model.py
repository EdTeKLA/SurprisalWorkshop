#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 16:36:50 2019
This script loads the trained language models and uses them to extract 
surprisal values for the human reading data.
@author: danny
"""

import argparse
import torch
import sys
import pickle
import numpy as np
import pandas as pd
import os

from collections import defaultdict

# Add your project functions path
sys.path.append('/content/language_models/transformer/functions')

from encoders import nwp_transformer  # make sure this function is defined there
from prep_text import word_2_index

# Argument parser
parser = argparse.ArgumentParser(description='Get file name')
parser.add_argument('-data_loc', type=str, default='/content/language_models/transformer/wiki_validation_pos.txt')
parser.add_argument('-model_loc', type=str, default='/content/language_models/transformer/nwp_model_1_11711340')
parser.add_argument('-output_id', type=str, default='surprisal_output')
parser.add_argument('-dict_loc', type=str, default='/content/language_models/transformer/wiki_train_word_pos_indices')
parser.add_argument('-output_file', type=str, default='/content/language_models/transformer/surprisal_output.csv')
args = parser.parse_args()

# Set device to CPU
device = torch.device("cpu")

# Pretrained model & data
model_loc = args.model_loc
dict_loc = args.dict_loc
data_loc = args.data_loc

def load_obj(loc):
    with open(loc + '.pkl', 'rb') as f:
        return pickle.load(f)

nwp_dict = load_obj(dict_loc)
dict_size = len(nwp_dict) + 1

def index_2_word(dictionary, indices):
    rev_dictionary = defaultdict(str)
    for x, y in dictionary.items():
        rev_dictionary[y] = x
    return [[rev_dictionary[int(i)] for i in ind] for ind in indices]

def calc_surprisal(data_loc, model):
    sent = []
    with open(data_loc) as file:
        for line in file:
            sent.append(['<s>'] + line.strip().split() + ['</s>'])


    original_sent = [s[1:] for s in sent ]


    sent_idx, lengths = word_2_index(sent, len(sent), nwp_dict)

    # Move tensors to CPU
    sent_idx = torch.FloatTensor(sent_idx).to(device)
    lengths = torch.LongTensor(lengths).to(device)

    predictions, targets = model(sent_idx, lengths)

    # Calculate surprisal
    surprisal = -torch.log_softmax(predictions, dim=2).squeeze()
    surprisal = surprisal.gather(-1, targets.unsqueeze(-1)).squeeze()
    surprisal = surprisal.detach().cpu().numpy()

    targets = targets.cpu().numpy()
    surprisal = [s[:l-1] for s, l in zip(surprisal, lengths.cpu())]  # skip <s>, drop </s>
    targets = [t[:l-1] for t, l in zip(targets, lengths.cpu())]  
    return surprisal, targets, original_sent

def clean_surprisal(surprisal, targets, original_sent):
    # print(surprisal,'\n',targets,'\n', original_sent)
    # Create mask where surprisal is not NaN
    keep = [[not np.isnan(val) for val in s] for s in surprisal]

    # Apply the mask
    surprisal = [[s[i] for i in range(len(s)) if k[i]] for s, k in zip(surprisal, keep)]
    targets = [[t[i] for i in range(len(t)) if k[i]] for t, k in zip(targets, keep)]
    original_sent = [[w[i] for i in range(len(w)) if k[i]] for w, k in zip(original_sent, keep)]


    # Flatten
    rows = [(i+1, j+1, val) for i, sent in enumerate(surprisal) for j, val in enumerate(sent)]
    words = [w for sent in targets for w in sent]
    orig_words = [w for sent in original_sent for w in sent]

    # print(rows)
    df = pd.DataFrame(rows, columns=["sent_nr", "word_pos", "surprisal"])
    df["item"] = df.sent_nr * 100 + df.word_pos
    df["word"] = index_2_word(nwp_dict, [[w] for w in words])
    # print(df["word"])
    df["original word"] = orig_words

    return df

# Define model config
model_config = {
    'embed': {'n_embeddings': dict_size, 'embedding_dim': 400, 'sparse': False, 'padding_idx': 0},
    'tf': {'in_size': 400, 'fc_size': 1024, 'n_layers': 2, 'h': 8, 'max_len': 54},
    'cuda': False # Removed for CPU
}

# Load model
model = nwp_transformer(model_config)
model.load_state_dict(torch.load(model_loc, map_location=device))
model.to(device)
model.eval()

# Calculate
surprisal, targets, original_sent = calc_surprisal(data_loc, model)
df = clean_surprisal(surprisal, targets, original_sent)

# Round and save
df["surprisal"] = df["surprisal"].round(4)
df.sent_nr = df.sent_nr.astype(int)
df.word_pos = df.word_pos.astype(int)

df.to_csv(args.output_file, index=False)
# print(f"Saved surprisal values to {args.output_file}")
