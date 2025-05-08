#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 12:34:49 2019
Run once to prepare the training database
@author: danny
"""
from collections import defaultdict, Counter
import numpy as np
import pickle
import os
# this script saves 3 files, training data with <s> and </s> tokens, the dictionary
# mapping tokens to embedding indices and the word log-frequency dictionary.

# function to save pickled data
def save_obj(obj, loc):
    with open(str(loc) + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def preprocess(train_loc, preproc_train_loc, emb_dict_loc, freq_dict_loc):
    # open the file with the training data
    sentences = []
    with open(train_loc) as file:
        for line in file:
            sentences.append(line)
    print(len(sentences))
    # create a frequency dictionary in order to create the log-frequency feature
    # for the LMER analysis.
    freq_dict = Counter(word for sent in sentences for word in sent.split())
    for key in freq_dict.keys():
        freq_dict[key] = -np.log(freq_dict[key])

    # create the dictionary which will contain the embedding indices
    emb_dict = defaultdict(int)
    ind = 1

    for idx, sent in enumerate(sentences):
        # split the sentence and add beginning and end of sentence tokens
        words = sent.split()
        words.append('</s>')
        words.insert(0, '<s>')
        for w in words:
            if emb_dict[w] == 0:
                emb_dict[w] = ind
                ind += 1
                # join the sentence back together with the two new tokens
        sentences[idx] = ' '.join(words)

    # save the processed text data
    with open(preproc_train_loc, mode='w') as file:
        for line in sentences:
            file.write(line + '\n')

    ## save the index and frequency dictionary
    save_obj(emb_dict, emb_dict_loc)
    save_obj(freq_dict, freq_dict_loc)


# # location of the training database
# train_loc = '/Users/jairiley/Desktop/BOW_Ngrams/corpus/wiki_train_word.txt'

# preproc_train_loc = '/Users/jairiley/Desktop/BOW_Ngrams/transformer/wiki_train_word.txt'

# emb_dict_loc = '/Users/jairiley/Desktop/BOW_Ngrams/transformer/wiki_train_word_indices'

# freq_dict_loc = '/Users/jairiley/Desktop/BOW_Ngrams/transformer/wiki_train_word_freq'

# preprocess(train_loc, preproc_train_loc, emb_dict_loc, freq_dict_loc)


# # location of the validation database
# train_loc = '/Users/jairiley/Desktop/BOW_Ngrams/corpus/wiki_validation_word.txt'

# preproc_train_loc = '/Users/jairiley/Desktop/BOW_Ngrams/transformer/wiki_validation_word.txt'

# emb_dict_loc = '/Users/jairiley/Desktop/BOW_Ngrams/transformer/wiki_train_validation_indices'

# freq_dict_loc = '/Users/jairiley/Desktop/BOW_Ngrams/transformer/wiki_train_validation_freq'

# preprocess(train_loc, preproc_train_loc, emb_dict_loc, freq_dict_loc)

# location of the training word database
model_dir = os.environ.get('SLURM_TMPDIR')
train_loc = '/home/jairiley/projects/def-cepp/jairiley/wiki_train_pos.txt'

preproc_train_loc = '/home/jairiley/projects/def-cepp/jairiley/transformer/wiki_train_pos.txt'
# os.path.join(model_dir, 'wiki_train_word_indices')
emb_dict_loc = os.path.join(model_dir, 'wiki_train_pos_indices')

freq_dict_loc = os.path.join(model_dir, 'wiki_train_pos_freq')

preprocess(train_loc, preproc_train_loc, emb_dict_loc, freq_dict_loc)


# location of the validation word database
train_loc = '/home/jairiley/projects/def-cepp/jairiley/wiki_validation_word_pos.txt'

preproc_train_loc = '/home/jairiley/projects/def-cepp/jairiley/transformer/wiki_validation_pos.txt'

emb_dict_loc = os.path.join(model_dir, 'wiki_train_validation_pos_indices')

freq_dict_loc = os.path.join(model_dir, 'wiki_train_validation_pos_freq')

preprocess(train_loc, preproc_train_loc, emb_dict_loc, freq_dict_loc)



# # location of the training pos database
# train_loc = '/Users/jairiley/Desktop/BOW_Ngrams/corpus/wiki_train_word_pos.txt'

# preproc_train_loc = '/Users/jairiley/Desktop/BOW_Ngrams/transformer/wiki_train_word_pos.txt'

# emb_dict_loc = '/Users/jairiley/Desktop/BOW_Ngrams/transformer/wiki_train_word_pos_indices'

# freq_dict_loc = '/Users/jairiley/Desktop/BOW_Ngrams/transformer/wiki_train_word_pos_freq'

# preprocess(train_loc, preproc_train_loc, emb_dict_loc, freq_dict_loc)


# # location of the validation pos database
# train_loc = '/Users/jairiley/Desktop/BOW_Ngrams/corpus/wiki_validation_word_pos.txt'

# preproc_train_loc = '/Users/jairiley/Desktop/BOW_Ngrams/transformer/wiki_validation_word_pos.txt'

# emb_dict_loc = '/Users/jairiley/Desktop/BOW_Ngrams/transformer/wiki_train_validation_word_pos_indices'

# freq_dict_loc = '/Users/jairiley/Desktop/BOW_Ngrams/transformer/wiki_train_validation_word_pos_freq'

# preprocess(train_loc, preproc_train_loc, emb_dict_loc, freq_dict_loc)


