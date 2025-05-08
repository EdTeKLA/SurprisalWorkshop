#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 16:36:50 2019
This script loads the trained language models and uses them to extract
surprisal values for the human reading data.
@author: danny
"""
import torch
import sys
import pickle
import numpy as np
import pandas as pd
import os
import argparse

# make sure this script is in the same folder as the functions folder of the
# nwp project
sys.path.append('IncrementalReadingLanguageModelling/src/language_models/transformer/functions')

from collections import defaultdict
from encoders import *
from prep_text import word_2_index

data_type = "pos"
parser = argparse.ArgumentParser(description='Get file name')
parser.add_argument('-data_loc', type=str,
                    default='IncrementalReadingLanguageModelling/data/stimuli/L1/stimuli_L1_data_type.txt',
                    help='location of the sentences we want surprisal values for')

parser.add_argument('-output_id', type=str,
                    default='L1',
                    help='identifying value for output')
args = parser.parse_args()
lrs = [0.0002, 0.001, 0.005, 0.025, 0.125]
batch_sizes = [5, 10, 20, 40, 80]
# lrs = [0.126]
# batch_sizes = [5]

for lr in lrs:
    for batch_size in batch_sizes:
        if batch_size == 80 and lr == 0.125:
            continue
        model_loc = f'IncrementalReadingLanguageModelling/src/language_models/transformer/parameters_{data_type}_{lr}_{batch_size}'
        # location of a pre-trained model
        # location of the sentences to be encoded.
        data_loc = args.data_loc
        dict_loc = f'IncrementalReadingLanguageModelling/data/wiki/sequences/{data_type}/transformer/wiki_train_{data_type}_indices'

        # list all the pretrained models
        model_list = [x for x in os.walk(model_loc)]
        model_list = [os.path.join(x[0], y) for x in model_list for y in x[2] if not '.out' in y]
        model_list.sort()


        # function to load a pickled dictionary with the indices of each possible token
        def load_obj(loc):
            with open(loc + '.pkl', 'rb') as f:
                return pickle.load(f)


        nwp_dict = load_obj(dict_loc)
        dict_size = len(nwp_dict) + 1


        def index_2_word(dictionary, indices):
            rev_dictionary = defaultdict(str)
            for x, y in dictionary.items():
                rev_dictionary[y] = x
            sentences = [[rev_dictionary[int(i)] for i in ind] for ind in indices]
            return (sentences)


        # function to produce the surprisal ratings for the test sentences
        def calc_surprisal(data_loc, model):
            sent = []
            with open(data_loc) as file:
                for line in file:
                    # split the sentence into tokens
                    sent.append(['<s>'] + line.split() + ['</s>'])
            # convert text to indices,
            original_sent = sent
            sent, l = word_2_index(sent, len(sent), nwp_dict)

            # get the predictions and targets for this sentence
            print(sent)
            print(l)
            predictions, targets = model(torch.FloatTensor(sent), l)

            # convert the predictions to surprisal (negative log softmax)
            surprisal = -torch.log_softmax(predictions, dim=2).squeeze()
            # extract only the surpisal ratings for the target words
            surprisal = surprisal.gather(-1, targets.unsqueeze(-1)).squeeze()
            # finally remove any padding applied by word_2_index and remove end of
            # sentence prediction
            surprisal = surprisal.data.numpy()
            surprisal = [s[:l - 2] for s, l in zip(surprisal, l)]
            return (surprisal, targets, original_sent)


        # set words followed by a comma to nan as they were excluded in the original
        # experiment.
        def clean_surprisal(surprisal, targets, original_sent):
            # for s, t in zip(surprisal, targets):
            #     # iterate over all test data to find occurences of punctuation in the
            #     # targets
            #     for index, word in enumerate(t):
            #         if ',' in word:
            #             # set the previous token to nan
            #             s[index - 1] = np.nan
            # construct lists of which words to keep, that is set keep to false for
            # all items for which the previous value was nan. Set the first item to
            # True by default (has no previous item)
            keep = [[True] + [not (np.isnan(s[ind - 1])) for ind in range(1, len(s))] for s in surprisal]
            # now keep only those surprisal values we need to keep
            surprisal = [[s[x] for x in range(len(s)) if k[x]] for s, k in zip(surprisal, keep)]
            targets = [[t[x] for x in range(len(s)) if k[x]] for s, t, k in zip(surprisal, targets, keep)]
            original_sent = [t[1:-2] + ['.'] for t in original_sent]

            # add sentence and word position indices to the surprisal ratings and
            # convert to DataFrame object
            surprisal = [(sent_index + 1, word_index + 1, word) for sent_index,
                                                                    sent in enumerate(surprisal) for word_index,
                                                                                                     word in
                         enumerate(sent)
                         ]
            targets = [word for t in targets for word in t]
            original_sent = [word for t in original_sent for word in t]

            surprisal = pd.DataFrame(np.array(surprisal))
            targets = pd.Series(np.array(targets))

            return surprisal, targets, original_sent


        # config settings for the models;
        # tf_1l_config = {'embed': {'n_embeddings': dict_size,
        #                           'embedding_dim': 400, 'sparse': False,
        #                           'padding_idx': 0
        #                           },
        #                 'tf': {'in_size': 400, 'fc_size': 1024, 'n_layers': 1,
        #                        'h': 8, 'max_len': 54
        #                        },
        #                 'cuda': False
        #                 }
        tf_2l_config = {'embed': {'n_embeddings': dict_size,
                                  'embedding_dim': 400, 'sparse': False,
                                  'padding_idx': 0
                                  },
                        'tf': {'in_size': 400, 'fc_size': 1024, 'n_layers': 2,
                               'h': 8, 'max_len': 54
                               },
                        'cuda': False
                        }

        # tf_4l_config = {'embed': {'n_embeddings': dict_size,
        #                           'embedding_dim': 400, 'sparse': False,
        #                           'padding_idx': 0
        #                           },
        #                 'tf': {'in_size': 400, 'fc_size': 1024, 'n_layers': 4,
        #                        'h': 8, 'max_len': 54
        #                        },
        #                 'cuda': False
        #                 }
        #
        # gru_1l_config = {'embed': {'n_embeddings': dict_size, 'embedding_dim': 400,
        #                            'sparse': False, 'padding_idx': 0
        #                            },
        #                  'max_len': 54,
        #                  'rnn': {'in_size': 400, 'hidden_size': 500,
        #                          'n_layers': 1, 'batch_first': True,
        #                          'bidirectional': False, 'dropout': 0
        #                          },
        #                  'lin': {'hidden_size': 400
        #                          },
        #                  'att': {'in_size': 500, 'heads': 10
        #                          },
        #                  'cuda': False
        #                  }
        #
        # gru_2l_config = {'embed': {'n_embeddings': dict_size, 'embedding_dim': 400,
        #                            'sparse': False, 'padding_idx': 0
        #                            },
        #                  'max_len': 54,
        #                  'rnn': {'in_size': 400, 'hidden_size': 500,
        #                          'n_layers': 2, 'batch_first': True,
        #                          'bidirectional': False, 'dropout': 0
        #                          },
        #                  'lin': {'hidden_size': 400
        #                          },
        #                  'att': {'in_size': 500, 'heads': 10
        #                          },
        #                  'cuda': False
        #                  }

        # create the models
        # tf_1l = nwp_transformer(tf_1l_config)
        tf_2l = nwp_transformer(tf_2l_config)
        # tf_4l = nwp_transformer(tf_4l_config)

        # gru_1l = nwp_rnn_encoder(gru_1l_config)
        # gru_2l = nwp_rnn_encoder(gru_2l_config)
        # gru_tf = nwp_rnn_tf_att(gru_1l_config)

        # encoder_models = [tf_1l, tf_2l, tf_4l, gru_1l, gru_2l, gru_tf]
        encoder_models = [tf_2l]
        ###############################################################################
        data = pd.DataFrame()

        for model_loc in model_list:
            # load the pretrained model
            model = torch.load(model_loc, map_location='cpu')
            print(model_loc)
            for i, enc in enumerate(encoder_models):
                nwp_model = enc
                try:
                    nwp_model.load_state_dict(model)
                    break
                except:
                    continue
            # set requires grad to false for faster encoding
            for param in nwp_model.parameters():
                param.requires_grad = False
            # set to eval mode to disable dropout
            nwp_model.eval()
            # load the dictionary of indexes and create a reverse lookup dictionary so
            # we can look up target words by their index
            index_dict = load_obj(dict_loc)
            word_dict = defaultdict(str)
            for x, y in index_dict.items():
                word_dict[y] = x
            # get all the surprisal values and the target sequence (inputs shifted to
            # the left)
            surprisal, targets, original_sent = calc_surprisal(data_loc, nwp_model)
            # convert the target indices back to words
            targets = index_2_word(nwp_dict, targets)

            surprisal, targets, original_sent = clean_surprisal(surprisal, targets, original_sent)
            # create a unique name for the current surprisal values by combining the
            # model number with the nr of training samples of the model. N.B. the
            # indices here are hard coded for my specific folder structure and file
            # naming convention.
            surp_name = model_loc.split('/')[-2] + '_' + model_loc.split('/')[-1].split('.')[-1]
            surprisal.columns = ['sent_nr', 'word_pos', surp_name]

            item_nr = []
            for x, y in zip(surprisal.sent_nr, surprisal.word_pos):
                x = x * 100
                item_nr.append(int(x + y))
            surprisal['item'] = pd.Series(item_nr)
            surprisal['word'] = targets
            surprisal['original word'] = original_sent
            if not data.empty:
                data[surp_name] = data.join(surprisal[[surp_name, 'item']].set_index('item'),
                                            on='item')[surp_name]
            else:
                data = surprisal

        ###############################################################################
        # now sort the column names in in loading state_dict for nwp_transformer:
        col_names = data.columns.tolist()
        print(col_names)
        models = col_names[2:3] + col_names[6:]
        models.sort()
        col_names = col_names[0:2] + col_names[3:6] + models
        data = data[col_names]
        print(models)
        # round the surprisal to 4 decimals and convert the sent_nr and word_pos from
        # float to in
        data[models] = data[models].round(4)
        data.sent_nr = data.sent_nr.astype(int)
        data.word_pos = data.word_pos.astype(int)
        output_id = args.output_id
        output_path = f'IncrementalReadingLanguageModelling/output/transformer/{data_type}/hyperparameter_tuning/{output_id[0]}/{output_id}_{lr}_{batch_size}.csv'
        data.to_csv(path_or_buf=output_path, index=False)

