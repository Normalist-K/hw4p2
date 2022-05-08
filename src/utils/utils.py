import os
import random
import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import Levenshtein

LETTER_LIST = ['<sos>', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', \
               'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', "'", ' ', '<eos>']

def create_dictionaries(letter_list):
    '''
    Create dictionaries for letter2index and index2letter transformations
    based on LETTER_LIST
    Args:
        letter_list: LETTER_LIST
    Return:
        letter2index: Dictionary mapping from letters to indices
        index2letter: Dictionary mapping from indices to letters
    '''
    letter2index = dict()
    index2letter = dict()
    
    # TODO
    for idx, letter in enumerate(letter_list):
        letter2index[letter] = idx
        index2letter[idx] = letter
    
    return letter2index, index2letter

def transform_index_to_letter(batch_indices):
    '''
    Transforms numerical index input to string output by converting each index 
    to its corresponding letter from LETTER_LIST

    Args:
        batch_indices: List of indices from LETTER_LIST with the shape of (N, )
    
    Return:
        transcripts: List of converted string transcripts. This would be a list with a length of N
    '''
    transcripts = []
    letter2index, index2letter = create_dictionaries(LETTER_LIST)
    # TODO
    for indices in batch_indices:
        transcript_list = []
        for idx in indices:
            if idx == letter2index['<eos>']:
                break
            transcript_list.append(index2letter[idx])
        transcript = ''.join(transcript_list[1:])
        transcripts.append(transcript)
    return transcripts


def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def load_dataloader(cfg):
    print(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    dm = hydra.utils.instantiate(cfg.datamodule)
    dm.prepare_data()
    dm.setup(stage=None)
    train_loader = dm.train_dataloader()
    valid_loader = dm.val_dataloader()
    test_loader = dm.predict_dataloader()

    if cfg.trainer.scheduler.name == 'CosineAnnealingLR':
        cfg.len_train_loader = len(train_loader)

    return train_loader, valid_loader, test_loader

def save_submission(cfg, results):
    sub_dict = {'id': [], 'predictions': []}
    for i, label in enumerate(results):
        sub_dict['id'].append(i)
        sub_dict['predictions'].append(label)

    sub_df = pd.DataFrame(sub_dict)
    print(len(sub_df))
    sub_dir = cfg.path.submissions
    sub_name = f'{cfg.name}-{cfg.dt_string}.csv'
    sub_df.to_csv(os.path.join(sub_dir, sub_name), index=False)
    print(f'{sub_name} saved.')

def calculate_levenshtein(x_prob, y):
    x_string = torch.argmax(x_prob, dim=2).detach().cpu().numpy()
    y_string = y.detach().cpu().numpy()
    x_string = transform_index_to_letter(x_string)
    y_string = transform_index_to_letter(y_string)
    
    dist = 0
    for b in range(len(y_string)):
        dist += Levenshtein.distance(x_string[b], y_string[b])
    dist = dist / len(y_string)
    return dist