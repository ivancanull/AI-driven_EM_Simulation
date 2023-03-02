import os
import numpy as np
import pickle
import random

def get_model_path(output_col, postfix):   
    # Function to save the trained model to disk.

    # Define saved filepath
    if postfix != '':
        # checking if the directory demo_folder 
        # exist or not.
        model_dir = os.path.join("../Models/", postfix)
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)
        path = os.path.join(model_dir, output_col + '_best_model.pt')
    else:
        path = os.path.join("../Models/", output_col  + '_best_model.pt')

    return path
    
def get_indices(df, filename, read_idx=True):
    # Randomly shuffle the indices
    index_list = list(dict.fromkeys(df.index.get_level_values(0)))
    index_file = '../Data/Indices/index_%s.pkl' % filename
    if read_idx == True:
        with open(index_file, 'rb') as f:
            indices = pickle.load(f)
    else:
        random.seed(42)
        np.random.shuffle(index_list)
        
        # Split the indices into 80% training set, 10% testing set and 10% validation set
        indices = {}
        indices['train_idx'] = index_list[:int(len(index_list) * 0.8)]
        indices['val_idx'] = index_list[int(len(index_list) * 0.8):int(len(index_list) * 0.9)]
        indices['test_idx'] = index_list[int(len(index_list) * 0.9):]

        with open(index_file, 'wb') as f:
            pickle.dump(indices, f)
    
    return indices