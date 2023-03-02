import os
import re
import json
import torch
import random
import pickle
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import data_parser

# This dataset will dynamically read data from the folders
class CustomDataset(Dataset):
    def __init__(self, case, port, line_model, input_cols, output_col, indices, device):
        # case: simulation case name in ../Data/
        # port: the number of ports in the line model
        # line_model: name of line model in Tml Simulation
        # input_cols: ['W', 'Trap', 'Length'], for example
        # outpul_col: ['A(1,2)'] or ['P(2,4)'], for example
        
        self.case = case
        self.port = port
        self.home_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
        self.result_dir = os.path.join(self.home_dir, 'Data', '%s' % self.case)
        self.config_dir = os.path.join(self.home_dir, 'Data', 'Config', '%s.json' % line_model)
        
        # define df headers
        self.snp_headers = []
        for i in range(port):
            for j in range(port):
                for s in 'SR', 'SI':
                    self.snp_headers.append('%s(%d,%d)' % (s, i+1, j+1))
        
        # define input cols and parameters input
        self.input_cols = input_cols
        with open(self.config_dir) as f:
            self.config = json.load(f)
        self.parameters_df = data_parser.read_input_feature_xlsx(case, port).loc[indices][input_cols]
        for input_col in input_cols:
            self.parameters_df[input_col] = (self.parameters_df[input_col] - self.config[input_col]['min']) / (self.config[input_col]['max'] - self.config[input_col]['min'])
        
        # define output col
        # if output col is P(i,j), output is [sinP, cosP]
        self.output_col = output_col
        self.index = [eval(i) for i in re.findall(r"(\d)", self.output_col)] # get output indices
        if 'A' in self.output_col:
            self.index = ['A'] + self.index
        elif 'P' in self.output_col:
            self.index = ['P'] + self.index
        elif 'SR' in self.output_col:
            self.index = ['SR'] + self.index
        elif 'SI' in self.output_col:
            self.index = ['SI'] + self.index
        else:
            raise ValueError('Output col must be A(i,j) or P(i,j)!')
        
        self.device = device
        
        snp_df_list = []
        
        for idx in self.parameters_df.index:
            df = pd.read_csv(os.path.join(self.result_dir, idx, 'RLGC', 'TransmissionLine.s%dp' % self.port), \
                             skiprows=self.port + 3, delim_whitespace=True, header=None).loc[:, 1:]
            
            df.columns = self.snp_headers

            # process input
            parameters = torch.Tensor(self.parameters_df.loc[idx])

            # process output
            if self.index[0] == 'A':
                output = np.sqrt(df['SR(%d,%d)'% (self.index[1], self.index[2])] ** 2 + df['SI(%d,%d)'% (self.index[1], self.index[2])] ** 2)
            elif self.index[0] == 'P':
                phase = np.arctan2(df['SI(%d,%d)'% (self.index[1], self.index[2])] , df['SR(%d,%d)'% (self.index[1], self.index[2])])
                sinP = np.sin(phase)
                cosP = np.cos(phase)
                output = pd.concat([sinP, cosP], axis=1)
            elif self.index[0] == 'SR':
                output = df['SR(%d,%d)'% (self.index[1], self.index[2])]
            else: # self.index[0] == 'SI'
                output = df['SI(%d,%d)'% (self.index[1], self.index[2])]
            snp_df_list.append(output)
 
        self.snp_df = pd.concat(snp_df_list)
        self.parameters_tensor = torch.Tensor(self.parameters_df.to_numpy().astype(float)).to(self.device)
        if self.index[0] == 'P':
            self.snp_tensor = torch.Tensor(self.snp_df.to_numpy().reshape(len(self.parameters_df.index), -1, 2)).to(self.device)
        else:
            self.snp_tensor = torch.Tensor(self.snp_df.to_numpy().reshape(len(self.parameters_df.index), -1)).to(self.device)

        return

        
    def __len__(self):
        return len(self.parameters_df.index)

    def __getitem__(self, idx):
        return self.parameters_tensor[idx,:], self.snp_tensor[idx,:]

def generate_indices(case, port,):
    
    current_dir = os.getcwd()
    home_dir = os.path.abspath(os.path.join(current_dir, '..'))
    data_dir = os.path.join(home_dir, 'Data')
    result_dir = os.path.join(data_dir, '%s' % case)
    keys = []
    for dirs in os.listdir(result_dir):
        if os.path.isdir(os.path.join(result_dir, dirs, 'RLGC')):
            snp = 'TransmissionLine.s%dp' % port            
            if os.path.exists(os.path.join(result_dir, dirs, 'RLGC', snp)):
                keys.append(dirs)
    return keys

# This dataset will read the dataframe as a whole
class OneshotDataset(Dataset):
    def __init__(self, df, line_model, input_cols, output_col, indices, device, out_features):
        # line_model: the model config file name

        self.input_cols = input_cols
        self.output_col = output_col
        self.indices = indices
        self.device = device
        
        self.parameters_shape = (-1, len(self.input_cols))
        self.output_shape = (-1, out_features,)

        home_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
        self.config_dir = os.path.join(home_dir, 'Data', 'Config', '%s.json' % line_model)
        
        if 'A' in output_col:
            output = df.loc[indices][output_col]
        elif 'P' in output_col:
            index = [eval(i) for i in re.findall(r"(\d)", output_col)] # get output indices
            output = df.loc[indices][['sinP(%d,%d)' % (index[0], index[1]), 'cosP(%d,%d)' % (index[0], index[1])]]
            self.output_shape += (2, )
        elif 'SR' in output_col:
            output = df.loc[indices][output_col]
        elif 'SI' in self.output_col:
            output = df.loc[indices][output_col]
        else:
            raise ValueError('Output col must be A(i,j) or P(i,j)!')
        
        parameters = df.xs(0, level=1).loc[indices][input_cols]

        # get parameters
        with open(self.config_dir) as f:
            self.config = json.load(f)
        
        for input_col in input_cols:
            parameters[input_col] = (parameters[input_col] - self.config[input_col]['min']) / (self.config[input_col]['max'] - self.config[input_col]['min'])

        self.parameters = torch.Tensor(parameters.to_numpy().reshape(self.parameters_shape)).to(device)
        self.output = torch.Tensor(output.to_numpy().reshape(self.output_shape)).to(device)

        self.output_mean = torch.mean(self.output, axis=0, keepdims=True)
        self.output_std = torch.std(self.output, axis=0, keepdims=True)

        self.length = self.parameters.shape[0]

        return
    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.parameters[idx, ...], self.output[idx, ...]
    

# This dataset will read the dataframe as a whole
class OneshotDataset_v2(Dataset):
    def __init__(self, df, line_model, input_cols, index, indices, device, out_features):
        # line_model: the model config file name

        self.input_cols = input_cols
        self.output_col = []
        for k in 'A', 'sinP', 'cosP', 'SR', 'SI':
            self.output_col.append('%s(%d,%d)' % (k, index[0], index[1]))

        self.indices = indices
        self.device = device
        
        self.parameters_shape = (-1, len(self.input_cols))
        self.output_shape = (-1, out_features, 5)

        home_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
        self.config_dir = os.path.join(home_dir, 'Data', 'Config', '%s.json' % line_model)
        
        output = df.loc[indices][self.output_col]

        parameters = df.xs(0, level=1).loc[indices][input_cols]

        # get parameters
        with open(self.config_dir) as f:
            self.config = json.load(f)
        
        for input_col in input_cols:
            parameters[input_col] = (parameters[input_col] - self.config[input_col]['min']) / (self.config[input_col]['max'] - self.config[input_col]['min'])

        self.parameters = torch.Tensor(parameters.to_numpy().reshape(self.parameters_shape)).to(device)
        self.output = torch.Tensor(output.to_numpy().reshape(self.output_shape)).to(device)

        self.length = self.parameters.shape[0]

        return
    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.parameters[idx, ...], self.output[idx, ...]