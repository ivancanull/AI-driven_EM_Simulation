#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import re
import pandas as pd
import numpy as np


# from tqdm import tqdm

### parse the sweeeped data ###
def data_parse(case, model, port, optimetric, data_dir='./'):
    
    current_dir = os.getcwd()
    # Get the absolute path of the upper directory
    home_dir = os.path.abspath(os.path.join(current_dir, '..'))
    data_dir = os.path.join(home_dir, 'Data', 'Tml_sweep', 'Tml_sweep', case, model, 'Optimetrics', optimetric)

    data_path = os.path.join(home_dir, data_dir)

    rlgc_df_list = []
    snp_df_list = []
    keys = []
    snp_headers = []
    
    for i in range(port):
        for j in range(port):
            for s in 'SR', 'SI':
                snp_headers.append('%s(%d,%d)' % (s, i+1, j+1))

    for dirs in os.listdir(data_path):
        
        # skip with
        if os.path.isdir(os.path.join(data_path, dirs)):
            
            # read rlgc
            rlgc = dirs + '.rlgc'
           
            # read snp
            snp = dirs + '.s%dp' % port
            if os.path.exists(os.path.join(data_path, dirs, snp)) and os.path.exists(os.path.join(data_path, dirs, rlgc)):
                keys.append(dirs)
                rlgc_df_list.append(pd.read_csv(os.path.join(data_path, dirs, rlgc), skiprows=2, delim_whitespace=True))

                snp_df = pd.read_csv(os.path.join(data_path, dirs, snp), skiprows=port+3, delim_whitespace=True, header=None).loc[:, 1:]
                snp_df.columns = snp_headers
                snp_df_list.append(snp_df)

    df = pd.concat([pd.concat(rlgc_df_list, keys=keys), pd.concat(snp_df_list, keys=keys)], axis=1)

    vsf_file = os.path.join('Data', 'Tml_sweep', 'Tml_sweep', case, model, model+'.vsf')

    # open the file 
    with open(os.path.join(home_dir, vsf_file), "r") as file:
        
        # read the text
        text = file.read()
        
        # create the regular expression pattern
        start_pattern = ".VARIATIONID"
        end_pattern = ".ENDSRP_RESULTITEM"    

        lines = text.split("\n")
        
        parse_en = False
        # search for the matched line
        for i, line in enumerate(lines):
            if re.search(start_pattern, line):
                parse_en = True
                parameters = {}
                
            if re.search(end_pattern, line):
                if node_ID in df.index:
                    for key in parameters:
                        df.loc[node_ID, key] = parameters[key]
                parse_en = False
                
            if parse_en:
                # read the parameters
                parameter_pattern = r".VARCOMBINATION \"%s.([a-zA-Z]+)\"\|\"([\d.]+)\""%('Differential_Stripline')
                node_ID_pattern = r".SIGNATURE \"(\w+)\""
                
                # using re.search() to search for the first patterns
                parameter_match = re.search(parameter_pattern, line)
                node_ID_match = re.search(node_ID_pattern, line)
                
                # if matched, extract the parameter name, value and node IDs
                if parameter_match:
                    parameters[parameter_match.group(1)] = float(parameter_match.group(2))
                elif node_ID_match:
                    node_ID = node_ID_match.group(1)
        
    return df

### parse the mixed line batch data ###
def data_parse_v2(case, port,):
    
    current_dir = os.getcwd()
    home_dir = os.path.abspath(os.path.join(current_dir, '..'))
    data_dir = os.path.join(home_dir, 'Data')
    case_dir = os.path.join(data_dir, case)
    excel_dir = os.path.join(data_dir, '%s.xlsx' % case)
    result_dir = os.path.join(data_dir, '%s' % case)
    
    para_df = pd.read_excel(excel_dir, sheet_name='Mixed_N_Line_Stripline', skiprows=23)
    
    # Set new df
    new_para_df = para_df.T.loc[para_df.T.index[1:]].copy()
    new_para_df.columns = para_df.T.loc['batch list'].to_numpy()
    new_para_df = new_para_df.set_index('save dir file name')
    new_para_df.index.name = None
    
    
    # Get the absolute path of the upper directory

    # rlgc_df_list = []
    snp_df_list = []
    keys = []
    snp_headers = []
    
    for i in range(port):
        for j in range(port):
            for s in 'SR', 'SI':
                snp_headers.append('%s(%d,%d)' % (s, i+1, j+1))

    for dirs in os.listdir(result_dir):
                
        # skip with
        if os.path.isdir(os.path.join(result_dir, dirs, 'RLGC')):
            
            # read rlgc
            # rlgc = 'TransmissionLine.rlgc'
            
            # read snp
            snp = 'TransmissionLine.s%dp' % port
            
            if os.path.exists(os.path.join(result_dir, dirs, 'RLGC', snp)):
                keys.append(dirs)
                # rlgc_df_list.append(pd.read_csv(os.path.join(data_path, dirs, rlgc), skiprows=2, delim_whitespace=True))

                snp_df = pd.read_csv(os.path.join(result_dir, dirs, 'RLGC', snp), skiprows=port+3, delim_whitespace=True, header=None).loc[:, 1:]
                snp_df.columns = snp_headers
                
                for p in new_para_df.columns:
                    if p == 'W':
                        snp_df.loc[:, p] = float(new_para_df.loc[dirs, p].split(',')[0])
                    else:
                        snp_df.loc[:, p] = float(new_para_df.loc[dirs, p])
                
                snp_df_list.append(snp_df)
                
    # df = pd.concat([pd.concat(rlgc_df_list, keys=keys), pd.concat(snp_df_list, keys=keys)], axis=1)
    df = pd.concat(snp_df_list, keys=keys)

    return df
"""
Consider a dataset of the following structures:

-- dataset
    -- subset1
        -- line1
            --
        -- line2
            --
        ...
        script.xlsx
    -- subset2
        ...

-> save to an entire csv
"""


def data_parse_v3(dataset_dir, port, dataset_name):

    keys = []
    snp_headers = []
    para_df_list = []
    snp_df_list = []

    for i in range(port):
        for j in range(port):
            for s in 'SR', 'SI':
                snp_headers.append('%s(%d,%d)' % (s, i+1, j+1))
                
    for subset in os.listdir(dataset_dir):
        
        filepath = os.path.join(dataset_dir, '%s_%s.zip' % (dataset_name, subset))
        if os.path.exists(filepath):
            continue
        
        subset_dir = os.path.join(dataset_dir, subset)
        excel_files = [f for f in os.listdir(subset_dir) if f.endswith('.xlsx')]
        print(excel_files)
        # read para_df
        for excel_file in excel_files:
            df = pd.read_excel(os.path.join(dataset_dir, subset, excel_file), sheet_name='Mixed_N_Line_Stripline', skiprows=23)
            new_df = df.T.loc[df.T.index[1:]].copy()
            new_df.columns = df.T.loc['batch list'].to_numpy()
            new_df = new_df.set_index('save dir file name')
            new_df.index.name = None
            para_df_list.append(new_df)
        
        para_df = pd.concat(para_df_list)

        # read lines
        for dir in os.listdir(subset_dir):
            if os.path.isdir(os.path.join(subset_dir, dir, 'RLGC')):
                snp = 'TransmissionLine.s%dp' % port
                if os.path.exists(os.path.join(subset_dir, dir, 'RLGC', snp)):
                    keys.append(dir)
                    snp_df = pd.read_csv(os.path.join(subset_dir, dir, 'RLGC', snp), skiprows=port+3, delim_whitespace=True, header=None).loc[:, 1:]
                    snp_df.columns = snp_headers

                    for p in para_df.columns:
                        if p == 'W':
                            snp_df.loc[:, p] = float(para_df.loc[dir, p].split(',')[0])
                        else:
                            snp_df.loc[:, p] = float(para_df.loc[dir, p])
                    
                    snp_df_list.append(snp_df)

        df = pd.concat(snp_df_list, keys=keys)
        df.to_pickle(filepath, compression='zip')

    return


def read_input_feature_xlsx(case, port):
    
    current_dir = os.getcwd()
    home_dir = os.path.abspath(os.path.join(current_dir, '..'))
    data_dir = os.path.join(home_dir, 'Data')
    case_dir = os.path.join(data_dir, case)
    excel_dir = os.path.join(data_dir, '%s.xlsx' % case)
    result_dir = os.path.join(data_dir, '%s' % case)
    
    para_df = pd.read_excel(excel_dir, sheet_name='Mixed_N_Line_Stripline', skiprows=23)
    
    # Set new df
    new_para_df = para_df.T.loc[para_df.T.index[1:]].copy()
    new_para_df.columns = para_df.T.loc['batch list'].to_numpy()
    new_para_df = new_para_df.set_index('save dir file name')
    new_para_df.index.name = None
    
    def split_func(x):
        return float(x.split(',')[0])
    new_para_df['W'] = new_para_df['W'].apply(split_func)
    
    return new_para_df


def calculate_amp_phase(df, unwrap=True):
    # Calculate amplitude and phase
    for i in 1, 2, 3, 4:
        for j in 1, 2, 3, 4:
            idx = '('+str(i)+','+str(j)+')'
            df['A'+idx] = np.sqrt(df['SR'+idx] ** 2 + df['SI'+idx] ** 2)
            df['P'+idx] = np.arctan2(df['SI'+idx] , df['SR'+idx])
            if unwrap:
                index_list = list(dict.fromkeys(df.index.get_level_values(0)))
                for k in index_list:
                    df.loc[k]['P'+idx] = np.unwrap(df.loc[k]['P'+idx])
                    
    return df

def calculate_sin_cos(df):
    for i in 1, 2, 3, 4:
        for j in 1, 2, 3, 4:
            idx = '('+str(i)+','+str(j)+')'
            df['sinP'+idx] = np.sin(df['P'+idx])
            df['cosP'+idx] = np.cos(df['P'+idx])
            
    return df


def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=str, default='case1', help='project case name')
    parser.add_argument('--model', type=str, default='Differential_Stripline', help='line model for EM simulation')
    parser.add_argument('--optimetric', type=str, default='SweepSetup1', help='optimetric sweep name')
    parser.add_argument('--dir', type=str, default='./', help='data storage directory')
    parser.add_argument('--port', type=int, default=4, help='port number in the model')
    
    return parser.parse_args()
    
def main(args):
    
    case = args.case
    EM_model = args.model
    optimetric = args.optimetric
    port = args.port
    
    dataname = '%s_%s_%d_ports_%s' %(case, EM_model, port, optimetric)
    data_dir = os.path.abspath(args.dir)
    
    df = calculate_amp_phase(data_parse(case, EM_model, port, optimetric, data_dir), unwrap=True)
    current_dir = os.getcwd()
    
    # Get the absolute path of the upper directory
    saved_dir = os.path.abspath(os.path.join(data_dir, 'Data', '%s.pkl' % dataname))
    df.to_pickle(saved_dir) 

def parse_data():
    dataset_dir = '/media/cadlab/Dailow2/Tml_data/2-Line_WTL'
    dataset_name = '2-Line_WTL'
    df = data_parse_v3(dataset_dir, 4, dataset_name)

def check_data():
    dataset_dir = '/media/cadlab/Dailow2/Tml_data/2-Line_WTL'
    dataset_name = '2-Line_WTL'
    pickle_name = os.path.join(dataset_dir, '2-Line_WTL_2-Line-0WTL.zip')

    df = pd.read_pickle(pickle_name)
    df.iloc[0:1000].to_csv(os.path.join(dataset_dir, 'test.csv'), index_label=[0,1])
    print(df.iloc[0:1000].index)


if __name__ == "__main__":
    
    parse_data()

