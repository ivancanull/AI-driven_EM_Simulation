#!/usr/bin/env python
# coding: utf-8

import os
import re
import pandas as pd
import numpy as np


# from tqdm import tqdm

### PRASE THE DATA ###

def data_parse():

    current_dir = os.getcwd()
    # Get the absolute path of the upper directory
    home_dir = os.path.abspath(os.path.join(current_dir, '..'))
    data_dir = os.path.join('Data', 'Tml_sweep', 'Tml_sweep', 'case1', 'Differential_Stripline', 'Optimetrics', 'SweepSetup1')

    data_path = os.path.join(home_dir, data_dir)

    rlgc_df_list = []
    s4p_df_list = []
    keys = []
    s4p_headers = ["SR(1,1)", "SI(1,1)", "SR(1,2)", "SI(1,2)", "SR(1,3)", "SI(1,3)", "SR(1,4)", "SI(1,4)",
                "SR(2,1)", "SI(2,1)", "SR(2,2)", "SI(2,2)", "SR(2,3)", "SI(2,3)", "SR(2,4)", "SI(2,4)",
                "SR(3,1)", "SI(3,1)", "SR(3,2)", "SI(3,2)", "SR(3,3)", "SI(3,3)", "SR(3,4)", "SI(3,4)",
                "SR(4,1)", "SI(4,1)", "SR(4,2)", "SI(4,2)", "SR(4,3)", "SI(4,3)", "SR(4,4)", "SI(4,4)"]
    for dirs in os.listdir(data_path):
        
        # skip with
        if os.path.isdir(os.path.join(data_path, dirs)):
            
            keys.append(dirs)
            # read rlgc
            rlgc = dirs + '.rlgc'
            new_varnew_var = rlgc_df_list.append(pd.read_csv(os.path.join(data_path, dirs, rlgc), skiprows=2, delim_whitespace=True))
            

            # read s4p
            s4p = dirs + '.s4p'
            s4p_df = pd.read_csv(os.path.join(data_path, dirs, s4p), skiprows=7, delim_whitespace=True, header=None).loc[:, 1:]
            s4p_df.columns = s4p_headers
            s4p_df_list.append(s4p_df)

    df = pd.concat([pd.concat(rlgc_df_list, keys=keys), pd.concat(s4p_df_list, keys=keys)], axis=1)

    vsf_file = os.path.join('Data', 'Tml_sweep', 'Tml_sweep', 'case1', 'Differential_Stripline', 'Differential_Stripline.vsf')

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
                parameter_pattern = r".VARCOMBINATION \"Differential_Stripline.([a-zA-Z]+)\"\|\"([\d.]+)\""
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

def calculate_amp_phase(df):
    # Calculate amplitude and phase
    for i in 1, 2, 3, 4:
        for j in 1, 2, 3, 4:
            idx = '('+str(i)+','+str(j)+')'
            df['A'+idx] = np.sqrt(df['SR'+idx] ** 2 + df['SI'+idx] ** 2)
            df['P'+idx] = np.arctan2(df['SI'+idx] , df['SR'+idx])
    return df

def main():
    current_dir = os.getcwd()
    # Get the absolute path of the upper directory
    data_dir = os.path.abspath(os.path.join(current_dir, '..', 'Data', 'df.csv'))
    df = data_parse()
    df.to_csv(data_dir)

if __name__ == "__main__":
    main()
