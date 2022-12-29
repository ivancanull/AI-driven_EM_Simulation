import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import data_parser





def plot_abs_by_index(dataset, idx, column):
    df = dataset.loc[idx]
    F = df['F']
    y_R = df['SR'+column]
    y_I = df['SI'+column]
    y_abs = np.log(np.sqrt(y_R ** 2 + y_I ** 2))
    
    fig, ax = plt.subplots(figsize=(20, 4))
    ax.plot(F, y_abs)
    ax.set_xlabel('F')
    ax.set_ylabel('S'+column+'_abs')
    
    fig.savefig('../Figures/'+idx+'S'+column+'_abs.pdf')
    fig.show()

def plot_all_abs(dataset, idx):
    for i in range(4):
        for j in range(4):
            column = ('('+str(i+1)+','+str(j+1)+')')
            plot_abs_by_index(dataset, idx, column)


def main():
    df = data_parser.data_parse()
    idx = '402200'
    plot_all_abs(df, idx)

if __name__ == "__main__":
    main()