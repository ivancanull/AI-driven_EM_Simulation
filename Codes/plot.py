import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

from IPython.display import clear_output

import data_parser


def live_plot_X_y(y, pred, epoch, loss, r2, args):
    # args:
    # cos_sin: True / False
    # output_col: A(i,j) or P(i,j)
    # F: frequency points
    
    clear_output(wait=True)
    X = args['F']
    
    if args['mode'] == 'cos_sin':
        
        fig, ax = plt.subplots(2, 1, figsize=(20, 8))
        ax[0].plot(X, y[:,0].detach().numpy(), label='sin truth')
        ax[1].plot(X, y[:,1].detach().numpy(), label='cos truth')
        ax[0].plot(X, pred[:,0].detach().numpy(), label='sin prediction')
        ax[1].plot(X, pred[:,1].detach().numpy(), label='cos prediction')
        
        text_kwargs = dict(fontsize=18, )
        ax[0].text(0, y[:,0].max(), '%s, Batch %d Loss: %.4f R2 : %.4f' % (args['output_col'], epoch, loss, r2), **text_kwargs)
        ax[0].legend(loc='upper right')
        ax[1].legend(loc='upper right')
        
    elif args['mode'] == 'single':
        
        fig, ax = plt.subplots(figsize=(20, 4))
        ax.plot(X, y.detach().numpy(), label='truth')
        ax.plot(X, pred.detach().numpy(), label='prediction')
        
        text_kwargs = dict(fontsize=18, )
        plt.text(0, y.max(), '%s, Batch %d Loss: %.4f R2 : %.4f' % (args['output_col'], epoch, loss, r2), **text_kwargs)
        ax.legend(loc='upper right')
    
    elif args['mode'] == 'total':
        fig, ax = plt.subplots(5, 1, figsize=(20, 20))
        ax[0].plot(X, y[:,0].detach().numpy(), label='A truth')
        ax[1].plot(X, y[:,1].detach().numpy(), label='sin truth')
        ax[2].plot(X, y[:,2].detach().numpy(), label='cos truth')
        ax[3].plot(X, y[:,3].detach().numpy(), label='SR truth')
        ax[4].plot(X, y[:,4].detach().numpy(), label='SI truth')

        ax[0].plot(X, pred[:,0].detach().numpy(), label='A prediction')
        ax[1].plot(X, pred[:,1].detach().numpy(), label='sin prediction')
        ax[2].plot(X, pred[:,2].detach().numpy(), label='cos prediction')
        ax[3].plot(X, pred[:,3].detach().numpy(), label='SR prediction')
        ax[4].plot(X, pred[:,4].detach().numpy(), label='SI prediction')

        
        text_kwargs = dict(fontsize=18, )
        ax[0].text(0, y[:,0].max(), '%s, Batch %d Loss: %.4f R2 : %.4f' % (args['output_col'], epoch, loss, r2), **text_kwargs)
        ax[0].legend(loc='upper right')
        ax[1].legend(loc='upper right')
        ax[2].legend(loc='upper right')
        ax[3].legend(loc='upper right')
        ax[4].legend(loc='upper right')

    plt.show();


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