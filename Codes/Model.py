import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from IPython.display import clear_output
from sklearn.metrics import r2_score

import data_parser

def live_plot_X_y(X, y, pred=None, epoch=None, loss=None, r2=None, mer=None):
    clear_output(wait=True)
    fig, ax = plt.subplots(figsize=(20, 4))
    range(len(X))
    ax.plot(X.detach().numpy(), y.detach().numpy())
    if pred != None:
        text_kwargs = dict(fontsize=18, )
        ax.plot(X.detach().numpy(), pred.detach().numpy())
        plt.text(0, y.max(), 'Batch %d Loss: %.4f R2 : %.4f Max Error Rate : %.4f' % (epoch, loss, r2, mer), **text_kwargs)

    plt.show();
    
def plot_X_y(X, y, pred=None, epoch=None, loss=None, r2=None, mer=None):
    clear_output(wait=True)
    fig, ax = plt.subplots(figsize=(20, 4))
    range(len(X))
    ax.plot(X.detach().numpy(), y.detach().numpy())
    if pred != None:
        text_kwargs = dict(fontsize=18, )
        ax.plot(X.detach().numpy(), pred.detach().numpy())
        plt.text(0, y.max(), 'Final Results Loss: %.4f R2 : %.4f Max Error Rate : %.4f' % (loss, r2, mer), **text_kwargs)

    plt.show();


def initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias.data)

class data_processor():
    def __init__(self):
        self.dict = {}
    
    def initial_normalize(self, X, y):
        self.dict['X_std'] = X.std(dim=0)
        self.dict['X_mean'] = X.mean(dim=0)
        self.dict['y_std'] = y.std(dim=0)
        self.dict['y_mean'] = y.mean(dim=0)
        # new_X = (X - self.dict['X_mean']) / self.dict['X_std']
        # new_y = (y - self.dict['y_mean']) / self.dict['y_std']
        # return new_X, new_y
        return
    
    def normalize(self, X=None, y=None):
        new_X, new_y = None, None
        if X is not None:
            new_X = torch.nan_to_num((X - self.dict['X_mean']) / self.dict['X_std'])
        if y is not None:
            new_y = torch.nan_to_num((y - self.dict['y_mean']) / self.dict['y_std'])
        return new_X, new_y
    
    def denormalize(self, X=None, y=None):
        original_X, original_y = None, None
        if X is not None:
            original_X = X * self.dict['X_std'] + self.dict['X_mean']
        if y is not None:
            original_y = y + self.dict['y_mean']
            original_y = y * self.dict['y_std'] + self.dict['y_mean']
        return original_X, original_y
    
class Model():
    
    def __init__(self, df, network, indices, input_cols, output_col, in_features, out_features, device, postfix=''):
        
        train_idx = indices['train_idx']
        val_idx = indices['val_idx']
        test_idx = indices['test_idx']

        # Transform the training data to device
        X_train, y_train = torch.Tensor(df.loc[train_idx][input_cols].to_numpy()).to(device), torch.Tensor(df.loc[train_idx][output_col].to_numpy().reshape(-1, 1)).to(device)
        X_val, y_val = torch.Tensor(df.loc[val_idx][input_cols].to_numpy()).to(device), torch.Tensor(df.loc[val_idx][output_col].to_numpy().reshape(-1, 1)).to(device)
        X_test, y_test = torch.Tensor(df.loc[test_idx][input_cols].to_numpy()).to(device), torch.Tensor(df.loc[test_idx][output_col].to_numpy().reshape(-1, 1)).to(device)

        # Reduce the second dimension by torch.mean
        X_train, y_train = torch.mean(X_train.reshape(-1, out_features, in_features), dim=1), y_train.reshape(-1, out_features)
        X_val, y_val = torch.mean(X_val.reshape(-1, out_features, in_features), dim=1), y_val.reshape(-1, out_features)
        X_test, y_test = torch.mean(X_test.reshape(-1, out_features, in_features), dim=1), y_test.reshape(-1, out_features)

        # Generate frequency label for plotting
        train_idx_example = train_idx[0]
        self.F = torch.Tensor(df.loc[train_idx_example]['F']).reshape(-1, 1)
    
        # Preprocess the data
        dp = data_processor()
        dp.initial_normalize(torch.cat((X_train, X_val)), torch.cat((y_train, y_val)))
        X_train_norm, y_train_norm = dp.normalize(X_train, y_train)
        X_val_norm, y_val_norm = dp.normalize(X_val, y_val)
        X_test_norm, _ = dp.normalize(X=X_test)                
        
        # Store the data
        self.train_idx = train_idx
        self.test_idx = test_idx
        self.val_idx = val_idx
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.X_test, self.y_test = X_test, y_test
        self.X_train_norm, self.y_train_norm = X_train_norm, y_train_norm
        self.X_val_norm, self.y_val_norm = X_val_norm, y_val_norm
        self.X_test_norm = X_test_norm
        self.dp = dp
        self.device = device # where the data are kept
        
        # Define the model
        self.network = network
        self.model = network.to(device)
        initialize_weights(self.model)
        
        # Define saved filepath
        if postfix != '':
            # checking if the directory demo_folder 
            # exist or not.
            model_dir = os.path.join("../Models/", postfix)
            if not os.path.isdir(model_dir):
                os.makedirs(model_dir)
            self.path = os.path.join(model_dir, output_col + '_best_model.pt')
        else:
            self.path = os.path.join("../Models/", output_col + '_best_model.pt')
            
    def get_dp(self):
        return self.dp
        
    def save_model(self, epochs, model, optimizer, loss):
    
        # Function to save the trained model to disk.

        torch.save({
                    'epoch': epochs,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    'dp': self.dp,
                    }, self.path)
    
    def load_model(self, learning_rate=0.01):
        
        model = self.network.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        checkpoint = torch.load(self.path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        dp = checkpoint['dp']
        
        return epoch, model, optimizer, loss, dp
        
    def train(self, learning_rate, num_epochs):
        
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        best_loss = 100.0
        
        
        for epoch in range(num_epochs):
            
            self.model.train()
            
            # Clear the gradients
            optimizer.zero_grad()

            # Forward propagation
            outputs = self.model(self.X_train_norm)
 
            # Calculate the loss function
            loss = criterion(outputs, self.y_train_norm)

            # Backward propagation
            loss.backward()

            # Update parameters
            optimizer.step()

            # Validation
            if epoch % 100 == 99: 
                
                with torch.no_grad():
                    
                    self.model.eval()

                    predictions = self.model(self.X_val_norm)

                    _, predictions = self.dp.denormalize(y=predictions)
                    _, outputs = self.dp.denormalize(y=outputs)
                    
                    test_loss = criterion(predictions, self.y_val)
                    
                    if test_loss < best_loss:
                        best_loss = test_loss
                        self.save_model(epoch, self.model, optimizer, test_loss)

                    # Compute the R^2 score
                    r2 = r2_score(np.nan_to_num(predictions.detach().cpu().numpy()), self.y_val.detach().cpu().numpy())

                    # Compute the max error rate
                    mer = torch.nan_to_num(torch.max(torch.abs(predictions - self.y_val) / self.y_val))


                    # Plot validating results
                    i = random.randint(0, self.y_val.shape[0]-1)
                    live_plot_X_y(self.F.cpu(), self.y_val.cpu()[i,:], predictions.cpu()[i,:], epoch + 1, test_loss, r2, mer)
        
        
        
        return
    
    def test(self):
        
        _, model, _, _, _ = self.load_model()
        criterion = nn.MSELoss()
        
        with torch.no_grad():
            
            model.eval()

            # Make predictions on the test data
            predictions = model(self.X_test_norm)
            _, predictions = self.dp.denormalize(y=predictions)

            # Calculate the loss function
            test_loss = criterion(predictions, self.y_test)

            # Track the loss value 
            test_loss = test_loss.item()  

            # Compute the R^2 score
            r2 = r2_score(predictions.detach().cpu().numpy(), self.y_test.detach().cpu().numpy())

            # Compute the max error rate
            mer = torch.nan_to_num(torch.max(torch.abs(predictions - self.y_test) / self.y_test))

            plot_X_y(self.F.cpu(), self.y_test.cpu()[0,:], predictions.cpu()[0,:], 0, test_loss, r2, mer)

        return predictions
    
    def predict(self, X, y=None, plot=False):

        _, model, _, _, dp = self.load_model()
        criterion = nn.MSELoss()
        X_norm, _ = dp.normalize(X=X)

        with torch.no_grad():
            
            model.eval()

            # Make predictions on the test data
            predictions = model(X_norm)
            _, predictions = self.dp.denormalize(y=predictions)

            if y is not None:
                # Calculate the loss function
                loss = criterion(predictions, y)

                # Track the loss value 
                loss = loss.item()  

                # Compute the R^2 score
                r2 = r2_score(predictions.detach().cpu().numpy(), y.detach().cpu().numpy())

                # Compute the max error rate
                mer = torch.nan_to_num(torch.max(torch.abs(predictions - y) / y))

                if plot == True:
                    plot_X_y(self.F.cpu(), y.cpu()[0,:], predictions.cpu()[0,:], 0, loss, r2, mer)
        
                return predictions, loss, r2, mer
            else:
                return predictions



