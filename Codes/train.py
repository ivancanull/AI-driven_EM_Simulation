import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import r2_score
from plot import *

def initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias.data)
                
def save_model(epochs, model, optimizer, loss, args):

    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, args['path'])

def cos_sin_loss(output, target, beta):
    criterion = nn.MSELoss()
    losssinP = beta['sinP'] * criterion(output[..., 0], target[..., 0])
    losscosP = beta['cosP'] * criterion(output[..., 1], target[..., 1])

    lossP = beta['P'] * torch.mean(torch.abs(output[..., 0] ** 2 + output[..., 1] ** 2 - 1))
    loss = losssinP + losscosP + lossP
    return loss

def combined_loss(output, target, beta):
    criterion = nn.MSELoss()

    lossA = beta['A'] * criterion(output[..., 0], target[..., 0])
    losssinP = beta['sinP'] * criterion(output[..., 1], target[..., 1])
    losscosP = beta['cosP'] * criterion(output[..., 2], target[..., 2])
    lossSR = beta['SR'] * criterion(output[..., 3], target[..., 3])
    lossSI = beta['SI'] * criterion(output[..., 4], target[..., 4])
    lossSR_diff = beta['SR_diff'] * torch.mean(((output[..., 0] * output[..., 2]) - output[..., 3]) ** 2)
    lossSI_diff = beta['SI_diff'] * torch.mean(((output[..., 0] * output[..., 1]) - output[..., 3]) ** 2)
    lossP = beta['P'] * torch.mean(torch.abs(output[..., 1] ** 2 + output[..., 2] ** 2 - 1))
    loss = lossA + losssinP + losscosP + lossSI + lossSR + lossSR_diff + lossSI_diff + lossP
    return loss

def mse_loss(output, target, beta):
    criterion = nn.MSELoss()

    target_range = torch.max(target, dim=0)[0] - torch.min(target, dim=0)[0]
    return criterion(output / target_range, target / target_range) 

def train(model, train_dataloader, val_dataloader, learning_rate, loss_func, num_epochs, args):
    # args:
    # cos_sin: True / False
    # output_col: A(i,j) or P(i,j)
    # F: frequency points
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = loss_func
    best_loss = 100.0

    for epoch in range(num_epochs):
        
        total_loss = 0.0
        total_batch_size = 0
        
        for i, data in enumerate(train_dataloader, 0):
            
            # Fetch the data
            model.train()
            X_train, y_train = data
            batch_size = X_train.shape[0]
            
            # Clear the gradients
            optimizer.zero_grad()

            # Forward propagation
            outputs = model(X_train)
            
            # Calculate the loss function
            # y_train = (y_train - args['mean']) / args['std']
            loss = criterion(outputs, y_train, args['beta'])
            total_loss += loss.item() * batch_size
            total_batch_size += batch_size

            # Backward propagation
            loss.backward()

            # Update parameters
            optimizer.step()
        
        train_loss = total_loss / total_batch_size
        
        total_loss = 0.0
        total_r2 = 0.0

        total_batch_size = 0
        
        if epoch % 100 == 99:
        
            for i, data in enumerate(val_dataloader, 0):

                with torch.no_grad():

                    # Fetch the data
                    model.eval()
                    X_val, y_val = data
                    batch_size = X_val.shape[0]

                    # Forward propagation
                    predictions = model(X_val)

                    # Calculate the loss function
                    # predictions = predictions * args['std'] + args['mean']
                    loss = criterion(predictions, y_val, args['beta'])
                    total_loss += loss.item() * batch_size
                    # Compute the R^2 score
                    if args['mode'] == 'total':
                        r2 = r2_score(predictions[..., -2:].detach().cpu().numpy().reshape(-1), y_val[..., -2:].detach().cpu().numpy().reshape(-1))
                    else:
                        r2 = r2_score(predictions.detach().cpu().numpy().reshape(-1), y_val.detach().cpu().numpy().reshape(-1))
                    total_r2 += r2 * batch_size
                    total_batch_size += batch_size

                    if i == len(val_dataloader) - 1 :

                        test_loss = total_loss / total_batch_size
                        test_r2 = total_r2 / total_batch_size

                        idx = 0
                        live_plot_X_y(y_val.cpu()[idx,...], predictions.cpu()[idx,...], epoch + 1, test_loss, test_r2, args)

            test_loss = total_loss / total_batch_size
            if test_loss < best_loss:
                best_loss = test_loss
                save_model(epoch + 1, model, optimizer, test_loss, args)
        
    return model