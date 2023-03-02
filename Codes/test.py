import random
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import r2_score
from plot import *


def load_model(model, learning_rate, args):

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    checkpoint = torch.load(args['path'])
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    return epoch, model, optimizer, loss

def test(model, test_dataloader, learning_rate, num_epochs, args):

    _, model, _, _ = load_model(model, learning_rate, args)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    total_loss = 0.0
    total_r2 = 0.0
    total_batch_size = 0
    
    for i, data in enumerate(test_dataloader, 0):
        
        with torch.no_grad():
            
            # Fetch the data
            model.eval()
            X_test, y_test = data
            batch_size = X_test.shape[0]

            # Forward propagation
            predictions = model(X_test)

            # Calculate the loss function
            loss = criterion(predictions, y_test)
            total_loss += loss.item() * batch_size

            # Compute the R^2 score
            r2 = r2_score(predictions.detach().cpu().numpy(), y_test.detach().cpu().numpy())
            total_r2 += r2 * batch_size
            total_batch_size += batch_size

            if i == len(test_dataloader) - 1:

                test_loss = total_loss / total_batch_size
                test_r2 = total_r2 / total_batch_size

                idx = random.randint(0, batch_size - 1)
                live_plot_X_y(y_test.cpu()[idx,:], predictions.cpu()[idx,:], 0, test_loss, test_r2)

    return predictions


