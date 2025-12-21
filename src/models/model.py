import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DigitsClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(DigitsClassifier, self).__init__()

        #Layer 1: In(1, 28, 28) -> Out(32, 28, 28)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)

        #Layer 2: In(32, 28, 28) -> Out(64, 14, 14)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        #Layer 3: In
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        #Reduce image size by half (28 -> 14) (14 -> 7)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        

        self.flatten_size = 64 * 7 * 7
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
       
        x = self.pool(F.relu(self.conv1(x)))

        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(-1, self.flatten_size) #convert to vector

        x = F.relu(self.fc1(x))

        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)

        return x
    

class MLP(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.model = nn.Sequential(
            nn.Flatten(),                   # biến ảnh 2D → vector 1D (784)

            nn.Linear(28*28, 256),          # Layer 1
            nn.ReLU(),

            nn.Linear(256, 128),            # Layer 2
            nn.ReLU(),

            nn.Dropout(0.2),                # chống overfitting 

            nn.Linear(128, num_classes)     # Output layer
        )

    def forward(self, x):
        return self.model(x)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='model.pth', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'model.pth'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decreases.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
