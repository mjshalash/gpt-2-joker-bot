import torch
import torch.optim as optim
import torch.nn as nn
from torchviz import make_dot
import numpy as np

from torch.utils.data import Dataset, TensorDataset

###### Data Generation ######
np.random.seed(42)              # Important to seed so we can reproduce results
x = np.random.rand(100, 1)
y = 1 + 2 * x + .1 * np.random.randn(100, 1)

# Shuffles the indices
idx = np.arange(100)
np.random.shuffle(idx)

# Uses first 80 random indices for train
train_idx = idx[:80]
# Uses the remaining indices for validation
val_idx = idx[80:]

# Generates train and validation sets
x_train, y_train = x[train_idx], y[train_idx]
x_val, y_val = x[val_idx], y[val_idx]


# Custom dataset class


class CustomDataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor

    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return len(self.x)


x_train_tensor = torch.from_numpy(x_train).float()
y_train_tensor = torch.from_numpy(y_train).float()

# *Note for either of these we do not send to GPU, we do not want all training data put onto GPU RAM
# Create instance of custom dataset class with train_data
train_data = CustomDataset(x_train_tensor, y_train_tensor)
print(train_data[0])

# Use PyTorch's TensorDataset class since dataset is a couple of tensors
train_data = TensorDataset(x_train_tensor, y_train_tensor)
print(train_data[0])

