import torch
import torch.optim as optim
import torch.nn as nn
from torchviz import make_dot
import numpy as np

###### Data Generation ######
np.random.seed(42)
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


###### PyTorch Implementation ######
# If CUDA-enabled gpu is available, use it to train
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
    print("GPU detected, utilizing GPU")

# Initializes parameters "a" and "b" randomly
# Create tensors, require gradients and move to specified device
np.random.seed(42)
a = np.random.randn(1, requires_grad=True, dtype=torch.float, device=device)
b = np.random.randn(1, requires_grad=True, dtype=torch.float, device=device)


print("a and b after random initialization:")
print(a, b)

# Set learning rate and number of epochs
lr = 1e-1
n_epochs = 1000

# Batch Gradient Descent (1 Epoch = All Train Data)
for epoch in range(n_epochs):
    # Computes our model's predicted output
    yhat = a + b * x_train
    # How wrong is our model?
    error = (y_train - yhat)
    # Loss is mean squared error (MSE)
    loss = (error ** 2).mean()

    # Pytorch handles these manual computations through backwards()
    # a_grad = -2 * error.mean()
    # b_grad = -2 * (x_train * error).mean()

    # Work backwards from calculated loss
    loss.backward()

    print("Computed Gradients")
    print(a.grad)
    print(b.grad)

    # Use NO_GRAD to keep the update out of the gradient computation
    # This is due to PyTorch dynamic graph
    # No grad allows for regular Python operations on tensors
    # without a care for the computation graph
    with torch.no_grad():
        a -= lr * a.grad
        b -= lr * b.grad

    # Tell Pytorch to let go of computed gradients
    # We do NOT want to accumulate gradients
    a.grad.zero_()
    b.grad.zero_()

print("a and b after gradient descent:")
print(a, b)
