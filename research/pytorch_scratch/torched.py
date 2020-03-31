import torch
import torch.optim as optim
import torch.nn as nn
from torchviz import make_dot
import numpy as np

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


###### PyTorch Implementation ######
# If CUDA-enabled gpu is available, use it to train
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
    print("GPU detected, utilizing GPU")

# Our data was in Numpy arrays, but we need to transform them into PyTorch's Tensors
# and then we send them to the chosen device
x_train_tensor = torch.from_numpy(x_train).float().to(device)
y_train_tensor = torch.from_numpy(y_train).float().to(device)


# Initializes parameters "a" and "b" randomly
# Create tensors, require gradients and move to specified device
np.random.seed(42)
a = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
b = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)

### Simple Model for Regression ###


class ManualLinearRegression(nn.Module):
    def __init__(self):
        super().__init__()

        # Establish parameters of model
        # Essentially tell model, these tensors are parameters of you
        self.a = nn.Parameter(torch.randn(
            1, requires_grad=True, dtype=torch.float))
        self.b = nn.Parameter(torch.randn(
            1, requires_grad=True, dtype=torch.float))

    def forward(self, x):
        # Output Prediction
        return self.a + self.b * x


# Create Model and send to where data is located
model = ManualLinearRegression().to(device)
print(model.state_dict())


# Set learning rate and number of epochs
lr = 1e-1
n_epochs = 1000

# Define optimizer to handle updating our parameters based on gradients
# SGD = Stochastic Gradient Descent optimizer
optimizer = optim.SGD(model.parameters(), lr=lr)

# Define loss function
# Using Mean Squares Error in this case because of linear regression
# Can also use reduction=sum to change how individual points are aggregated
loss_fn = nn.MSELoss(reduction='mean')

# Batch Gradient Descent (1 Epoch = All Train Data)
for epoch in range(n_epochs):
    # Computes our model's predicted output
    # yhat = a + b * x_train_tensor
    model.train()  # Sets model to training mode
    yhat = model(x_train_tensor)

    # How wrong is our model?
    # error = (y_train_tensor - yhat)
    # Loss is mean squared error (MSE)
    # loss = (error ** 2).mean()
    loss = loss_fn(y_train_tensor, yhat)

    # Pytorch handles these manual computations through backwards()
    # a_grad = -2 * error.mean()
    # b_grad = -2 * (x_train * error).mean()

    # Work backwards from calculated loss
    loss.backward()

    # Use NO_GRAD to keep the update out of the gradient computation
    # This is due to PyTorch dynamic graph (only care about parameters with gradients)
    # No grad allows for regular Python operations on tensors
    # without a care for the computation graph
    # with torch.no_grad():
    #    a -= lr * a.grad
    #    b -= lr * b.grad

    # Update using our optimizer instead
    optimizer.step()

    # Tell Pytorch to let go of computed gradients
    # We do NOT want to accumulate gradients
    # a.grad.zero_()
    # b.grad.zero_()
    optimizer.zero_grad()

print(model.state_dict())
