import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from typing import Iterator

class MSE():
    """The Mean Squared Error loss"""
    def __call__(self, y: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        y = torch.Tensor(y)
        target = torch.Tensor(target)
        mse = ((y - target) ** 2).sum().sqrt().mean()
        return mse


class GDLinearRegression(nn.Module):
    """A simple Linear Regression model"""

    def __init__(self):
        super().__init__()
        # We're initializing our model with random weights
        self.w = nn.Parameter(torch.randn(6, requires_grad=True))
        self.b = nn.Parameter(torch.randn(1, requires_grad=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.Tensor(x)
        y = x @ self.w + self.b
        return y

    # PyTorch is accumulating gradients
    # After each Gradient Descent step we should reset the gradients

    def zero_grad(self):
        self.w.grad.zero_()
        self.b.grad.zero_()

class GD:
  """
  Gradient Descent optimizer
  """
  def __init__(self, params: Iterator[nn.Parameter], lr: int):
    self.w, self.b = list(params)
    self.lr = lr

  def step(self):
    """
    Perform a gradient decent step. Update the parameters w and b by using:
     - the gradient of the loss with respect to the parameters
     - the learning rate
    This method is called after backward(), so the gradient of the loss wrt
    the parameters is already computed.
    """
    with torch.no_grad():
      self.w -= self.w.grad * self.lr
      self.b -= self.b.grad * self.lr


def train(model: GDLinearRegression, data: torch.Tensor,
          target: torch.Tensor, optim: GD, criterion: MSE):
    """Linear Regression train routine"""
    # forward pass: compute predictions (hint: use model.forward)
    predictions = model(data)

    # forward pass: compute loss (hint: use criterion)
    loss = criterion(predictions, target)

    # backpropagation: compute gradients of loss wrt weights
    loss.backward()

    # GD step: update weights using the gradients (hint: use optim)
    optim.step()

    # reset the gradients (hint: use model)
    model.zero_grad()

    return model

def find_lr(x_train,y_train, start, end, num,mse):
    min_loss = 10 ** 1000
    best_lr = 0
    total_steps = 100
    for i in np.linspace(start,end,num):
        lr = i
        total_steps = 100

        model = GDLinearRegression()
        optimizer = GD(model.parameters(), lr=lr)
        criterion = MSE()

        for j in range(total_steps):
            train(model, x_train, y_train, optimizer, criterion)

        with torch.no_grad():
            y_pred = model(x_train)
        if min_loss > mse(y_pred, y_train).item():
            min_loss = mse(y_pred, y_train).item()
            best_lr = i

    return best_lr


data = pd.read_csv('Bucharest_HousePriceDataset.csv', sep=',')

predict = 'Nr Camere'

x = data.drop(columns=predict).values

y = data[[predict]].values.ravel()
y = y-1
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

scaler = StandardScaler()
scaler.fit(x)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

mse = MSE()

lr = 0.0015 #@param {type: "slider", min: 0.001, max: 2, step: 0.005}
total_steps = 100  #@param {type:"slider", min: 0, max: 100, step: 1}

model = GDLinearRegression()
optimizer = GD(model.parameters(), lr=lr)
criterion = MSE()

for i in range(total_steps):
    train(model, x_train, y_train, optimizer, criterion)

with torch.no_grad():
    y_pred = model(x_train)

print(accuracy_score(y_train, torch.round(y_pred).numpy()))

with torch.no_grad():
    y_pred = model(x_test)

print(accuracy_score(y_test, torch.round(y_pred).numpy()))

predicted = torch.round(y_pred).numpy()
regression_matrix = confusion_matrix(predicted,y_test)
regression_mse = mse(predicted,y_test)

print(regression_matrix)
print('MSE:', regression_mse)

#Clasificare

#using the nn from lab2 ex 3

class ThreeClassNN(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 hidden_activation_fn=nn.ReLU()):
        # Initialise the base class (nn.Module)
        super().__init__()

        # As stated above, we'll simply use `torch.nn.Linear` to define our 2 layers
        self._layer1 = nn.Linear(input_size, hidden_size)
        self._layer2 = nn.Linear(hidden_size, output_size)

        self._hidden_activation = hidden_activation_fn

    def forward(self, x):

        # Layer 1 using ReLU as activation
        h = self._hidden_activation(self._layer1(x))
        out = self._layer2(h)

        return out

predict = 'Nr Camere'

x = data.drop(columns=predict).values

y = data[[predict]].values
y = y-1
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

scaler = StandardScaler()
scaler.fit(x)
x_train = torch.tensor(scaler.transform(x_train)).float()
x_test = torch.tensor(scaler.transform(x_test)).float()
"""
model = ThreeClassNN(6, 1000, 9)

NUM_EPOCHS = 500



optim = torch.optim.Adam(model.parameters(), lr=1.5)
criterion = nn.CrossEntropyLoss()

for i in range(NUM_EPOCHS):
    # Set the model to train mode and reset the gradients
    model.train()
    optim.zero_grad()

    output = model(x_train)
    target = torch.tensor(y_train).long().squeeze(1)
    loss = criterion(output, target)

    loss.backward()
    optim.step()



predicted = np.array(torch.argmax(model(x_train), dim=-1))
accuracy = (predicted==y_train.ravel()).sum()/predicted.shape[0]
print(accuracy)

predicted = np.array(torch.argmax(model(x_test), dim=-1))
accuracy = (predicted==y_test.ravel()).sum()/predicted.shape[0]
print(accuracy)


class_matrix = confusion_matrix(predicted,y_test)
class_mse = mse(predicted,y_test)

print(class_matrix)
print('MSE:', class_mse)

"""

#loss when the network is not trained
#random variables uniform distribuited
#random  means that any number can occur with the same probability so the prob to be a certain number
#of rooms is 1/max number (max number = 9)
p = 1.0/9.0
Q = [p,p,p,p,p,p,p,p,p]
print(Q)
#calculate the probability for each nr of rooms from our dataset nr_favourite_cases/nr of posible cases

nr_y = [0,0,0,0,0,0,0,0,0]
for i in range(len(y)):
    nr_y[y[i][0]] += 1

P = [nr_y[i]/len(y) for i in range(len(nr_y))]

#Cross-entropy can be calculated using the probabilities of the events from P and Q, as follows:

#H(P, Q) = – sum x in X P(x) * log(Q(x))
H = 0
for i in range(len(Q)):
    H += P[i] * np.log(Q[i]+1e-9)

H = -H
print(H)
