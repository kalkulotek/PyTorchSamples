import torch
from torch import nn
import matplotlib.pyplot as plt

class LinearRegressionModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(
            1, requires_grad = True, dtype = torch.float
        ))
        self.bias = nn.Parameter(torch.rand(
            1, requires_grad = True, dtype = torch.float
        ))
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias

def plot_predictions(train_data, train_labels, 
                     test_data, test_labels,
                     pred_labels = None):
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_labels, c = "b", s = 4, label = "Training data")
    plt.scatter(test_data, test_labels, c = "g", s = 4, label = "Test data")
    if pred_labels != None:
        plt.scatter(test_data, pred_labels, c = "r", s = 4, label = "Predictions")
    plt.legend() 
    plt.show()

def get_linear_regression_data(weight, bias, start, end, step):
    x = torch.arange(start, end, step).unsqueeze(dim = 1)
    y = weight * x + bias
    return x, y

if __name__ == "__main__": 
    
    # create data using linear regression equation 
    # y = weight * x + bias
    weight = 0.7
    bias = 0.3
    start = 0
    end = 1
    step = 0.02
    x, y = get_linear_regression_data(weight, bias, start, end, step) 
    
    # split data into training and test sets (80% training data, 20% test data)
    train_split = int(0.8 * len(x))
    train_data, train_labels = x[:train_split], y[:train_split]
    test_data, test_labels = x[train_split:], y[train_split:]

    # make prediction using your model 
    model_0 = LinearRegressionModel()

    # training loop
    epochs = 100
    loss_function = nn.L1Loss()
    optimizer = torch.optim.SGD(
        params = model_0.parameters(), lr = 0.01 
    )
    for epoch in range(epochs):
        model_0.train()
        train_pred_labels = model_0(train_data)
        loss = loss_function(train_pred_labels, train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # predict using the trained model
    model_0.eval()
    with torch.inference_mode():
        test_pred_labels = model_0(test_data)
        test_loss = loss_function(test_pred_labels, test_labels)
        # visualize
        plot_predictions(train_data, train_labels, test_data, test_labels, test_pred_labels)
