import torch
import matplotlib.pyplot as plt

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
    wrong_pred_labels = test_labels + 0.5 

    # visualize
    plot_predictions(train_data, train_labels, test_data, test_labels, wrong_pred_labels)
