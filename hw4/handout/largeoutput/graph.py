import csv
import numpy as np
import matplotlib.pyplot as plt

def load_csv_data(file_path):
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file, delimiter='\t')
        data = np.array(list(csv_reader)).astype(float)
    return data

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_negative_log_likelihood(features, labels, weights):
    predictions = sigmoid(features @ weights)
    log_likelihood = -labels * np.log(predictions) - (1 - labels) * np.log(1 - predictions)
    return np.mean(log_likelihood)

def train_logistic_regression(X_train, y_train, X_val, y_val, epochs, learning_rate):
    weights = np.zeros(X_train.shape[1])
    train_nll = []
    val_nll = []

    for epoch in range(epochs):
        # Training step
        predictions_train = sigmoid(X_train @ weights)
        error_train = predictions_train - y_train
        gradient = X_train.T @ error_train / len(y_train)
        weights -= learning_rate * gradient
        
        # Compute NLL for training and validation datasets
        nll_train = compute_negative_log_likelihood(X_train, y_train, weights)
        nll_val = compute_negative_log_likelihood(X_val, y_val, weights)
        
        train_nll.append(nll_train)
        val_nll.append(nll_val)

    return weights, train_nll, val_nll

def plot_negative_log_likelihood(train_nll, val_nll):
    plt.plot(train_nll, label='Training')
    plt.plot(val_nll, label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Average Negative Log-Likelihood')
    plt.title('Negative Log Likelihood vs. Epoch')
    plt.legend()
    plt.show()

# Parameters
# Parameters
epochs = 1000  # Change this line
learning_rate = 0.1

# Load the data
train_data = load_csv_data('formatted_train_large.tsv')
val_data = load_csv_data('formatted_val_large.tsv')

# Separate the features and labels
X_train, y_train = train_data[:, :-1], train_data[:, -1]
X_val, y_val = val_data[:, :-1], val_data[:, -1]

# Add bias term
X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
X_val = np.hstack((np.ones((X_val.shape[0], 1)), X_val))

# Train the model and calculate negative log likelihoo
weights, train_nll, val_nll = train_logistic_regression(X_train, y_train, X_val, y_val, epochs, learning_rate)


# Plot
plot_negative_log_likelihood(train_nll, val_nll)

learning_rates = [1e-1, 1e-2, 1e-3]

# Initialize dictionary to hold the negative log likelihoods for each learning rate
nlls = {}

# Loop over each learning rate, train the model, and record the negative log likelihoods
for lr in learning_rates:
    weights, train_nll, _ = train_logistic_regression(X_train, y_train, X_val, y_val, epochs, lr)
    nlls[lr] = train_nll

# Plot the negative log likelihood over epochs for each learning rate
plt.figure(figsize=(10, 6))
for lr, nll in nlls.items():
    plt.plot(nll, label=f'Learning Rate: {lr}')

plt.xlabel('Epochs')
plt.ylabel('Average Negative Log-Likelihood')
plt.title('Training Negative Log Likelihood vs. Epoch for Different Learning Rates')
plt.legend()
plt.show()
