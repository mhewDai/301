import numpy as np
import argparse


def sigmoid(x : np.ndarray):
    """
    Implementation of the sigmoid function.

    Parameters:
        x (np.ndarray): Input np.ndarray.

    Returns:
        An np.ndarray after applying the sigmoid function element-wise to the
        input.
    """
    e = np.exp(x)
    return e / (1 + e)


def train(
    theta : np.ndarray, # shape (D,) where D is feature dim
    X : np.ndarray,     # shape (N, D) where N is num of examples
    y : np.ndarray,     # shape (N,)
    num_epoch : int, 
    learning_rate : float
) -> None:
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    for e in range(num_epoch):
        for i in range(X.shape[0]):
            xi = X[i]
            yi = y[i]
            prediction = sigmoid(xi.dot(theta))
            error = yi - prediction
            gradient = xi * error
            theta += learning_rate * gradient
    return theta

def predict(
    theta : np.ndarray,
    X : np.ndarray
) -> np.ndarray:
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    probs = sigmoid(X.dot(theta))
    return (probs >= 0.5).astype(int)

def compute_error(
    y_pred : np.ndarray, 
    y : np.ndarray
) -> float:
    return np.mean(y_pred != y)

def load_data(file):
    data = np.loadtxt(file, delimiter='\t', comments=None, dtype=float)
    return data[:, 1:], data[:, 0]

if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the learning rate, you can use `args.learning_rate`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to formatted training data')
    parser.add_argument("validation_input", type=str, help='path to formatted validation data')
    parser.add_argument("test_input", type=str, help='path to formatted test data')
    parser.add_argument("train_out", type=str, help='file to write train predictions to')
    parser.add_argument("test_out", type=str, help='file to write test predictions to')
    parser.add_argument("metrics_out", type=str, help='file to write metrics to')
    parser.add_argument("num_epoch", type=int, 
                        help='number of epochs of stochastic gradient descent to run')
    parser.add_argument("learning_rate", type=float,
                        help='learning rate for stochastic gradient descent')
    args = parser.parse_args()

    X_train, y_train = load_data(args.train_input)
    
    theta = np.zeros(X_train.shape[1] + 1)
    
    theta = train(theta, X_train, y_train, args.num_epoch, args.learning_rate)

    y_train_pred = predict(theta, X_train)
    train_error = compute_error(y_train_pred, y_train)

    np.savetxt(args.train_out, y_train_pred, fmt='%d')
    with open(args.metrics_out, 'w') as metrics_file:
        metrics_file.write(f'error(train): {train_error:.6f}\n')

    X_test, y_test = load_data(args.test_input)

    y_test_pred = predict(theta, X_test)
    test_error = compute_error(y_test_pred, y_test)

    np.savetxt(args.test_out, y_test_pred, fmt='%d')
    with open(args.metrics_out, 'a') as metrics_file:  # Append to the metrics file
        metrics_file.write(f'error(test): {test_error:.6f}\n')
