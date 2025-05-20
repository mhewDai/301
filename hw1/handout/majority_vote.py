import numpy as np
import sys

def load_data(file_path):
    data = np.loadtxt(file_path, delimiter='\t', skiprows=1)
    labels = data[:, -1]
    return labels

def predict_majority(labels):
    values, counts = np.unique(labels, return_counts=True)
    max_count = np.max(counts) 
    min_count = np.min(counts)

    if max_count == min_count:
        return 1 
    majority_vals = values[counts == max_count]
    
    return majority_vals

def calculate_error(predictions, labels):
    errors = predictions != labels
    error_rate = errors.mean()
    return error_rate

def write_predictions(file_path, predictions):
    with open(file_path, 'w') as f_out:
        for pred in predictions:
            f_out.write(f"{int(pred)}\n")

def write_metrics(file_path, train_error, test_error):
    with open(file_path, 'w') as f_out:
        f_out.write(f"error(train): {train_error:.6f}\n")
        f_out.write(f"error(test): {test_error:.6f}\n")

def process_dataset(train_input, test_input, train_out, test_out, metrics_out):
    train_labels = load_data(train_input)
    test_labels = load_data(test_input)

    majority_val = predict_majority(train_labels)

    train_predictions = np.full(train_labels.shape, majority_val)
    test_predictions = np.full(test_labels.shape, majority_val)

    write_predictions(train_out, train_predictions)
    write_predictions(test_out, test_predictions)

    train_error = calculate_error(train_predictions, train_labels)
    test_error = calculate_error(test_predictions, test_labels)
    write_metrics(metrics_out, train_error, test_error)

if __name__ == '__main__':
    train = sys.argv[1]
    test = sys.argv[2]
    train_labels = sys.argv[3]
    test_labels = sys.argv[4]
    metrics = sys.argv[5]

    process_dataset(train, test, train_labels, test_labels, metrics)