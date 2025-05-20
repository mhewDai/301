import numpy as np
import sys

def load_data(file_path):
    # Load the data from a .tsv file and return the features and labels
    data = np.loadtxt(file_path, delimiter='\t', skiprows=1)
    features, labels = data[:, :-1], data[:, -1]
    return features, labels

def predict_majority(labels):
    # Predict the majority class
    (values, counts) = np.unique(labels, return_counts=True)
    majority_val = values[np.argmax(counts)]
    return majority_val

def calculate_error(predictions, labels):
    # Calculate the error rate of predictions
    errors = predictions != labels
    error_rate = errors.mean()
    return error_rate

def write_predictions(file_path, predictions):
    with open(file_path, 'w') as f_out:
        for pred in predictions:
            f_out.write(f"{int(pred)}\n")

def write_metrics(file_path, train_error, test_error):
    # Write the training and testing error to a file
    with open(file_path, 'w') as f_out:
        f_out.write(f"error(train): {train_error:.6f}\n")
        f_out.write(f"error(test): {test_error:.6f}\n")

def process_dataset(train_input, test_input, train_out, test_out, metrics_out):
    # Load training and test data
    _, train_labels = load_data(train_input)
    _, test_labels = load_data(test_input)

    # Determine the majority class from the training data
    majority_val = predict_majority(train_labels)

    # Predict majority class for training and test sets
    train_predictions = np.full_like(train_labels, majority_val)
    test_predictions = np.full_like(test_labels, majority_val)

    # Write the predictions to output files
    write_predictions(train_out, train_predictions)
    write_predictions(test_out, test_predictions)

    # Calculate and write the error metrics
    train_error = calculate_error(train_predictions, train_labels)
    test_error = calculate_error(test_predictions, test_labels)
    write_metrics(metrics_out, train_error, test_error)

heart_train_input = '/Users/eir/Downloads/301/hw1/handout/heart_train.tsv'
heart_test_input = '/Users/eir/Downloads/301/hw1/handout/heart_test.tsv'

education_train_input = '/Users/eir/Downloads/301/hw1/handout/education_train.tsv'
education_test_input = '/Users/eir/Downloads/301/hw1/handout/education_test.tsv'

heart_train_out = '/Users/eir/Downloads/301/hw1/handout/solutions/heart_train_labels.txt'
heart_test_out = '/Users/eir/Downloads/301/hw1/handout/solutions/heart_test_labels.txt'
heart_metrics_out = '/Users/eir/Downloads/301/hw1/handout/solutions/heart_metrics.txt'

education_train_out = '/Users/eir/Downloads/301/hw1/handout/solutions/education_train_labels.txt'
education_test_out = '/Users/eir/Downloads/301/hw1/handout/solutions/education_test_labels.txt'
education_metrics_out = '/Users/eir/Downloads/301/hw1/handout/solutions/education_metrics.txt'

print("Processing the heart dataset...")
process_dataset(heart_train_input, heart_test_input, heart_train_out, heart_test_out, heart_metrics_out)
print("Heart dataset processed.")

print("Processing the education dataset...")
process_dataset(education_train_input, education_test_input, education_train_out, education_test_out, education_metrics_out)
print("Education dataset processed.")

