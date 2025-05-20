import argparse
import numpy as np

class Node:
    def __init__(self, attr = None, thresh = None, left = None, right = None, vote = None, depth = 0, stats = None):
        self.attr = attr
        self.thresh = thresh
        self.left = left
        self.right = right
        self.vote = vote
        self.depth = depth
    
    def is_leaf(self):
        return self.vote is not None
    
    def add_left_child(self, node):
        self.left = node

    def add_right_child(self, node):
        self.right = node

def load_data(filepath):
    data = np.genfromtxt(filepath, delimiter='\t', names=True, dtype=None, encoding=None)
    feature_names = data.dtype.names[:-1]  
    features = np.array([list(row)[:-1] for row in data])
    labels = np.array([row[-1] for row in data]).astype(int)
    return features, labels, feature_names


def entropy_cal(labels):
    values, cnt = np.unique(labels, return_counts=True)
    probab = cnt / cnt.sum()
    entropy = -np.sum(probab * np.log2(probab))
    return entropy

def predict(node, example):
    if node.is_leaf():
        return node.vote
    else:
        attr = node.attr
        if example[attr] <= node.thresh:  # assuming binary split: 0 or 1
            return predict(node.left, example)
        else:
            return predict(node.right, example)
        
def write_metrics(train_error, test_error, metric_file):
    with open(metric_file, 'w') as file:
        file.write(f"error(train): {train_error}\n")
        file.write(f"error(test): {test_error}\n")

def write_predictions(predictions, predictions_file):
    with open(predictions_file, 'w') as file:
        for prediction in predictions:
            file.write(f"{prediction}\n")

def predict_majority(labels):
    values, counts = np.unique(labels, return_counts=True)
    label_counts = dict(zip(values, counts))

    if 0 not in label_counts:
        label_counts[0] = 0
    if 1 not in label_counts:
        label_counts[1] = 0

    majority_label = 1 
    if label_counts[0] > label_counts[1]:
        majority_label = 0

    return majority_label

def build_tree(features, labels, depth=0, max_depth=None, min_samples_split=2, min_samples_leaf=1):
    if max_depth is not None and max_depth == 0:
        return Node(vote=predict_majority(labels), depth=depth)

    if depth >= max_depth or len(labels) < min_samples_split:
        return Node(vote=predict_majority(labels), depth=depth)

    best_attr, best_thresh, best_info_gain = None, None, -np.inf
    base_entropy = entropy_cal(labels)

    for attr in range(features.shape[1]):
        thresholds = np.unique(features[:, attr])
        for thresh in thresholds:
            left_indices = features[:, attr] <= thresh
            right_indices = features[:, attr] > thresh
            labels_left, labels_right = labels[left_indices], labels[right_indices]

            if len(labels_left) >= min_samples_leaf and len(labels_right) >= min_samples_leaf:
                weight_left = len(labels_left) / len(labels)
                weight_right = len(labels_right) / len(labels)
                new_entropy = weight_left * entropy_cal(labels_left) + weight_right * entropy_cal(labels_right)
                info_gain = base_entropy - new_entropy

                if info_gain > best_info_gain:
                    best_attr, best_thresh, best_info_gain = attr, thresh, info_gain
                    best_labels_left, best_labels_right = labels_left, labels_right
                    best_features_left, best_features_right = features[left_indices], features[right_indices]

    if best_attr is None:
        return Node(vote=predict_majority(labels), depth=depth)

    left_subtree = build_tree(best_features_left, best_labels_left, depth + 1, max_depth, min_samples_split, min_samples_leaf)
    right_subtree = build_tree(best_features_right, best_labels_right, depth + 1, max_depth, min_samples_split, min_samples_leaf)

    return Node(attr=best_attr, thresh=best_thresh, left=left_subtree, right=right_subtree, depth=depth)

def calculate_error(predictions, labels):
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    incorrect = (predictions != labels).sum()
    
    error_rate = incorrect / len(labels)
    return error_rate

def print_tree(node, depth=0, file=None, feature_names=None, value_counts=None):
    # indent = ' ' * depth * 2

    # if depth == 0 and value_counts is None:
    #     value_counts = [sum(train_labels == 0), sum(train_labels == 1)]

    # if node.is_leaf():
    #     print(f"{indent}Leaf: [{value_counts[0]} {value_counts[1]}]", file=file)
    # else:
    #     feature_name = feature_names[node.attr]
    #     left_counts = [sum(node.left.stats == 0), sum(node.left.stats == 1)]
    #     right_counts = [sum(node.right.stats == 0), sum(node.right.stats == 1)]

    #     print(f"{indent}{feature_name} <= {node.thresh}: [{left_counts[0]} {left_counts[1]}]", file=file)
    #     print_tree(node.left, depth + 1, file=file, feature_names=feature_names, value_counts=left_counts)

    #     print(f"{indent}{feature_name} > {node.thresh}: [{right_counts[0]} {right_counts[1]}]", file=file)
    #     print_tree(node.right, depth + 1, file=file, feature_names=feature_names, value_counts=right_counts)
    pass



if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the train_input path, you can use `args.train_input`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to training input .tsv file')
    parser.add_argument("test_input", type=str, help='path to the test input .tsv file')
    parser.add_argument("max_depth", type=int, 
                        help='maximum depth to which the tree should be built')
    parser.add_argument("train_out", type=str, 
                        help='path to output .txt file to which the feature extractions on the training data should be written')
    parser.add_argument("test_out", type=str, 
                        help='path to output .txt file to which the feature extractions on the test data should be written')
    parser.add_argument("metrics_out", type=str, 
                        help='path of the output .txt file to which metrics such as train and test error should be written')
    parser.add_argument("print_out", type=str,
                        help='path of the output .txt file to which the printed tree should be written')
    args = parser.parse_args()

    train_features, train_labels, feature_names = load_data(args.train_input)
    test_features, test_labels, _ = load_data(args.test_input) 

    tree = build_tree(train_features, train_labels, max_depth=args.max_depth)

    train_predictions = [predict(tree, example) for example in train_features]
    test_predictions = [predict(tree, example) for example in test_features]

    train_error = calculate_error(train_predictions, train_labels)
    test_error = calculate_error(test_predictions, test_labels)

    write_predictions(train_predictions, args.train_out)
    write_predictions(test_predictions, args.test_out)

    write_metrics(train_error, test_error, args.metrics_out)
    
    #Here's an example of how to use argparse
    print_out = args.print_out

    #Here is a recommended way to print the tree to a file
    # with open(print_out, "w") as file:
    #     print_tree(dTree, file)

    with open(args.print_out, 'w') as f:
        print_tree(tree, depth=0, file=f, feature_names=feature_names)