import argparse
import numpy as np

def entropy_cal(labels):
    values, cnt = np.unique(labels, return_counts=True)
    probab = cnt / cnt.sum()
    entropy = -1 * np.sum(probab * np.log2(probab))
    return entropy

def error_cal(labels):
    values, cnt = np.unique(labels, return_counts=True)
    majority_vote_cnt = cnt.max()
    error = (cnt.sum() - majority_vote_cnt) / cnt.sum()
    return error

def main(input_file, output_file):
    data = np.loadtxt(input_file, delimiter='\t', skiprows=1)
    labels = data[:, -1].astype(int)  

    entropy = entropy_cal(labels)
    error = error_cal(labels)

    with open(output_file, 'w') as file:
        file.write(f'entropy: {entropy:.6f}\n')
        file.write(f'error: {error:.6f}\n')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculating entropy and error rate of label.')
    parser.add_argument('input', type=str, help='The path of the input file.')
    parser.add_argument('output', type=str, help='The path of the output file.')

    args = parser.parse_args()

    main(args.input, args.output)


