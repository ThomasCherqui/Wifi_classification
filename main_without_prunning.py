import numpy as np
import json
from numpy.random import default_rng
from Decision_Tree_Class import DecisionTree
from evaluation import evaluate, averaging
import matplotlib.pyplot as plt 

def k_fold_split(n_splits, n_instances, random_generator=default_rng()):

    # generate a random permutation of indices from 0 to n_instances
    shuffled_indices = random_generator.permutation(n_instances)

    # split shuffled indices into almost equal sized splits
    split_indices = np.array_split(shuffled_indices, n_splits)

    return split_indices

def train_test_k_fold(n_folds, n_instances, random_generator=default_rng()):
    """ Generate train and test indices at each fold.

    Args:
        n_folds (int): Number of folds
        n_instances (int): Total number of instances
        random_generator (np.random.Generator): A random generator

    Returns:
        list: a list of length n_folds. Each element in the list is a list (or tuple)
            with two elements: a numpy array containing the train indices, and another
            numpy array containing the test indices.
    """

    # split the dataset into k splits
    split_indices = k_fold_split(n_folds, n_instances, random_generator)

    folds = []
    for k in range(n_folds):
        # TODO: Complete this
        # take the splits from split_indices and keep the k-th split as testing
        # and concatenate the remaining splits for training

        test_indices = split_indices[k]
        train_indices = np.concatenate([split_indices[i] for i in range(n_folds) if i!=k])
        folds.append([train_indices, test_indices])

    return folds

def main(n_folds = 10):
    print("Decision Tree Coursework")
    
    seed = 60013
    rng = default_rng(seed)

    #Load the data
    clean_dataset = np.loadtxt('wifi_db/clean_dataset.txt')
    noisy_data = np.loadtxt('wifi_db/noisy_dataset.txt')

    #Choose the dataset to use
    raw_data = noisy_data
    rng.shuffle(raw_data)
    all_results = []

    for train_indices, test_indices in train_test_k_fold(n_folds, len(raw_data), rng):
            # set up the dataset for this fold
        train_dataset = raw_data[train_indices, :]

        test_dataset = raw_data[test_indices, :]

        
        dt = DecisionTree()
        dt.train(train_dataset)
        results = evaluate(test_dataset, dt) 
        all_results.append(results)

    # Save the decision tree for each fold
    dt.visualize_tree3()
    plt.savefig(f'decision_tree_fold_sans_pruning.png')
    plt.show()
        
    average_results = averaging(all_results)
    print(average_results)
    
main()

