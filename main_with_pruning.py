import numpy as np
import json
from numpy.random import default_rng
from Decision_Tree_Class import DecisionTree
from evaluation import evaluate, averaging
import matplotlib.pyplot as plt 

def k_fold_split(n_splits, n_instances, random_generator):

    # generate a random permutation of indices from 0 to n_instances
    shuffled_indices = random_generator.permutation(n_instances)

    # split shuffled indices into almost equal sized splits
    split_indices = np.array_split(shuffled_indices, n_splits)
    
    return split_indices


def train_val_test_k_fold(n_folds, n_instances, random_generator):

    # split the dataset into k splits
    split_indices = k_fold_split(n_folds, n_instances, random_generator)

    folds = []
    for k in range(n_folds):

        test_indices = split_indices[k]

        # Need validation set to do pruning
        val_indices = split_indices[(k+1)%n_folds]
        train_indices = np.concatenate([split_indices[i] for i in range(n_folds) if (i!=k and i !=(k+1)%n_folds)])

        folds.append([train_indices, val_indices, test_indices])

    return folds

def main(n_folds = 10):
    print("Decision Tree Coursework: With Pruning")
    
    seed = 60013
    rng = default_rng(seed)

    #Load the data
    clean_dataset = np.loadtxt('wifi_db/clean_dataset.txt')
    clean_confusion_matrix_filepath = 'visualisations/confusion_matrix_clean_with_pruning.png'
    noisy_data = np.loadtxt('wifi_db/noisy_dataset.txt')
    noisy_confusion_matrix_filepath = 'visualisations/confusion_matrix_noisy_with_pruning.png'


    #Choose the dataset to use
    raw_data = noisy_data
    outfile = noisy_confusion_matrix_filepath

    #Initialise list to store evalution metrics
    all_results = []

    for train_indices, val_indices, test_indices in train_val_test_k_fold(n_folds, len(raw_data), rng):

        # Set up the dataset for given fold
        train_dataset = raw_data[train_indices, :]
        validation_dataset = raw_data[val_indices, :]
        test_dataset = raw_data[test_indices, :]
        
        
        dt = DecisionTree()

        #Train tree on the train dataset
        dt.train(train_dataset)

        #Use validation set to prune the tree
        dt.pruning(validation_dataset[:,:-1], validation_dataset[:,-1])

        #Evaluate the tree using the test data
        results = evaluate(test_dataset, dt) 

        all_results.append(results)

    # Save the decision tree for each fold
    #plt.savefig(f'decision_tree_fold_avec_pruning.png')
    #dt.visualize_tree3()
    #plt.show()
        
    average_results = averaging(all_results, outfile)
    print(average_results)
    
main()

