import numpy as np
import json
from numpy.random import default_rng
from Decision_Tree_Class import DecisionTree
from evaluation import evaluate, averaging
import matplotlib.pyplot as plt 

def k_fold_split(n_splits, n_instances, random_generator):

    shuffled_indices = random_generator.permutation(n_instances)
    split_indices = np.array_split(shuffled_indices, n_splits)

    return split_indices

def train_test_k_fold(n_folds, n_instances, random_generator):
    
    split_indices = k_fold_split(n_folds, n_instances, random_generator)

    folds = []

    for k in range(n_folds):

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
    clean_confusion_matrix_filepath = 'visualisations/confusion_matrix_clean.png'
    noisy_data = np.loadtxt('wifi_db/noisy_dataset.txt')
    noisy_confusion_matrix_filepath = 'visualisations/confusion_matrix_noisy.png'


    #Choose the dataset to use
    raw_data = clean_dataset #noisy_data
    outfile = clean_confusion_matrix_filepath


    all_results = []

    for train_indices, test_indices in train_test_k_fold(n_folds, len(raw_data), rng):

        train_dataset = raw_data[train_indices, :]
        test_dataset = raw_data[test_indices, :]

        dt = DecisionTree()

        #Train tree on training dataset
        dt.train(train_dataset)

        #Evaluate tree on test data
        results = evaluate(test_dataset, dt) 
        all_results.append(results)

    #dt.visualize_tree3()
    #plt.savefig(f'decision_tree_fold_sans_pruning.png')
    #plt.show()
        
    average_results = averaging(all_results, outfile)
    print(average_results)
    
main()

