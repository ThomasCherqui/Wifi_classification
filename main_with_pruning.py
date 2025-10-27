import numpy as np
from numpy.random import default_rng
from Decision_Tree_Class import DecisionTree
from evaluation_utils import evaluate, averaging
import matplotlib.pyplot as plt
from cross_validation_utils import train_test_k_fold


def main(n_folds=10,n_inner_folds=9):
    print("Decision Tree Coursework: With Pruning")

    seed = 60013
    rng = default_rng(seed)

    # The user chooses the dataset
    dataset = input("Choose dataset ('clean' or 'noisy'): ").strip().lower()

    if dataset not in {"clean", "noisy"}:
        raise ValueError("Invalid dataset. Choose 'clean' or 'noisy'.")

    raw_data = np.loadtxt(f"wifi_db/{dataset}_dataset.txt")
    outfile = f"visualisations/confusion_matrix_{dataset}.png"

    # Initialise list to store evaluation metrics
    all_results = []

    #nested cross validation
    for train_and_validation_indices, test_indices in train_test_k_fold(n_folds, len(raw_data), rng):
        test_dataset = raw_data[test_indices, :]
        train_and_validation_dataset = raw_data[train_and_validation_indices,:]

        
        for train_indices, validation_indices in train_test_k_fold(n_inner_folds,len(train_and_validation_dataset),rng):
            train_dataset = train_and_validation_dataset[train_indices,:]
            validation_dataset = train_and_validation_dataset[validation_indices,:]
            
            dt = DecisionTree()

            # Train tree on the train dataset
            dt.train(train_dataset)

            # Use validation set to prune the tree
            dt.pruning(validation_dataset[:, :-1], validation_dataset[:, -1])
            
            # Evaluate the tree using the test data
            results = evaluate(test_dataset, dt)
            
            all_results.append(results)
            
    # Ask user for visualization option
    visualise = (
        input("Do you want to visualise the last decision tree? (yes/no): ").strip().lower()
    )
    if visualise == "yes":
        dt.visualize_tree3(f"decision_tree_{dataset}_data.png")
        
    average_results = averaging(all_results, outfile)
    print("average results", average_results)


main()
