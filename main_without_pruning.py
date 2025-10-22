import numpy as np
import json
import matplotlib.pyplot as plt
from numpy.random import default_rng
from Decision_Tree_Class import DecisionTree
from evaluation_utils import evaluate, averaging
from cross_validation_utils import train_test_k_fold


def main(n_folds=10):
    print("Decision Tree Coursework")

    seed = 60013
    rng = default_rng(seed)

    # The user chooses the dataset
    dataset = input("Choose dataset ('clean' or 'noisy'): ").strip().lower()

    if dataset not in {"clean", "noisy"}:
        raise ValueError("Invalid dataset. Choose 'clean' or 'noisy'.")

    raw_data = np.loadtxt(f"wifi_db/{dataset}_dataset.txt")
    outfile = f"visualisations/confusion_matrix_{dataset}.png"

    # Initialize list to store evaluation metrics
    all_results = []

    for train_indices, test_indices in train_test_k_fold(n_folds, len(raw_data), rng):
        # Set up the dataset for given fold
        train_dataset = raw_data[train_indices, :]
        test_dataset = raw_data[test_indices, :]

        dt = DecisionTree()

        # Train tree on training dataset
        dt.train(train_dataset)

        # Evaluate tree on test data
        results = evaluate(test_dataset, dt)
        all_results.append(results)

    # Ask user for visualization option
    visualise = (
        input("Do you want to visualise the decision tree? (yes/no): ").strip().lower()
    )
    if visualise == "yes":
        dt.visualize_tree3(f"decision_tree_{dataset}_data.png")

    average_results = averaging(all_results, outfile)
    print(average_results)


main()
