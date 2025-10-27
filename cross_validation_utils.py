import numpy as np


def k_fold_split(n_splits, n_instances, random_generator):
    # generate a random permutation of indices from 0 to n_instances
    shuffled_indices = random_generator.permutation(n_instances)

    # split shuffled indices into almost equal sized splits
    split_indices = np.array_split(shuffled_indices, n_splits)

    return split_indices


def train_test_k_fold(n_folds, n_instances, random_generator):
    # Split the dataset into k splits
    split_indices = k_fold_split(n_folds, n_instances, random_generator)

    folds = []

    for k in range(n_folds):
        # Get test and train indices
        test_indices = split_indices[k]
        train_indices = np.concatenate(
            [split_indices[i] for i in range(n_folds) if i != k]
        )
        folds.append([train_indices, test_indices])

    return folds
