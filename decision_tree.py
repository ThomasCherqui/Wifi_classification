import numpy as np
from numpy.random import default_rng
from Decision_Tree_Class import DecisionTree




def split_dataset(x, y, test_proportion, random_generator=default_rng()):

    shuffled_indices = random_generator.permutation(len(x))
    n_test = round(len(x) * test_proportion)
    n_train = len(x) - n_test
    x_train = x[shuffled_indices[:n_train]]
    y_train = y[shuffled_indices[:n_train]]
    x_test = x[shuffled_indices[n_train:]]
    y_test = y[shuffled_indices[n_train:]]
    train_dataset = np.column_stack((x_train, y_train))
    test_dataset = np.column_stack((x_test, y_test))

    return train_dataset, test_dataset



def main():
    print("Decision Tree Coursework")
    
    seed = 60012
    rg = default_rng(seed)

    #Load the data
    clean_dataset = np.loadtxt('wifi_db/clean_dataset.txt')
    noisy_data = np.loadtxt('wifi_db/noisy_dataset.txt')

    raw_data = clean_dataset
    
    classes = np.unique(raw_data[:, -1])
    x = raw_data[:, :-1]
    y = raw_data[:, -1]

    train_dataset, test_dataset = split_dataset(x, y, test_proportion=0.3, random_generator=rg)

    dt = DecisionTree()
    dt.train(train_dataset)
    tree_dict = dt.to_dict()
main()