import numpy as np
import json
from numpy.random import default_rng
from Decision_Tree_Class import DecisionTree
from evaluation import evaluate, averaging
import matplotlib.pyplot as plt 

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

    #choose the dataset to use
    raw_data = clean_dataset
    np.random.shuffle(raw_data)
    

    k = 10

    num_samples = len(raw_data) // k
    all_results = []
    for i in range(0, k):
        start = i*num_samples
        end = start + num_samples

        test_dataset = raw_data[start:end] 
        train_dataset = np.concatenate([raw_data[:start], raw_data[end:]])

        dt = DecisionTree()
        dt.train(train_dataset)
        results = evaluate(test_dataset, dt) 
        all_results.append(results)

        # Save the decision tree for each fold
        dt.visualize_tree()
        plt.savefig(f'decision_tree_fold_{i+1}.png')
        plt.show()

    average_results = averaging(all_results)
    print(average_results)
    
main()

