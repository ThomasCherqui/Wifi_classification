import numpy as np
from numpy.random import default_rng
from Decision_Tree_Class import DecisionTree
from evaluation import evaluate, averaging



def main():
    print("Decision Tree Coursework")
    
    seed = 60013
    rng = default_rng(seed)

    #Load the data
    clean_dataset = np.loadtxt('wifi_db/clean_dataset.txt')
    noisy_data = np.loadtxt('wifi_db/noisy_dataset.txt')

    #choose the dataset to use
    raw_data = clean_dataset
    rng.shuffle(raw_data)
    

    k = 10

    num_samples = len(raw_data) // k
    all_results = []
    for i in range(0, k):
        start = i*num_samples
        end = start + num_samples

        test_dataset = raw_data[start:end] 
        train_dataset = np.concatenate([raw_data[:start], raw_data[end:]])
    
        print(test_dataset.shape)
        print(train_dataset.shape)

        dt = DecisionTree()
        dt.train(train_dataset)

        results = evaluate(test_dataset, dt) 
        all_results.append(results)
        
    average_results = averaging(all_results)
    print(average_results)
    
main()
