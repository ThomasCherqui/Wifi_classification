import numpy as np
from numpy.random import default_rng
from Decision_Tree_Class import DecisionTree
from evaluation import evaluate



def main():
    print("Decision Tree Coursework")
    
    seed = 60013
    rng = default_rng(seed)

    #Load the data
    clean_dataset = np.loadtxt('wifi_db/clean_dataset.txt')
    noisy_data = np.loadtxt('wifi_db/noisy_dataset.txt')

    raw_data = clean_dataset
    rng.shuffle(raw_data)
    

    k = 10

    num_samples = len(raw_data) // k

    for i in range(0, k):
        start = i*num_samples
        end = start + num_samples

        test_dataset = raw_data[start:end] 
        train_dataset = np.concatenate([raw_data[:start], raw_data[end:]])
    
        print(test_dataset)
        print(train_dataset.shape)

        dt = DecisionTree()
        dt.train(train_dataset)

 

        results = evaluate(test_dataset, dt) 
        print(results)

    



main()