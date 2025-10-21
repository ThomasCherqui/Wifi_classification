import numpy as np
from numpy.random import default_rng
from Decision_Tree_Class import DecisionTree
from evaluation import evaluate, averaging
import matplotlib.pyplot as plt 


def main():
    print("Decision Tree Coursework \n")
    
    seed = 60013
    rng = default_rng(seed)
    k = 10 #number of folds is 10


    print("Building Tree using Clean Dataset WITH PRUNING:")

    #Load the data
    clean_dataset = np.loadtxt('wifi_db/clean_dataset.txt')

    #Choose the dataset to use
    raw_data = clean_dataset
    rng.shuffle(raw_data)


    num_samples = len(raw_data) // k

    all_results = []
    for i in range(0, k):
        start = i*num_samples
        end = start + num_samples

        #Split data into test and train
        test_dataset = raw_data[start:end] 
        train_dataset = np.concatenate([raw_data[:start], raw_data[end:]])

        #Train decision tree
        dt = DecisionTree()
        dt.train(train_dataset)
        dt.pruning(train_dataset[:,:-1], train_dataset[:,-1])

        #Produce predictions for test dataset and evaluate
        results = evaluate(test_dataset, dt, 'visualisations/confusion_matrix_clean_w_pruning.png')  

        #Append results for each fold
        all_results.append(results)


            
    average_results = averaging(all_results)
    print(average_results)


    print("\n")
    print("Building Tree using Noisy Dataset WITH PRUNING:")

    noisy_data = np.loadtxt('wifi_db/noisy_dataset.txt')

    raw_data = noisy_data
    rng.shuffle(raw_data)
    
    num_samples = len(raw_data) // k
    all_results = []
    for i in range(0, k):
        start = i*num_samples
        end = start + num_samples

        test_dataset = raw_data[start:end] 
        train_dataset = np.concatenate([raw_data[:start], raw_data[end:]])

        dt = DecisionTree()
        dt.train(train_dataset)
        dt.pruning(train_dataset[:,:-1], train_dataset[:,-1])
        results = evaluate(test_dataset, dt, 'visualisations/confusion_matrix_noisy_w_pruning.png') 
        all_results.append(results)
        


    average_results = averaging(all_results)
    print(average_results)
        
main()

