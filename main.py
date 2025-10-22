import numpy as np
import json
from numpy.random import default_rng
from Decision_Tree_Class import DecisionTree
from evaluation_utils import evaluate, averaging
import matplotlib.pyplot as plt 

def main():
    print("Decision Tree Coursework")
    
    seed = 60013
    rng = default_rng(seed)

    #Load the data
    clean_dataset = np.loadtxt('wifi_db/clean_dataset.txt')
    noisy_data = np.loadtxt('wifi_db/noisy_dataset.txt')

    #Choose the dataset to use
    raw_data = noisy_data
    rng.shuffle(raw_data)
    
    dt = DecisionTree()
    dt.train(clean_dataset)
    print(dt.to_dict())
    dt.visualize_tree3("decision_tree_clean_data.png")

if __name__ == "__main__":
    main()