# Coursework 1 - Decision Tree


| File                        | Description                                                                |
| --------------------------- | -------------------------------------------------------------------------- |
| `cross_validation_utils.py` | Contains functions for splitting data into folds for cross-validation.     |
| `decision_tree_class.py`    | Implements the Decision Tree class, including support for pruning and visualization. |
| `evaluation.py`             | Functions to evaluate model performance (e.g. accuracy, confusion matrix). |
| `main_without_pruning.py`   | Main script to run the Decision Tree **without pruning**.                  |
| `main_with_pruning.py`      | Main script to run the Decision Tree **with pruning**.                     |


## Choose the dataset
Inside each main file, you will find the following lines:

   *##Choose the dataset to use*
    *# raw_data, outfile = clean_dataset, clean_confusion_matrix_filepath*
    *raw_data, outfile = noisy_data, noisy_confusion_matrix_filepath*

Uncomment the dataset you want to use and comment the other.

## Run with or without pruning
Run *python main_without_pruning.py* or *python main_with_pruning.py*

## Outputs
After running the script, the average cross-validation metrics will be printed:

Accuracy

Recall

Precision

F1 Score

The average confusion matrix will be saved in the visualisations/ folder.

## Bonus : visualisation
To generate and save a figure of the trained Decision Tree (without pruning), run: *python main.py*

This will save a .png of the tree trained on the selected dataset.