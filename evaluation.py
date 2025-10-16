import numpy as np
import matplotlib.pyplot as plt

def confusion_matrix(y_true, y_pred):
    """
    Compute the confusion matrix to evaluate the accuracy of a classification.

    Parameters
    ----------
    y_true (np.ndarray): true labels of the data (groundtruth)
    y_pred (np.ndarray): predicted labels

    Returns
    -------
    C (np.ndarray)
    """

    #Initialize the confusion matrix
    classes = np.unique(y_true)
    n_classes = classes.shape[0]
    C = np.zeros((n_classes, n_classes), dtype=int)
    #Fill the confusion matrix
    for i in range(n_classes):
        for j in range(n_classes):
            C[i, j] = np.sum((y_true == classes[i]) & (y_pred == classes[j]))
            
    return C

def accuracy(y_true, y_pred):
    """
    Compute the accuracy given the ground truth and predictions

    Args:
    y_true (np.ndarray): the correct ground truth standard labels
    y_pred (np.ndarray): the predicted labels

    Returns:
    float : the accuracy
    """
    
    try:
        return np.sum(y_true == y_pred) / len(y_true)
    except ZeroDivisionError:
        return 0


def accuracy_from_confusion_matrix(C):
    """
    Compute the accuracy given the confusion matrix (TP+TN / (P+N))

    Args:
    C (np.ndarray): the confusion matrix

    Returns:
    float : the accuracy
    """
    try:
        total_population = np.sum(C)
        total_good_pred = 0
        for i in range(C.shape[0]):
            total_good_pred += C[i,i]
        return (total_good_pred/total_population)
    except ZeroDivisionError:
        return 0
    
def recall(C):
    """
    Compute the recall for each class (TP/(TP+FN)) given the confusion matrix
    
    Args:
    C (np.ndarray): the confusion matrix
    
    Returns : list of recall for each class
    """
    recalls = []
    n_classes = C.shape[0]
    
    for i in range(n_classes):
        TP = C[i, i]
        #Sum on row i to get all actual class i
        FN = np.sum(C[i, :]) - TP
        try:
            recalls.append(TP / (TP + FN))
        except ZeroDivisionError:
            recalls.append(0)
    
    return recalls

def precision_rates(C):
    """
    Compute tHe precision rate for each class (TP/(TP+FP)) given the confusion matrix
    
    Args:
    C (np.ndarray): the confusion matrix
    
    Returns : list of precision rate for each class
    """
    
    precisions = []
    n_classes = C.shape[0]
    
    for i in range(n_classes):
        TP = C[i, i]
        #Sum on column i to get all predicted as class i
        FP = np.sum(C[:, i]) - TP
        try:
            precisions.append(TP / (TP + FP))
        except ZeroDivisionError:
            precisions.append(0)
    
    return precisions

def f1_scores(precisions, recalls):
    """
    Compute the f1 score for each class given the precision and recall rates
    
    Args:
    precisions (list): list of precision rates for each class
    recalls (list): list of recall rates for each class
    
    Returns : list of f1 scores for each class
    """
    
    f1s = []
    n_classes = len(precisions)
    
    for i in range(n_classes):
        try:
            f1s.append(2 * (precisions[i] * recalls[i]) / (precisions[i] + recalls[i]))
        except ZeroDivisionError:
            f1s.append(0)
    
    return f1s

def visualize_confusion_matrix(C, class_names):
    """
    Visualize the confusion matrix using matplotlib

    Args:
    C (np.ndarray): the confusion matrix
    class_names (list): list of class names
    """
    plt.figure(figsize=(8, 6))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    
def evaluate(test_db,trained_tree):
    """
    Evaluate the performance of a trained decision tree on a test dataset

    Args:
    test_db (np.ndarray): the test dataset
    trained_tree (DecisionTree): the trained decision tree

    Returns:
    dict : a dictionary containing accuracy, recall, precision, f1 scores and confusion matrix
    """
    
    x_test = test_db[:, :-1]
    y_true = test_db[:, -1]
    
    y_pred = trained_tree.predict(x_test)
    
    C = confusion_matrix(y_true, y_pred)
    acc = accuracy_from_confusion_matrix(C)
    rec = recall(C)
    prec = precision_rates(C)
    f1 = f1_scores(prec, rec)
    visualize_confusion_matrix(C, class_names=np.unique(y_true).tolist())
    
    results = {
        'confusion_matrix': C,
        'accuracy': acc,
        'recall': rec,
        'precision': prec,
        'f1_scores': f1
    }
    
    return results

