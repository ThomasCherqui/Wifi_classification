import numpy as np
import matplotlib.pyplot as plt

class DecisionTree:

    def __init__(self,depth=0):
        self.left_branch = None
        self.right_branch = None
        self.attribute = None
        self.value = None
        self.depth = depth
        self.label = None


    def calculate_entropy(self, subset):

        len_dataset = len(subset[:, -1])
        classes, counts = np.unique(subset[:, -1], return_counts=True)

        entropy = 0
        for i in range(len(classes)):
            p = counts[i] / len_dataset
            entropy -= p * np.log2(p)   # ← signe négatif + log base 2

        return entropy

    def find_split(self, data):

        n_attributes = data.shape[1]-1
        
        # Matrix dimension (6 x 3) to store for each attribute: the attribute, entropy of split, the value to split on
        best_attribute_matrix = np.zeros((n_attributes,3))
       
        # Loop over all the attributes
        for attribute in range(n_attributes):

            # Get values and class for given attribute
            values = np.array(data[:, [attribute,-1]]) 
            
            # Sort the values first by the label and then the values (resolves edge case)
            values = values[values[:,1].argsort()]
            values = values[values[:,0].argsort(kind='stable')]

            # Initialise list to keep track of all locations where we have a class change between values
            class_change = []

            # Loop over all samples
            for i in range(len(values)-1): 
                # Check if class changes between two consecutive rows
                if values[i,-1] != values[i+1,-1]:
                    # Check if value changes between two consecutive rows --> can't split if val doesn't change
                    if values[i, 0] != values[i+1, 0]:
                        class_change.append(i)

            # If we do not find anything in the list where the class and value changes, need to flip the sorting array to descending order of classes
            # this ensures we don't miss the case when our class is at the end of the list
            if len(class_change) == 0:
                values = np.array(data[:, [attribute,-1]]) 
                values = values[values[:,1].argsort()[::-1]]
                values = values[values[:,0].argsort(kind='stable')]
                # Initialise list to keep track of all locations where we have a class change between values
                class_change = []

                # Loop over all samples
                for i in range(len(values)-1): 
                # Check if class changes between two consecutive rows
                    if values[i,-1] != values[i+1,-1]:
                        # Check if value changes between two consecutive rows --> can't split if val doesn't change
                        if values[i, 0] != values[i+1, 0]:
                            class_change.append(i)
            
            # If still zero set to infinity so it chooses another attribute
            if len(class_change) == 0:
                best_attribute_matrix[attribute,:] = [attribute, np.inf, np.inf]
                continue

            # Initialise array to contain entropy of potential splits
            remainder_array = np.zeros((len(class_change),2))

            # Loop over each index where the class and value change, calculate the entropy of the split
            for indices in class_change : 
                remainder = len(values[:indices+1,:])/len(values) * self.calculate_entropy(values[:indices+1,:]) + len(values[indices+1:,:])/len(values) * self.calculate_entropy(values[indices+1:,:])
                
                #Store the location of the split and the entropy of the 'remainder'
                remainder_array[class_change.index(indices),:] = [int(indices), remainder]

            # Get the minimum remainder --> largest information gain overall
            best_split_index_reminder_array = np.argmin(remainder_array[:,1])

            # Get index of best split
            best_split_info = remainder_array[best_split_index_reminder_array,:]

            # Calculate the threshold value as the mean between the split point and the next datapoint in the sorted values
            threshold_value = (values[int(best_split_info[0]),0] + values[int(best_split_info[0]+1),0]) / 2

            # Store the lowest entropy and needed value split for each attribute
            best_attribute_matrix[attribute,:] = [attribute, best_split_info[1],threshold_value]


        # Find the best information gain as the minimum of the entropies over the attributes
        best_attribute_index = np.argmin(best_attribute_matrix[:,1])
        

        return best_attribute_matrix[best_attribute_index,[0,2]]
    
    def train(self, dataset):

        #Base case
        if len(np.unique(dataset[:, -1]))== 1:
            self.label = int(np.unique(dataset[:, -1]))
            return

        else:
            self.attribute, self.value = self.find_split(dataset)
            self.attribute = int(self.attribute)

           
            #Split the dataset in two
            l_dataset = np.array([x for x in dataset if x[int(self.attribute)] <= self.value])
            r_dataset = np.array([x for x in dataset if x[int(self.attribute)] > self.value])

              # If one of the datasets is empty, create a leaf node with the majority class --> ensures we do not infinitely recurr
            if len(l_dataset) == 0 or len(r_dataset) == 0:
                self.label = np.bincount(dataset[:, -1].astype(int)).argmax()
                return self.label

            self.left_branch = DecisionTree(self.depth + 1)
            self.left_branch.train(l_dataset)
            self.right_branch = DecisionTree(self.depth + 1)
            self.right_branch.train(r_dataset)


    def predict(self,test):
        if self.label is not None:
            return self.label
        if test[self.attribute] > self.value:
            return self.right_branch.predict(test)
        else:
            return self.left_branch.predict(test)
        
    def to_dict(self):
        # Convert the decision tree to a dictionary for easier visualization
            if self.label is not None:
                return {"label": int(self.label)}
            return {
                "attribute": int(self.attribute),
                "value": float(self.value),
                "left": self.left_branch.to_dict() if self.left_branch else None,
                "right": self.right_branch.to_dict() if self.right_branch else None
            }
    
    def visualize_tree(self, x=1, y=0.5, dx=0.1, dy=0.1, ax=None):
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(70, 40))
            ax.axis('off')

        # Determine if leaf node
        if hasattr(self, 'label') and self.label is not None:
            label = f"Room {self.label} ✓"
            bbox_props = dict(boxstyle="circle,pad=0.4", fc="pink", ec="black", lw=1)
        else:
            label = f"X[{self.attribute}] ≤ {self.value:.2f}"
            bbox_props = dict(boxstyle="round,pad=0.4", fc="orange", ec="black", lw=1)

        ax.text(x, y, label, ha='center', va='center', bbox=bbox_props)

        # Plot left child if exists
        if hasattr(self, 'left_branch') and self.left_branch is not None:
            ax.plot([x, x-dx], [y - 0.02, y - dy + 0.02], 'k-')
            self.left_branch.visualize_tree(x - dx, y - dy, dx / 1.5, dy, ax)

        # Plot right child if exists
        if hasattr(self, 'right_branch') and self.right_branch is not None:
            ax.plot([x, x + dx], [y - 0.02, y - dy + 0.02], 'k-')
            self.right_branch.visualize_tree(x + dx, y - dy, dx / 1.5, dy, ax)

        # Show plot if this is the top call
        if ax is None:
            plt.show()
            
    def pruning(self, X_val, y_val):

        if self.label is not None:
            return self.label

        if self.left_branch is not None:
            self.left_branch.pruning(X_val, y_val)
        if self.right_branch is not None:
            self.right_branch.pruning(X_val, y_val)

        if (self.left_branch is not None and self.right_branch is not None and
            self.left_branch.label is not None and self.right_branch.label is not None):

            y_pred_before = np.array([self.predict(x) for x in X_val])
            acc_before = np.mean(y_pred_before == y_val)

            y_majority = np.bincount(y_val.astype(int)).argmax()
            
            backup_left = self.left_branch
            backup_right = self.right_branch
            backup_label = self.label

            self.left_branch = None
            self.right_branch = None
            self.label = y_majority

            y_pred_after = np.array([self.predict(x) for x in X_val])
            acc_after = np.mean(y_pred_after == y_val)

            if acc_after < acc_before:
                self.left_branch = backup_left
                self.right_branch = backup_right
                self.label = backup_label
