import numpy as np

class DecisionTree:

    def __init__(self,depth=0):
        self.left_branch = None
        self.right_branch = None
        self.attribute = None
        self.value = None
        self.depth = 0
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
        
        best_attribute_matrix = np.zeros((data.shape[1]-1,3))

        for attribute in range(data.shape[1]-1):
            values = np.array(data[:, [attribute,-1]]) #get only the attribute and class columns
            values = values[values[:,0].argsort()]
            class_change = []

            for i in range(len(values)-1): #check if class changes between two consecutive rows
                if values[i,-1] != values[i+1,-1]:
                    class_change.append(i)

        # print(class_change)

            remainder_array = np.zeros((len(class_change),2))
            for indices in class_change : #for each index where the class changes, calculate the entropy of the split
                remainder = len(values[:indices+1,:])/len(values) * self.calculate_entropy(values[:indices+1,:]) + len(values[indices+1:,:])/len(values) * self.calculate_entropy(values[indices+1:,:])
                remainder_array[class_change.index(indices),:] = [int(indices), remainder]

            best_split_index_reminder_array = np.argmin(remainder_array[:,1])

            best_split_info = remainder_array[best_split_index_reminder_array,:]

            threshold_value = (values[int(best_split_info[0]),0] + values[int(best_split_info[0]+1),0]) / 2

        #  print("best split info is", best_split_info)

            best_attribute_matrix[attribute,:] = [attribute, best_split_info[1],threshold_value]


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
            l_dataset = np.array([x for x in dataset if x[int(self.attribute)] < self.value])
            r_dataset = np.array([x for x in dataset if x[int(self.attribute)] >= self.value])

            # If one of the datasets is empty, create a leaf node with the majority class
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