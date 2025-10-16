class DecisionTree:

    def __init__(self,depth=0):
        self.left_branch = None
        self.right_branch = None
        self.attribute = None
        self.value = None
        self.depth = 0
        self.label = None

    def train(self, dataset):

        #Base case
        if len(np.unique(dataset[:, -1]))== 1:
            self.label = int(np.unique(dataset[:, -1]))
            return

        else:
            self.attribute, self.value = find_split(dataset)
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