import numpy as np
import matplotlib.pyplot as plt


class DecisionTree:

    def __init__(self, depth=0):
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
            entropy -= p * np.log2(p)  # ← signe négatif + log base 2

        return entropy

    def find_split(self, data):

        n_attributes = data.shape[1] - 1

        # Matrix dimension (6 x 3) to store for each attribute: the attribute, entropy of split, the value to split on
        best_attribute_matrix = np.zeros((n_attributes, 3))

        # Loop over all the attributes
        for attribute in range(n_attributes):

            # Get values and class for given attribute
            values = np.array(data[:, [attribute, -1]])

            # Sort the values first by the label and then the values (resolves edge case)
            values = values[values[:, 1].argsort()]
            values = values[values[:, 0].argsort(kind="stable")]

            # Initialise list to keep track of all locations where we have a class change between values
            class_change = []

            # Loop over all samples
            for i in range(len(values) - 1):
                # Check if class changes between two consecutive rows
                if values[i, -1] != values[i + 1, -1]:
                    # Check if value changes between two consecutive rows --> can't split if val doesn't change
                    if values[i, 0] != values[i + 1, 0]:
                        class_change.append(i)

            # If we do not find anything in the list where the class and value changes, need to flip the sorting array to descending order of classes
            # this ensures we don't miss the case when our class is at the end of the list
            if len(class_change) == 0:
                values = np.array(data[:, [attribute, -1]])
                values = values[values[:, 1].argsort()[::-1]]
                values = values[values[:, 0].argsort(kind="stable")]
                # Initialise list to keep track of all locations where we have a class change between values
                class_change = []

                # Loop over all samples
                for i in range(len(values) - 1):
                    # Check if class changes between two consecutive rows
                    if values[i, -1] != values[i + 1, -1]:
                        # Check if value changes between two consecutive rows --> can't split if val doesn't change
                        if values[i, 0] != values[i + 1, 0]:
                            class_change.append(i)

            # If still zero set to infinity so it chooses another attribute
            if len(class_change) == 0:
                best_attribute_matrix[attribute, :] = [attribute, np.inf, np.inf]
                continue

            # Initialise array to contain entropy of potential splits
            remainder_array = np.zeros((len(class_change), 2))

            # Loop over each index where the class and value change, calculate the entropy of the split
            for indices in class_change:
                remainder = len(values[: indices + 1, :]) / len(
                    values
                ) * self.calculate_entropy(values[: indices + 1, :]) + len(
                    values[indices + 1 :, :]
                ) / len(
                    values
                ) * self.calculate_entropy(
                    values[indices + 1 :, :]
                )

                # Store the location of the split and the entropy of the 'remainder'
                remainder_array[class_change.index(indices), :] = [
                    int(indices),
                    remainder,
                ]

            # Get the minimum remainder --> largest information gain overall
            best_split_index_reminder_array = np.argmin(remainder_array[:, 1])

            # Get index of best split
            best_split_info = remainder_array[best_split_index_reminder_array, :]

            # Calculate the threshold value as the mean between the split point and the next datapoint in the sorted values
            threshold_value = (
                values[int(best_split_info[0]), 0]
                + values[int(best_split_info[0] + 1), 0]
            ) / 2

            # Store the lowest entropy and needed value split for each attribute
            best_attribute_matrix[attribute, :] = [
                attribute,
                best_split_info[1],
                threshold_value,
            ]

        # Find the best information gain as the minimum of the entropies over the attributes
        best_attribute_index = np.argmin(best_attribute_matrix[:, 1])

        return best_attribute_matrix[best_attribute_index, [0, 2]]

    def train(self, dataset):

        # Base case
        if len(np.unique(dataset[:, -1])) == 1:
            self.label = int(np.unique(dataset[:, -1]))
            return

        else:
            self.attribute, self.value = self.find_split(dataset)
            self.attribute = int(self.attribute)

            # Split the dataset in two
            l_dataset = np.array(
                [x for x in dataset if x[int(self.attribute)] <= self.value]
            )
            r_dataset = np.array(
                [x for x in dataset if x[int(self.attribute)] > self.value]
            )

            # If one of the datasets is empty, create a leaf node with the majority class --> ensures we do not infinitely recurr
            if len(l_dataset) == 0 or len(r_dataset) == 0:
                self.label = np.bincount(dataset[:, -1].astype(int)).argmax()
                return self.label

            self.left_branch = DecisionTree(self.depth + 1)
            self.left_branch.train(l_dataset)
            self.right_branch = DecisionTree(self.depth + 1)
            self.right_branch.train(r_dataset)

    def predict(self, test):
        if self.label is not None:
            return self.label
        if test[self.attribute] > self.value:
            return self.right_branch.predict(test)
        else:
            return self.left_branch.predict(test)
        
    def pruning(self, X_val, y_val):

        if self.label is not None:
            return self.label

        if self.left_branch is not None:
            l_dataset_X = X_val[X_val[:, int(self.attribute)] <= self.value]
            l_dataset_y = y_val[X_val[:, int(self.attribute)] <= self.value]

            if len(l_dataset_y) != 0:
                self.left_branch.pruning(l_dataset_X, l_dataset_y)

        if self.right_branch is not None:
            r_dataset_X = X_val[X_val[:, int(self.attribute)] > self.value]
            r_dataset_y = y_val[X_val[:, int(self.attribute)] > self.value]

            if len(r_dataset_y) != 0:
                self.right_branch.pruning(r_dataset_X, r_dataset_y)

        y_pred_before = np.array([self.predict(x) for x in X_val])
        acc_before = np.mean(y_pred_before == y_val)

        y_majority = np.bincount(y_val.astype(int)).argmax()
        new_labels = np.repeat(y_majority, len(y_val))
        acc_after = np.mean(new_labels == y_val)

        if acc_after >= acc_before:
            self.left_branch = None
            self.right_branch = None
            self.label = y_majority

    def to_dict(self):
        # Convert the decision tree to a dictionary for easier visualization
        if self.label is not None:
            return {"label": int(self.label)}
        return {
            "attribute": int(self.attribute),
            "value": float(self.value),
            "left": self.left_branch.to_dict() if self.left_branch else None,
            "right": self.right_branch.to_dict() if self.right_branch else None,
        }

    def visualize_tree(self, save_path="decision_tree_simple.png"):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse, Circle

        positions = {}
        x_counter = [0]

        def assign_positions(node, depth=0):
            if node is None:
                return None
            if node.label is not None:
                x = x_counter[0]
                x_counter[0] += 1
                positions[node] = (x, depth)
                return x
            lx = assign_positions(node.left_branch, depth + 1)
            rx = assign_positions(node.right_branch, depth + 1)
            if lx is None or rx is None:
                x = lx if rx is None else rx
            else:
                x = (lx + rx) / 2
            positions[node] = (x, depth)
            return x

        assign_positions(self)

        fig, ax = plt.subplots(figsize=(22, 12))
        ax.axis("off")

        def draw(node):
            x, depth = positions[node]
            y = -depth * 2

            if node.label is not None:
                circ = Circle((x, y), radius=0.45, color="lightgreen", ec="black")
                ax.add_patch(circ)
                ax.text(x, y, str(int(node.label)), ha="center", va="center", fontsize=9)
            else:
                label_text = f"X[{int(node.attribute)}] ≤ {node.value:.1f}"
                width = max(1.8, 0.2 * len(label_text))
                ell = Ellipse((x, y), width=width, height=0.9, color="skyblue", ec="black")
                ax.add_patch(ell)
                ax.text(x, y, label_text, ha="center", va="center", fontsize=8)

            if node.left_branch:
                x2, d2 = positions[node.left_branch]
                ax.plot([x, x2], [y-0.5, -d2 * 2+0.5], color="black", lw=1)
                draw(node.left_branch)
            if node.right_branch:
                x2, d2 = positions[node.right_branch]
                ax.plot([x, x2], [y-0.5, -d2 * 2+0.5], color="black", lw=1)
                draw(node.right_branch)

        draw(self)

        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print("Tree successfully saved as", save_path)