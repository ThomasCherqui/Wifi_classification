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

    def visualize_tree(self, x=1, y=0.5, dx=0.2, dy=0.1, ax=None):
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(70, 40))
            ax.axis("off")

        # Determine if leaf node
        if hasattr(self, "label") and self.label is not None:
            label = f"Room {self.label} ✓"
            bbox_props = dict(boxstyle="circle,pad=0.4", fc="pink", ec="black", lw=1)
        else:
            label = f"X[{self.attribute}] ≤ {self.value:.2f}"
            bbox_props = dict(boxstyle="round,pad=0.4", fc="orange", ec="black", lw=1)

        ax.text(x, y, label, ha="center", va="center", bbox=bbox_props)

        # Plot left child if exists
        if hasattr(self, "left_branch") and self.left_branch is not None:
            ax.plot([x, x - dx], [y - 0.02, y - dy + 0.02], "k-")
            self.left_branch.visualize_tree(x - dx, y - dy, dx / 1.5, dy, ax)

        # Plot right child if exists
        if hasattr(self, "right_branch") and self.right_branch is not None:
            ax.plot([x, x + dx], [y - 0.02, y - dy + 0.02], "k-")
            self.right_branch.visualize_tree(x + dx, y - dy, dx / 1.5, dy, ax)

        # Show plot if this is the top call
        if ax is None:
            plt.show()

    def visualize_tree2(self, save_path="decision_tree_simple.png"):
        """
        Visualisation simple, claire et lisible de l'arbre de décision.
        Les nœuds de décision sont des ellipses bleues, les feuilles des cercles verts.
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse, Circle

        # ---- fonctions utilitaires ----
        def count_leaves(node):
            if node is None:
                return 0
            if node.label is not None:
                return 1
            return count_leaves(node.left_branch) + count_leaves(node.right_branch)

        positions = {}
        x_counter = [0]

        def assign_positions(node, depth=0):
            if node is None:
                return None
            assign_positions(node.left_branch, depth + 1)
            x = x_counter[0]
            x_counter[0] += 1
            positions[node] = (x, depth)
            assign_positions(node.right_branch, depth + 1)

        assign_positions(self)

        # ---- dessin ----
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.axis("off")

        def draw(node):
            x, depth = positions[node]
            y = -depth

            if node.label is not None:
                # --- Feuille : cercle vert ---
                circ = Circle((x, y), radius=0.4, color="lightgreen", ec="black")
                ax.add_patch(circ)
                ax.text(
                    x, y, str(int(node.label)), ha="center", va="center", fontsize=9
                )
            else:
                # --- Noeud de décision : ellipse bleue ---
                ell = Ellipse(
                    (x, y), width=2.2, height=0.9, color="skyblue", ec="black"
                )
                ax.add_patch(ell)
                ax.text(
                    x,
                    y,
                    f"X[{int(node.attribute)}] ≤ {node.value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

            # --- Branches ---
            if node.left_branch:
                x2, d2 = positions[node.left_branch]
                ax.plot([x, x2], [y, -d2], color="black", lw=1)
                draw(node.left_branch)
            if node.right_branch:
                x2, d2 = positions[node.right_branch]
                ax.plot([x, x2], [y, -d2], color="black", lw=1)
                draw(node.right_branch)

        draw(self)

        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print("Tree successfully saved as", save_path)

    def visualize_tree3(
        self,
        save_path="decision_tree.png",
        node_w=1.6,
        node_h=0.9,
        x_gap=0.8,
        y_step=1.6,
        dpi=200,
    ):
        """
        Trace un arbre proprement :
        - placement en in-order sans overlap horizontal,
        - centrage des parents une fois les enfants visités,
        - connecteurs clairs, sauvegarde PNG propre.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        # ---------- 1) Mesures structurelles ----------
        def count_leaves(node):
            if node is None:
                return 0
            if node.label is not None:
                return 1
            return count_leaves(node.left_branch) + count_leaves(node.right_branch)

        def max_depth(node):
            if node is None:
                return 0
            if node.label is not None:
                return 1
            return 1 + max(max_depth(node.left_branch), max_depth(node.right_branch))

        n_leaves = max(1, count_leaves(self))
        depth = max_depth(self)

        # ---------- 2) Placement via parcours in-order ----------
        positions = {}  # node → (x, depth)
        x_counter = [0.0]  # mutable compteur

        def assign_positions(node, d=0):
            """Parcours in-order: place chaque feuille dans une colonne unique"""
            if node is None:
                return None
            # gauche
            left_x = assign_positions(node.left_branch, d + 1)
            # position du nœud courant
            if node.label is not None:
                x = x_counter[0]
                x_counter[0] += 1
            else:
                # si enfants : centre entre eux
                lx = left_x
                rx = assign_positions(node.right_branch, d + 1)
                if lx is not None and rx is not None:
                    x = (lx + rx) / 2.0
                elif lx is not None:
                    x = lx
                elif rx is not None:
                    x = rx
                else:
                    x = x_counter[0]
                    x_counter[0] += 1
                positions[node] = (x, d)
                return x
            positions[node] = (x, d)
            return x

        assign_positions(self)

        # Normalisation des X → espacés régulièrement
        xs = [x for x, _ in positions.values()]
        x_min, x_max = min(xs), max(xs)
        for k, (x, d) in list(positions.items()):
            x_norm = (x - x_min) / max(1e-9, (x_max - x_min))
            positions[k] = (x_norm * (n_leaves - 1) * (1 + x_gap), d)

        # ---------- 3) Tracé ----------
        fig_w = max(10, int(n_leaves * (node_w + x_gap) * 0.6))
        fig_h = max(5, int(depth * (y_step) * 0.7))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.axis("off")

        def draw_box(ax, x, y, text, is_leaf=False):
            if is_leaf:
                circ = patches.Circle(
                    (x, y), radius=node_w / 2, fc="#90EE90", ec="black", lw=1
                )
                ax.add_patch(circ)
                ax.text(x, y, text, ha="center", va="center", fontsize=9)
                return (x, y + node_w / 2), (x, y - node_w / 2)
            else:
                rect = patches.FancyBboxPatch(
                    (x - node_w / 2, y - node_h / 2),
                    node_w,
                    node_h,
                    boxstyle="round,pad=0.12",
                    fc="#FFD580",
                    ec="black",
                    lw=1,
                )
                ax.add_patch(rect)
                ax.text(x, y, text, ha="center", va="center", fontsize=8)
                return (x, y + node_h / 2), (x, y - node_h / 2)

        def elbow(ax, x1, y1, x2, y2):
            ym = (y1 + y2) / 2.0
            ax.plot([x1, x1], [y1, ym], "k-", lw=1)
            ax.plot([x1, x2], [ym, ym], "k-", lw=1)
            ax.plot([x2, x2], [ym, y2], "k-", lw=1)

        def draw(node):
            x, depth_idx = positions[node]
            y = -depth_idx * y_step
            if node.label is not None:
                up, down = draw_box(ax, x, y, f"{int(node.label)}", is_leaf=True)
            else:
                up, down = draw_box(
                    ax,
                    x,
                    y,
                    f"X[{int(node.attribute)}] ≤ {node.value:.2f}",
                    is_leaf=False,
                )

            # enfants
            if node.left_branch:
                x2, d2 = positions[node.left_branch]
                y2 = -d2 * y_step
                elbow(ax, down[0], down[1], x2, y2 + node_h / 2)
                draw(node.left_branch)
            if node.right_branch:
                x2, d2 = positions[node.right_branch]
                y2 = -d2 * y_step
                elbow(ax, down[0], down[1], x2, y2 + node_h / 2)
                draw(node.right_branch)

        draw(self)

        xs, ys = zip(*[(x, -d * y_step) for x, d in positions.values()])
        ax.set_xlim(min(xs) - node_w, max(xs) + node_w)
        ax.set_ylim(min(ys) - y_step, y_step)
        fig.tight_layout()
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"✅ Arbre sauvegardé dans {save_path}")

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
