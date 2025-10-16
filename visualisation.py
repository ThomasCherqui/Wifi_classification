import matplotlib.pyplot as plt

def count_leaves(node):
    if node["leaf"]:
        return 1
    return count_leaves(node["left"]) + count_leaves(node["right"])

def estimate_text_width(text, scale=0.015):
    """Estime une largeur horizontale à réserver pour le texte."""
    return len(text) * scale

def plot_tree_smart(node, x=0.5, y=1.0, dx=0.1, dy=0.15, ax=None):
    """
    Trace un arbre de décision lisible sans superpositions.
    Espacement = fonction du nombre de feuilles + taille du texte.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.axis("off")
        total_leaves = count_leaves(node)
        plot_tree_smart(node, x=0.5, y=1.0, dx=1/(2*total_leaves), dy=dy, ax=ax)
        plt.show()
        return

    # Définir le texte du nœud
    if node["leaf"]:
        text = str(node["label"])
        color = "lightgreen"
    else:
        text = f"X{node['attribute']} < {node['value']:.2f}"
        color = "lightblue"

    text_width = estimate_text_width(text)

    # Dessine le texte
    ax.text(x, y, text,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.4", fc=color, ec="black"))

    # Cas feuille → stop
    if node["leaf"]:
        return

    # --- Espacement dynamique des enfants ---
    n_left = count_leaves(node["left"])
    n_right = count_leaves(node["right"])

    # Largeur ajustée selon le texte et la structure
    total_span = dx * (n_left + n_right) + text_width

    left_x = x - total_span / 2 + n_left * dx / 2
    right_x = x + total_span / 2 - n_right * dx / 2
    child_y = y - dy

    # Branches
    ax.plot([x, left_x], [y - 0.02, child_y + 0.02], "k-")
    ax.plot([x, right_x], [y - 0.02, child_y + 0.02], "k-")

    # Appels récursifs
    plot_tree_smart(node["left"], left_x, child_y, dx, dy, ax)
    plot_tree_smart(node["right"], right_x, child_y, dx, dy, ax)



plot_tree_smart(node)
