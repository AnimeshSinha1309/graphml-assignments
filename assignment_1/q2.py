import numpy as np
import networkx as nx
from matplotlib import pyplot as plt


def check_isomorphism(g_1, g_2, iterations=5):
    """
    Color refinement process on WL kernels to check if two graphs are isomorphic
    """
    color_map_1 = np.zeros(shape=g_1.number_of_nodes(), dtype=np.int32)
    color_map_2 = np.zeros(shape=g_2.number_of_nodes(), dtype=np.int32)
    not_isomorphic = False
    plt.figure(figsize=(25, 10))

    for iteration in range(iterations):
        neighbors_1 = [
            np.sort([color_map_1[neighbor] for neighbor in list(g_1.neighbors(node))])
            for node in g_1.nodes()
        ]
        color_map_1 = np.array(
            [hash(",".join(list(map(str, neighbor)))) % 100 for neighbor in neighbors_1]
        )

        neighbors_2 = [
            np.sort([color_map_2[neighbor] for neighbor in list(g_2.neighbors(node))])
            for node in g_2.nodes()
        ]
        color_map_2 = np.array(
            [hash(",".join(list(map(str, neighbor)))) % 100 for neighbor in neighbors_2]
        )

        if not np.all(np.sort(color_map_1) == np.sort(color_map_2)):
            print(color_map_1)
            print(color_map_2)
            not_isomorphic = True

        plt.subplot(2, iterations, iteration + 1)
        nx.draw(g_2, node_color=color_map_1)
        plt.title(
            "Graph 1 \n"
            + ("Not Isomorphic" if not_isomorphic else "Possibly Isomorphs")
        )
        plt.subplot(2, iterations, iteration + iterations + 1)
        nx.draw(g_2, node_color=color_map_2)
        plt.title(
            "Graph 2 \n"
            + ("Not Isomorphic" if not_isomorphic else "Possibly Isomorphs")
        )

    plt.show()
    return not not_isomorphic


if __name__ == "__main__":
    g_1 = nx.Graph()
    g_1.add_nodes_from([0, 1, 2, 3, 4, 5])
    g_1.add_edges_from(
        [
            (0, 2),
            (0, 4),
            (0, 5),
            (1, 4),
            (1, 5),
            (2, 3),
            (2, 4),
            (4, 5),
        ]
    )

    g_2 = nx.Graph()
    g_2.add_nodes_from([0, 1, 2, 3, 4, 5])
    g_2.add_edges_from(
        [
            (5, 0),
            (5, 2),
            (5, 4),
            (1, 2),
            (1, 4),
            (0, 3),
            (0, 2),
            (2, 4),
        ]
    )
    check_isomorphism(g_1, g_2)

    g_3 = nx.Graph()
    g_3.add_nodes_from([0, 1, 2, 3, 4, 5])
    g_3.add_edges_from(
        [
            (0, 2),
            (0, 4),
            (0, 5),
            (1, 5),
            (2, 4),
            (4, 5),
            (4, 3),
            (1, 2),
        ]
    )
    check_isomorphism(g_1, g_3)
