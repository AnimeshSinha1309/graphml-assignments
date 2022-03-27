import numpy as np

from walk_embeddings import Graph


if __name__ == "__main__":
    g = Graph(
        number_of_nodes=8,
        edge_list=[
                    [0, 2],
                    [0, 4],
                    [0, 5],
                    [1, 4],
                    [1, 5],
                    [2, 3],
                    [2, 4],
                    [4, 5],
                    [4, 6],
                    [4, 7]
                            ],
    )
    mat = np.zeros((g.num_nodes, g.num_nodes,))
    for u, v in g.edge_list:
        mat[u, v] += 1
        mat[v, u] += 1
    mat = mat / np.sum(mat, axis=0)
    print(np.round(mat, decimals=4))

    eigenvalues, eigenvectors = np.linalg.eig(mat)
    for value, vector in zip(eigenvalues, eigenvectors):
        print(value, np.round(vector, decimals=4))
    

# \begin{bmatrix}
#     0.   & 0.25 & 1.   & 1.   & 1.   & 0.   & 0.   & 0.   \\
#     0.25 & 0.   & 0.   & 0.   & 0.   & 1.   & 0.5  & 0.5  \\
#     0.25 & 0.   & 0.   & 0.   & 0.   & 0.   & 0.   & 0.   \\
#     0.25 & 0.   & 0.   & 0.   & 0.   & 0.   & 0.   & 0.   \\
#     0.25 & 0.   & 0.   & 0.   & 0.   & 0.   & 0.   & 0.   \\
#     0.   & 0.25 & 0.   & 0.   & 0.   & 0.   & 0.   & 0.   \\
#     0.   & 0.25 & 0.   & 0.   & 0.   & 0.   & 0.   & 0.5  \\
#     0.   & 0.25 & 0.   & 0.   & 0.   & 0.   & 0.5  & 0.   \\
# \end{bmatrix}
